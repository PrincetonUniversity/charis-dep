
import cython
from cython.parallel import prange, parallel
import numpy as np


@cython.wraparound(False)
@cython.boundscheck(False)

def allcutouts(double [:, :] im, double [:, :] isig, long [:, :] x, 
               long [:, :] y, long [:] indx, double [:, :, :] psflets, 
               int dx=3, int maxproc=4):

    cdef int maxsize, nlam, ix, iy, x0, x1, y0, y1, dx0, dy0, xdim, ydim
    cdef int ii, j, k, nlens, n, jj

    # How large must the arrays be? 
    
    #nlens = x.shape[1]
    nlens = indx.shape[0]
    nlam = psflets.shape[0]
    xdim = im.shape[1]
    ydim = im.shape[0]

    size_np = np.zeros((nlens), np.int64)
    cdef long [:] size = size_np

    with nogil, parallel(num_threads=maxproc):
        for jj in prange(nlens, schedule='dynamic'):
            ii = indx[jj]

            x0 = xdim + 1
            x1 = -1
            y0 = ydim + 1
            y1 = -1
            for j in range(nlam):
                if x[j, ii] > x1:
                    x1 = x[j, ii]
                if x[j, ii] < x0:
                    x0 = x[j, ii]
                if y[j, ii] > y1:
                    y1 = y[j, ii]
                if y[j, ii] < y0:
                    y0 = y[j, ii]

            y0 = y0 - dx
            x0 = x0 - dx
            y1 = y1 + dx + 1
            x1 = x1 + dx + 1
            dy0 = y1 - y0
            dx0 = x1 - x0
            
            size[jj] = dy0*dx0

    maxsize = np.amax(size_np)
    A_np = np.zeros((nlens, maxsize, nlam))
    b_np = np.zeros((nlens, maxsize))

    cdef double [:, :, :] A = A_np
    cdef double [:, :] b = b_np

    with nogil, parallel(num_threads=maxproc):
        for jj in prange(nlens, schedule='dynamic'):
            ii = indx[jj]

            x0 = xdim + 1
            x1 = -1
            y0 = ydim + 1
            y1 = -1
            for j in range(nlam):
                if x[j, ii] > x1:
                    x1 = x[j, ii]
                if x[j, ii] < x0:
                    x0 = x[j, ii]
                if y[j, ii] > y1:
                    y1 = y[j, ii]
                if y[j, ii] < y0:
                    y0 = y[j, ii]

            y0 = y0 - dx
            x0 = x0 - dx
            y1 = y1 + dx + 1
            x1 = x1 + dx + 1
            dy0 = y1 - y0
            dx0 = x1 - x0


            for j in range(nlam):
                k = 0
                for iy in range(y0, y1):
                    for ix in range(x0, x1):
                        #k = (iy - y0)*dx0 + ix - x0
                        if j == 0:
                            b[jj, k] = im[iy, ix]*isig[iy, ix]
                        A[jj, k, j] = psflets[j, iy, ix]*isig[iy, ix]
                        k = k + 1

    return A_np, b_np, size_np



@cython.boundscheck(False)
@cython.wraparound(False)

def dot(double [:, :] A, double [:, :] B, int maxproc=4):
    
    """
    Compute and return the simple dot product of the input matrices
    A and B, performing operations in parallel.  Return the product
    A*B.  If A is n x m and B is m x l, the output will be n x l.

    The input matrices should be double precision, the output will
    also be double precision.
    """

    cdef int i, j, k, n1, n2, n3
    cdef double x

    n1 = A.shape[0]
    n2 = A.shape[1]
    n3 = B.shape[1]
    assert A.shape[1] == B.shape[0]

    result_np = np.empty((n1, n3))
    cdef double [:, :] result = result_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(n1, schedule='dynamic'):
            for j in range(n3):
                x = 0
                for k in range(n2):
                    x = x + A[i, k]*B[k, j]
                result[i, j] = x

    return result_np


@cython.boundscheck(False)
@cython.wraparound(False)



def dot_3d(double [:, :, :] A, double [:, :, :] B, int maxproc=4):
    
    """
    Compute and return the simple dot product of the input matrices
    A and B, performing operations in parallel.  Return the product
    A*B.  If A is n x m and B is m x l, the output will be n x l.

    The input matrices should be double precision, the output will
    also be double precision.
    """

    cdef int i, j, k, ii, n1, n2, n3, nmat
    cdef double x

    nmat = A.shape[0]
    n1 = A.shape[1]
    n2 = A.shape[2]
    n3 = B.shape[2]
    assert A.shape[2] == B.shape[1]

    result_np = np.empty((nmat, n1, n3))
    cdef double [:, :, :] result = result_np

    with nogil, parallel(num_threads=maxproc):
        for ii in prange(nmat, schedule='dynamic'):
            for i in range(n1):
                for j in range(n3):
                    x = 0
                    for k in range(n2):
                        x = x + A[ii, i, k]*B[ii, k, j]
                    result[ii, i, j] = x

    return result_np




@cython.wraparound(False)
@cython.boundscheck(False)

def lstsq(double [:, :, :] A, double [:, :] b, long [:] indx, long [:] size, int ncoef, int maxproc=4):

    """

    """

    cdef int flag, its, jj, j, ii, i, l, k, nm, n, mm, m, inc, di
    cdef double c, f, h, s, x, y, z, tmp, tmp1, tmp2, sw, eps, tsh
    cdef double anorm, g, scale

    cdef extern from "math.h" nogil:
        double sqrt(double _x)
        double fabs(double _x)

    mm = A.shape[1]
    n = A.shape[2]
    inc = 1
    eps = 2.3e-16
    di = A.shape[0]

    ###############################################################
    # None of these arrays will be visible outside of this routine.
    # They are used to construct the SVD.
    ###############################################################

    tmparr_np = np.empty((di, n))
    cdef double [:, :] tmparr = tmparr_np
    su_np = np.empty((di, mm))
    cdef double [:, :] su = su_np
    sv_np = np.empty((di, n))
    cdef double [:, :] sv = sv_np
    w_np = np.empty((di, n))
    cdef double [:, :] w = w_np
    rv1_np = np.empty((di, n))
    cdef double [:, :] rv1 = rv1_np

    v_np = np.empty((di, mm, mm))
    cdef double [:, :, :] v = v_np
    
    ###############################################################
    # coef is the array that will hold the answer.
    # It will be returned by the function.
    ###############################################################

    coef_np = np.zeros((ncoef, n))
    cdef double [:, :] coef = coef_np

    ###############################################################
    # The code below is largely copied from Numerical Recipes and from
    # http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
    ###############################################################

    with nogil, parallel(num_threads=maxproc):

        for ii in prange(di, schedule='dynamic'):
            #if not good[ii]:
            #    continue
            m = size[ii]
            
            scale = 0.
            g = 0.
            anorm = 0.
            for i in range(n):
                l = i + 1
                rv1[ii, i] = scale*g
                g = 0.
                s = 0.
                scale = 0.
                if i < m:
                    for k in range(i, m):
                        scale = scale + fabs(A[ii, k, i])
                    if scale != 0:
                        for k in range(i, m):
                            A[ii, k, i] = A[ii, k, i]/scale
                            s = s + A[ii, k, i]*A[ii, k, i]
                        f = A[ii, i, i]
                        g = -1*sqrt(s)
                        if f < 0:
                            g = -1*g
                        h = f*g - s
                        A[ii, i, i] = f - g
                        if i != n - 1:
                            for j in range(l, n):
                                s = 0
                                for k in range(i, m):
                                    s = s + A[ii, k, i]*A[ii, k, j]
                                f = s/h
                                for k in range(i, m):
                                    A[ii, k, j] = A[ii, k, j] + f*A[ii, k, i]
                        for k in range(i, m):
                            A[ii, k, i] = A[ii, k, i]*scale

                w[ii, i] = scale*g
                g = 0.
                s = 0.
                scale = 0.
                if i < m and i != n - 1:
                    for k in range(l, n):
                        scale = scale + fabs(A[ii, i, k])
                    if scale != 0:
                        for k in range(l, n):
                            A[ii, i, k] = A[ii, i, k]/scale
                            s = s + A[ii, i, k]*A[ii, i, k]
                        f = A[ii, i, l]
                        g = -1*sqrt(s)
                        if f < 0:
                            g = -1*g
                        h = f*g - s
                        A[ii, i, l] = f - g
                        for k in range(l, n):
                            rv1[ii, k] = A[ii, i, k]/h
                        if i != m - 1:
                            for j in range(l, m):
                                s = 0
                                for k in range(l, n):
                                    s = s + A[ii, j, k]*A[ii, i, k]
                                for k in range(l, n):
                                    A[ii, j, k] = A[ii, j, k] + s*rv1[ii, k]
    
                        for k in range(l, n):
                            A[ii, i, k] = A[ii, i, k]*scale
    
                if fabs(w[ii, i]) + fabs(rv1[ii, i]) > anorm:
                    anorm = fabs(w[ii, i]) + fabs(rv1[ii, i])
            
            for i in range(n - 1, -1, -1):
                if i < n - 1:
                    if g != 0:
                        for j in range(l, n):
                            v[ii, j, i] = A[ii, i, j]/A[ii, i, l]/g
                        for j in range(l, n):
                            s = 0
                            for k in range(l, n):
                                s = s + A[ii, i, k]*v[ii, k, j]
                            for k in range(l, n):
                                v[ii, k, j] = v[ii, k, j] + s*v[ii, k, i]
                    for j in range(l, n):
                        v[ii, i, j] = 0.
                        v[ii, j, i] = 0.
                v[ii, i, i] = 1.
                g = rv1[ii, i]
                l = i
    
            for i in range(n - 1, -1, -1):
                l = i + 1
                g = w[ii, i]
                if i < n - 1:
                    for j in range(l, n):
                        A[ii, i, j] = 0.
                if g != 0:
                    g = 1./g
                    if i != n - 1:
                        for j in range(l, n):
                            s = 0
                            for k in range(l, m):
                                s = s + A[ii, k, i]*A[ii, k, j]
                            f = (s/A[ii, i, i])*g
                            for k in range(i, m):
                                A[ii, k, j] = A[ii, k, j] + f*A[ii, k, i]
                    for j in range(i, m):
                        A[ii, j, i] = A[ii, j, i]*g
                else:
                    for j in range(i, m):
                        A[ii, j, i] = 0.
                A[ii, i, i] = A[ii, i, i] + 1.
    
            for k in range(n - 1, -1, -1):
                for its in range(30):
                    flag = 1
                    for l in range(k, -1, -1):
                        nm = l - 1
                        if fabs(rv1[ii, l]) + anorm == anorm:
                            flag = 0
                            break
                        if fabs(w[ii, nm]) + anorm == anorm:
                            break
                    if flag != 0:
                        c = 0.
                        s = 1.
                        for i in range(l, k + 1):
                            f = s*rv1[ii, i]
                            if fabs(f) + anorm != anorm:
                                g = fabs(w[ii, i])
                                h = sqrt(f*f + g*g)
                                w[ii, i] = h
                                h = 1./h
                                c = g*h
                                s = -1.*f*h
                                for j in range(m):
                                    y = A[ii, j, nm]
                                    z = A[ii, j, i]
                                    A[ii, j, nm] = y*c + z*s
                                    A[ii, j, i] = z*c - y*s
                    z = w[ii, k]
                    if l == k:
                        if z < 0.:
                            w[ii, k] = -1.*z
                            for j in range(n):
                                v[ii, j, k] = -1.*v[ii, j, k]
                        break
                    #if its >= 30:
                    # no convergence
                    
                    x = w[ii, l]
                    nm = k - 1
                    y = w[ii, nm]
                    g = rv1[ii, nm]
                    h = rv1[ii, k]
                    f = ((y - z)*(y + z) + (g - h)*(g + h))/(2.*h*y)
    
                    g = sqrt(1. + f*f)
                    tmp = g
                    if f < 0:
                        tmp = -1*tmp
                    
                    f = ((x - z)*(x + z) + h*((y/(f + tmp)) - h))/x
                    
                    c = 1.
                    s = 1.
                    for j in range(l, nm + 1):
                        i = j + 1
                        g = rv1[ii, i]
                        y = w[ii, i]
                        h = s*g
                        g = c*g
                                               
                        z = sqrt(f*f + h*h)

                        rv1[ii, j] = z
                        c = f/z
                        s = h/z
                        f = x*c + g*s
                        g = g*c - x*s
                        h = y*s
                        y = y*c
                        for jj in range(n):
                            x = v[ii, jj, j]
                            z = v[ii, jj, i]
                            v[ii, jj, j] = x*c + z*s
                            v[ii, jj, i] = z*c - x*s
                        
                        z = sqrt(f*f + h*h)

                        w[ii, j] = z
                        if z != 0:
                            z = 1./z
                            c = f*z
                            s = h*z
                        f = c*g + s*y
                        x = c*y - s*g
                        for jj in range(m):
                            y = A[ii, jj, j]
                            z = A[ii, jj, i]
                            A[ii, jj, j] = y*c + z*s
                            A[ii, jj, i] = z*c - y*s
    
                    rv1[ii, l] = 0.
                    rv1[ii, k] = f
                    w[ii, k] = x
    
            inc = 1
            while True:
                inc = inc*3 + 1
                if inc > n:
                    break
            while True:
                inc = inc/3
                for i in range(inc, n):
                    sw = w[ii, i]
                    for k in range(m):
                        su[ii, k] = A[ii, k, i]
                    for k in range(n):
                        sv[ii, k] = v[ii, k, i]
                    j = i
                    while w[ii, j - inc] < sw:
                        w[ii, j] = w[ii, j - inc]
                        for k in range(m):
                            A[ii, k, j] = A[ii, k, j - inc] 
                        for k in range(n):
                            v[ii, k, j] = v[ii, k, j - inc]
                        j = j - inc
                        if j < inc:
                            break
                    w[ii, j] = sw
                    for k in range(m):
                        A[ii, k, j] = su[ii, k]
                    for k in range(n):
                        v[ii, k, j] = sv[ii, k]
                if inc <= 1:
                    break
            for k in range(n):
                jj = 0
                for i in range(m):
                    if A[ii, i, k] < 0:
                        jj = jj + 1
                for j in range(n):
                    if v[ii, j, k] < 0:
                        jj = jj + 1
                if jj > (m + n)/2:
                    for i in range(m):
                        A[ii, i, k] = -1.*A[ii, i, k]
                    for j in range(n):
                        v[ii, j, k] = -1.*v[ii, j, k]


            #eps = 2.3e-16
            tsh = 0.5*sqrt(m + n + 1.)*w[ii, 0]*eps
    
            for j in range(n):
                s = 0.
                if w[ii, j] > tsh:
                    for i in range(m):
                        s = s + A[ii, i, j]*b[ii, i]
                    s = s/w[ii, j]
                tmparr[ii, j] = s*1.
            for j in range(n):
                s = 0.
                for jj in range(n):
                    s = s + v[ii, j, jj]*tmparr[ii, jj]
                #coef[ii, j] = s*1.
                coef[indx[ii], j] = s*1.
    
    return coef_np
    
    
