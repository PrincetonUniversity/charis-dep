
import cython
from cython.parallel import prange, parallel
import numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)

def crosscorr(float [:, :, :] calimage, double [:, :] image,
              double [:, :] ivar, long [:] offsets, 
              int n1=0, int n2=-1, int m1=0, int m2=-1, int maxproc=4):

    cdef int i, j, ii, k1, k2, n, dj, nlam, upsamp, mmin, mmax
    cdef double x, y
    cdef extern from "math.h" nogil:
        double sqrt(double _x)
        double exp(double _x)

    upsamp = calimage.shape[2]/image.shape[1]
    n = offsets.shape[0]
    nlam = calimage.shape[0]
    n1 = max(n1, 0)
    if n2 < 0:
        n2 = image.shape[0] - n2
    n2 = min(image.shape[0] - 0, n2)
    m1 = max(m1, 4)
    if m2 < 0:
        m2 = image.shape[1] - m2
    m2 = min(image.shape[1] - 4, m2)
    if m2 < 4 or n2 < 4 or n1 > image.shape[0] - 4 or m1 > image.shape[1] - 4:
        return None
    if m2 <= m1 or n2 <= n1:
        return None

    maskedim_np = np.zeros((n2 - n1, m2 - m1))
    cdef double [:, :] maskedim = maskedim_np

    corrvals_np = np.zeros((nlam, n, n2 - n1))
    cdef double [:, :, :] corrvals = corrvals_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(n2 - n1, schedule='dynamic'):
            ii = i + n1
            x = 0
            y = 1e10
            for j in range(m1, m2):
                if image[ii, j] > x:
                    x = image[ii, j]
                if image[ii, j] < y and ivar[ii, j] > 0:
                    y = image[ii, j]
            if y < 0:
                y = 0
            if y >= 10*x:
                y = x/10.

            mmin = m1
            mmax = m2
            for j in range(m1, m2):
                if ivar[ii, j] > 0 and image[ii, j] < 5*y:
                    mmin = j
                    break
            for j in range(m2 - 1, m1, -1):
                if ivar[ii, j] > 0 and image[ii, j] < 5*y:
                    mmax = j
                    break

            for j in range(m1, mmin):
                maskedim[i, j - m1] = 0
            for j in range(m2 - 1, mmax, -1):
                maskedim[i, j - m1] = 0
                
            if mmax <= mmin:
                continue

            for j in range(mmin, mmax + 1):
            #for j in range(m1, m2):
                maskedim[i, j - m1] = image[ii, j]*sqrt(ivar[ii, j])

    with nogil, parallel(num_threads=maxproc):
        for i in prange(nlam, schedule='dynamic'):
            #for k1 in range(4, n1 - 4):
            for k1 in range(n1, n2):
                ii = k1 - n1
                for j in range(n):
                    y = 0
                    dj = offsets[j]
                    #for k2 in range(4, n2 - 4):
                    for k2 in range(m1, m2):
                        y = y + maskedim[ii, k2-m1]*calimage[i, k1, k2*upsamp + dj]
                    corrvals[i, j, k1 - n1] = corrvals[i, j, k1 - n1] + y

    return np.sum(corrvals_np, axis=0)


@cython.wraparound(False)
@cython.boundscheck(False)

def interpcal(float [:, :, :] calimage, double [:, :] image,
              unsigned short [:, :] mask, double [:, :] offsets, int maxproc=4):

    cdef int i, j, k, k1, k2, dj, nlam, noffset, upsamp, n1, n2, m1, m2
    cdef int nn1, nn2, mm1, mm2
    cdef double x

    upsamp = calimage.shape[2]/image.shape[1]
    nlam = calimage.shape[0]
    n1 = image.shape[0]
    n2 = image.shape[1]
    m1 = offsets.shape[0]
    m2 = offsets.shape[1]

    calinterp_np = np.zeros((nlam, n1, n2))
    cdef double [:, :, :] calinterp = calinterp_np

    xarr_np = np.arange(int(np.floor(np.amin(offsets))) - 1, 
                        int(np.floor(np.amax(offsets))) + 3)
    cdef long [:] xarr = xarr_np
    noffset = xarr_np.shape[0]

    fac_np = np.ones((m1, m2, xarr_np.shape[0]))
    cdef double [:, :, :] fac = fac_np

    with nogil, parallel(num_threads=maxproc):
        for k1 in prange(m1, schedule='dynamic'):
            for k2 in range(m2):
                for j in range(noffset):
                    for k in range(noffset):
                        if j != k:
                            fac[k1, k2, j] = fac[k1, k2, j]*(offsets[k1, k2] - xarr[k])*1./(xarr[j] - xarr[k])

    with nogil, parallel(num_threads=maxproc):
        for i in prange(nlam, schedule='dynamic'):
            for k1 in range(4, n1 - 4):
                for k2 in range(4, n2 - 4):
                    x = 0
                    for j in range(noffset):
                        x = x + fac[k1, k2, j]*calimage[i, k1, k2*upsamp + xarr[j]]
                    calinterp[i, k1, k2] = x

    return calinterp_np
                        


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

    ylim_np = np.zeros((nlens, 2), np.int64)
    ylim_np[:, 0] = im.shape[0] + 1
    ylim_np[:, 1] = -1
    cdef long [:, :] ylim = ylim_np
    xlim_np = np.zeros((nlens, 2), np.int64)
    xlim_np[:, 0] = im.shape[1] + 1
    xlim_np[:, 1] = -1
    cdef long [:, :] xlim = xlim_np

    with nogil, parallel(num_threads=maxproc):
        for jj in prange(nlens, schedule='dynamic'):
            ii = indx[jj]

            if False:
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
            else:
                for j in range(nlam):
                    if x[j, ii] > xlim[jj, 1]:
                        xlim[jj, 1] = x[j, ii]
                    if x[j, ii] < xlim[jj, 0]:
                        xlim[jj, 0] = x[j, ii]
                    if y[j, ii] > ylim[jj, 1]:
                        ylim[jj, 1] = y[j, ii]
                    if y[j, ii] < ylim[jj, 0]:
                        ylim[jj, 0] = y[j, ii]
                
                ylim[jj, 0] = max(ylim[jj, 0] - dx, 0)
                ylim[jj, 1] = min(ylim[jj, 1] + dx + 1, ydim)
                xlim[jj, 0] = max(xlim[jj, 0] - dx, 0)
                xlim[jj, 1] = min(xlim[jj, 1] + dx + 1, xdim)
                
                dy0 = ylim[jj, 1] - ylim[jj, 0]
                dx0 = xlim[jj, 1] - xlim[jj, 0]
            
            size[jj] = dy0*dx0

    maxsize = np.amax(size_np)
    A_np = np.zeros((nlens, maxsize, nlam))
    b_np = np.zeros((nlens, maxsize))

    cdef double [:, :, :] A = A_np
    cdef double [:, :] b = b_np

    with nogil, parallel(num_threads=maxproc):
        for jj in prange(nlens, schedule='dynamic'):
            ii = indx[jj]

            if False:
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
            else:
                y0 = ylim[jj, 0]
                y1 = ylim[jj, 1]
                x0 = xlim[jj, 0]
                x1 = xlim[jj, 1]

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


@cython.wraparound(False)
@cython.boundscheck(False)

def lstsq(double [:, :, :] A, double [:, :] b, long [:] indx, long [:] size, int ncoef, int returncov=0, int maxproc=4):

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

    v_np = np.empty((di, n, n))
    cdef double [:, :, :] v = v_np
    
    ###############################################################
    # coef is the array that will hold the answer.
    # It will be returned by the function.
    ###############################################################

    coef_np = np.zeros((ncoef, n))
    cdef double [:, :] coef = coef_np

    cov_np = np.ones((ncoef, n, n))*np.inf
    cdef double [:, :, :] cov = cov_np

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

            # Compute the covariance matrix if needed.
            for i in range(n):
                if returncov == 0:
                    continue
                for j in range(i + 1):
                    s = 0.
                    for k in range(n):
                        if w[ii, k] > tsh:
                            s = s + v[ii, i, k]*v[ii, j, k]/(w[ii, k]*w[ii, k])
                    cov[indx[ii], j, i] = s
                    cov[indx[ii], i, j] = s

    if returncov == 0:
        return coef_np
    else:
        return coef_np, cov_np
    

@cython.wraparound(False)
@cython.boundscheck(False)

def optext(double [:, :] im, double [:, :] ivar, 
           double [:, :, :] xindx, double[:, :, :] yindx,
           double [:, :, :] loglamindx, int [:, :] nlam,
           double [:] refloglam, int nmax, 
           int delt_x=7, double sig=0.7, int maxproc=4):
    """
    """

    cdef int i, j, k, n, nx, ny, ix, iy, xdim, ydim, nref, i1, i2
    cdef double x, dx, num, denom, w1, w2, wtot

    cdef extern from "math.h" nogil:
        double exp(double _x)

    ####################################################################
    # Relevant dimensions and limits for loops
    ####################################################################

    xdim = im.shape[1]
    ydim = im.shape[0]
    nref = refloglam.shape[0]

    nx = xindx.shape[0]
    ny = xindx.shape[1]

    ####################################################################
    # Allocate arrays and create memory views.  Only coefs and ivar_tot 
    # will be returned; coefs_num and coefs_denom are internal.
    ####################################################################

    coefs_num_np = np.zeros((nx, ny, nmax))
    cdef double [:, :, :] coefs_num = coefs_num_np
    coefs_denom_np = np.zeros((nx, ny, nmax))
    cdef double [:, :, :] coefs_denom = coefs_denom_np

    coefs_np = np.zeros((nref, nx, ny))
    cdef double [:, :, :] coefs = coefs_np
    ivar_tot_np = np.zeros((nref, nx, ny))
    cdef double [:, :, :] ivar_tot = ivar_tot_np

    lamref_np = np.zeros((nx, nmax))
    cdef double [:, :] lamref = lamref_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(nx, schedule='dynamic'):
            for j in range(ny):
                n = nlam[i, j] - 1
                if (xindx[i, j, 0] < 10 or xindx[i, j, n] < 10 or
                    xindx[i, j, 0] > xdim - 10 or xindx[i, j, n] > xdim - 10 or
                    yindx[i, j, 0] < 10 or yindx[i, j, n] < 10 or
                    yindx[i, j, 0] > ydim - 10 or yindx[i, j, n] > ydim - 10):
                    continue
                
                for k in range(n):
                    num = 0
                    denom = 0
                    x = xindx[i, j, k]
                    i1 = (int)(x + 1 - delt_x/2.)
                    wtot = 0
                    for ix in range(i1, i1 + delt_x):
                        dx = x - ix
                        w1 = exp(-dx*dx/(2.*sig*sig))
                        wtot = wtot + w1

                        iy = (int)(yindx[i, j, k])
                        num = num + w1*im[iy, ix]*ivar[iy, ix]
                        denom = denom + w1*w1*ivar[iy, ix]

                    coefs_num[i, j, n - k - 1] = num/wtot
                    coefs_denom[i, j, n - k - 1] = denom/(wtot*wtot)
                    lamref[i, n - k - 1] = loglamindx[i, j, k]

                i2 = 1
                for k in range(nref):
                    while lamref[i, i2] < refloglam[k]:
                        i2 = i2 + 1
                        if i2 == n - 1:
                            break

                    i1 = i2 - 1
                    w1 = lamref[i, i2] - refloglam[k]
                    w2 = refloglam[k] - lamref[i, i1]
                    wtot = w1 + w2
                    w1 = w1/wtot
                    w2 = w2/wtot
                    
                    num = coefs_num[i, j, i1]*w1 + coefs_num[i, j, i2]*w2
                    denom = coefs_denom[i, j, i1]*w1 + coefs_denom[i, j, i2]*w2
                    coefs[k, i, j] = num/(denom + 1e-300)
                    ivar_tot[k, i, j] = denom
                    
    return coefs_np, ivar_tot_np 
