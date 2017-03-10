
import cython
from cython.parallel import prange, parallel
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

def fit_expdecay(float [:, :, :] cts, double [:, :] ref, 
                 unsigned short [:, :] mask, float tol=1e-5, 
                 int read0=2, double maxresid=100, int maxproc=4):

    """

    """

    cdef double x, y, best, factor, num, denom, t0
    cdef int i, j, k, imax, jmax, ii, its, nread, ny, nx, nchan, ichan

    cdef extern from "math.h" nogil:
        double exp(double _x)
        double fabs(double _x)

    nread, ny, nx = [cts.shape[0], cts.shape[1], cts.shape[2]]
    nchan = nx/64

    ####################################################################
    # Fit a short ramp to the part of the detector we will use to fit
    # an exponential decay of the reference voltage
    ####################################################################

    imax = nread
    jmax = 500
    if imax > 10:
        imax = 10
    factor = 12.0/((imax - 1)**3 - (imax - 1))
    weights_np = factor*(np.arange(1, imax) - imax/2.0)
    cdef double [:] weights = weights_np
    ctrates_np = np.zeros((ny, nx))
    cdef double [:, :] ctrates = ctrates_np
        
    with nogil, parallel(num_threads=maxproc):
        for j in prange(jmax, schedule='dynamic'):
            for i in range(imax - 1):
                for k in range(nx):
                    ichan = k/64
                    x = weights[i]*(cts[i + 1, j, k] - ref[i + 1, ichan])
                    ctrates[j, k] = ctrates[j, k] + x

    ####################################################################
    # Compute the difference between the actual and expected values
    # for the first read.  
    #
    # Expected counts: <cts_i - ref_i - i*rate> - ref_0, where the
    # average is over reads 1-imax and ref_i is the mean reference in a
    # given channel in the first read.
    ####################################################################

    resid_np = np.empty((jmax, nx))
    resid_np[:] = cts[0, :jmax]     # Actual counts, first read
    cdef double [:, :] resid = resid_np
    t_np = np.empty((jmax, nx), int)
    cdef long [:, :] t = t_np
    
    with nogil, parallel(num_threads=maxproc):
        for j in prange(jmax, schedule='dynamic'):
            for i in range(1, imax):
                for k in range(nx):
                    ichan = k/64   # channel, 0-31
                    x = cts[i, j, k] - ref[i, ichan] - i*ctrates[j, k]
                    resid[j, k] = resid[j, k] - x/(imax - 1.)
            for k in range(nx):
                ichan = k/64      # channel, 0-31
                resid[j, k] = resid[j, k] - ref[0, ichan]

                ########################################################
                # Ignore high count rates and masked pixels.
                ########################################################

                if ctrates[j, k] > 500 or mask[j, k] == 0:
                    resid[j, k] = 0

                ########################################################
                # Readout order alternates between channels.  Also
                # account for 8 pixel dead time as readout returns to
                # the next row.
                ########################################################

                if (k/64)%2 == 0:
                    t[j, k] = k%64 + j*72
                else:
                    t[j, k] = 63 - k%64 + j*72

    ####################################################################
    # Now we need to fit an exponential decay to resid(t), separately
    # for the reference and non-reference pixels.  Minimize chi
    # squared by locally fitting parabolas to chi squared as a
    # function of the assumed decay rate, equivalent to using Newton's
    # method to find where the derivative vanishes.  Note that for a
    # given decay rate the best-fit normalizations may be easily
    # computed, turning a many-D minimization into a 1D minimization.
    ####################################################################

    t0 = 29*72  # initial guess, ~21 ms decay constant (100 kHz pixel rate)
    imax = 72*(jmax + 1)

    expval_np = np.empty(imax)
    cdef double [:] expval = expval_np
    norm_np = np.zeros(nchan + 1)
    cdef double [:] norm = norm_np

    dx_np = np.zeros(3)
    cdef double [:] dx = dx_np
    dy_np = np.zeros(3)
    cdef double [:] dy = dy_np
    chisq_np = np.zeros(jmax)
    cdef double [:] chisq = chisq_np

    for its in range(10):   # Maximum number of iterations
        for ii in range(3): # Three points for a parabolic fit

            dx[ii] = t0 + 10*(ii - 1)

            for i in range(imax):
                expval[i] = exp(-1.*i/dx[ii])
            
            ############################################################
            # Least-squares fit for the normalization in each channel:
            # <expval*resid>/<expval**2>
            ############################################################

            with nogil, parallel(num_threads=maxproc):
                for ichan in prange(nchan, schedule='dynamic'): # Channel 0-31
                    num = 0
                    denom = 0
                    for j in range(4, jmax):
                        for i in range(64):
                            if (ichan == 0 and i < 4) or (ichan == nchan - 1 and i > 59):
                                continue  # Neglect reference pixels
                            x = expval[t[j, i + 64*ichan]]
                            y = resid[j, i + 64*ichan]*mask[j, i + 64*ichan]
                            num = num + y*x
                            if y != 0:
                                denom = denom + x*x
                    norm[ichan] = num/denom

            ############################################################
            # Sum of the squared residuals.  Mask residuals greater
            # than maxresid in the second and subsequent iterations.
            ############################################################

            chisq_np[:] = 0
            with nogil, parallel(num_threads=maxproc):
                for j in prange(4, jmax, schedule='dynamic'):
                    for i in range(4, nx - 4):
                        ichan = i/64
                        if mask[j, i] == 0 or resid[j, i] == 0:
                            chisq[j] = chisq[j] + maxresid*maxresid
                            continue
                        x = norm[ichan]*expval[t[j, i]] - resid[j, i]
                        x = x*x
                        if its == 0 or x < maxresid*maxresid:
                            chisq[j] = chisq[j] + x
                        else:
                            resid[j, i] = 0
                            chisq[j] = chisq[j] + maxresid*maxresid
            dy[ii] = np.sum(chisq_np)
            
        ################################################################
        # Vertex of the parabolic fit --> minimum residual
        ################################################################

        num = (dx[2]**2*(dy[0] - dy[1]) 
               + dx[1]**2*(dy[2] - dy[0]) 
               + dx[0]**2*(dy[1] - dy[2]))
        denom = 2.*(dx[2]*(dy[0] - dy[1]) 
                    + dx[1]*(dy[2] - dy[0]) 
                    + dx[0]*(dy[1] - dy[2]))

        best = num/denom
        if fabs((best - t0)/t0) < tol:  # Convergence
            t0 = best
            break
        else:
            t0 = best
        
    ####################################################################
    # Remove the fitted exponential decay of the reference voltage.
    # Fit a separate coefficient (index 32) for the reference pixels.
    ####################################################################

    for i in range(imax):
        expval[i] = exp(-1.*i/t0)

    with nogil, parallel(num_threads=maxproc):
        for ichan in prange(nchan + 1, schedule='dynamic'):
            num = 0
            denom = 0
            if ichan < nchan:
                for j in range(4, jmax):
                    for i in range(64):
                        if (ichan == 0 and i < 4) or (ichan == nchan - 1 and i > 59):
                            continue # Ignore reference pixels
                        x = expval[t[j, i + 64*ichan]]
                        y = resid[j, i + 64*ichan]*mask[j, i + 64*ichan]
                        num = num + y*x
                        if y != 0:
                            denom = denom + x*x
            else: # Reference pixels
                for j in range(4):
                    for i in range(nx):
                        x = expval[t[j, i]]
                        y = resid[j, i]
                        num = num + y*x
                        if y != 0:
                            denom = denom + x*x
                for j in range(4, jmax):
                    for i in range(4):
                        x = expval[t[j, i]]
                        y = resid[j, i]
                        num = num + y*x
                        if y != 0:
                            denom = denom + x*x
                    for i in range(nx - 4, nx):
                        x = expval[t[j, i]]
                        y = resid[j, i]
                        num = num + y*x
                        if y != 0:
                            denom = denom + x*x
            norm[ichan] = num/denom

    ####################################################################
    # Subtract the exponential decay.
    ####################################################################

    for j in range(jmax):
        for i in range(nx):
            if j < 4 or i < 4 or i > nx - 5:
                ichan = nchan  # Reference pixels
            else:
                ichan = i/64   # Channel
            x = norm[ichan]*expval[t[j, i]]
            cts[0, j, i] = cts[0, j, i] - x

    #print t0
    #for i in range(nchan + 1):
    #    print norm[i]
    #print "Removed exponential decay of the reference voltage from the first read."
    #print "Fitted decay constant: %.2f ms" % (t0*1e-5*1000)

    if t0 < 28*72 or t0 > 31*72:
        print "Warning: fitted exponential decay rate of reference voltage in"
        print "fitnonlin.pyx does not appear to match time constant observed in"
        print "late 2016.  Please visually check 2D image.  To diable fitting"
        print "of an exponential decay, call fit_ramp with fitexpdecay=0."
    
    return




@cython.boundscheck(False)
@cython.wraparound(False)

def fit_nonlin(float [:, :, :] cts, double [:, :] ctrates, double [:, :] ref, 
               float tol=1e-5, int read0=1, 
               double a=0.991, double b=-1.8e-6, double c=-2e-11, 
               double threshold=5000., double sat=50000., int maxproc=4):

    """

    """

    cdef double rate, reset, chisq, x, y, best, num, denom
    cdef int i, j, k, kmax, ii, its, nread, ny, nx, nchan, ichan

    cdef extern from "math.h" nogil:
        double fabs(double _x)
    
    nread, ny, nx = [cts.shape[0], cts.shape[1], cts.shape[2]]

    ####################################################################
    # Thread-safe arrays to be edited in parallel.
    ####################################################################

    fit_np = np.zeros((ny, nread))
    cdef double [:, :] fit = fit_np
    cts_local_np = np.zeros((ny, nread))
    cdef double [:, :] cts_local = cts_local_np
    dx_np = np.zeros((ny, 3))
    cdef double [:, :] dx = dx_np
    dy_np = np.zeros((ny, 3))
    cdef double [:, :] dy = dy_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(ny, schedule='dynamic'):
            for j in range(nx):
                
                ########################################################
                # Keep up-the-ramp count rates for pixels that aren't
                # close to saturation.  
                ########################################################

                rate = ctrates[i, j]
                if rate*(read0 + nread - 1) < threshold:
                    continue
                    
                ########################################################
                # Otherwise, use the up-the-ramp count rate as an
                # initial guess.  Make a C-contiguous copy of the
                # counts that we will repeatedly access.  Then
                # minimize chi squared by locally fitting parabolas to
                # chi squared as a function of the assumed count rate,
                # equivalent to using Newton's method to find where
                # the derivative vanishes.
                ########################################################
                
                for k in range(nread):
                    ichan = j/64
                    cts_local[i, k] = cts[k, i, j] - ref[k, ichan]

                ########################################################
                # Only use reads that should not be heavily saturated.
                ########################################################

                x = fabs(cts_local[i, 1] - cts_local[i, 0]) + 1e-10
                kmax = (int)(sat/x - read0)
                if kmax < 2:
                    kmax = 2
                elif kmax > nread:
                    kmax = nread

                for its in range(100):  # Maximum number of iterations
                    for ii in range(3): # Three points to fit a parabola
                        dx[i, ii] = rate + ii - 1.

                        for k in range(kmax):
                            y = dx[i, ii]*(k + read0)
                            if y > threshold:
                                x = y - threshold
                                y = threshold + a*x + b*x*x + c*x*x*x
                            fit[i, k] = y

                        reset = 0
                        for k in range(kmax):
                            reset = reset + cts_local[i, k] - fit[i, k]
                        reset = reset/kmax

                        chisq = 0
                        for k in range(kmax):
                            x = cts_local[i, k] - fit[i, k] - reset
                            chisq = chisq + x*x
                        dy[i, ii] = chisq
                    
                    ####################################################
                    # Vertex of the parabolic fit --> minimum residual
                    ####################################################
                        
                    num = (dx[i, 2]**2*(dy[i, 0] - dy[i, 1]) 
                           + dx[i, 1]**2*(dy[i, 2] - dy[i, 0]) 
                           + dx[i, 0]**2*(dy[i, 1] - dy[i, 2]))
                    denom = 2.*(dx[i, 2]*(dy[i, 0] - dy[i, 1]) 
                                + dx[i, 1]*(dy[i, 2] - dy[i, 0]) 
                                + dx[i, 0]*(dy[i, 1] - dy[i, 2]))
                    
                    if denom == 0 or num == 0:
                        break  # Either one will cause division by zero
                    best = num/denom

                    if fabs((best - rate)/rate) < tol:  # Convergence
                        ctrates[i, j] = best
                        break
                    else:
                        rate = best

    return


@cython.boundscheck(False)
@cython.wraparound(False)

def fit_ramp(float [:, :, :] cts, unsigned short [:, :] mask, 
             float tol=1e-5, int read0=1, 
             double a=0.991, double b=-1.8e-6, double c=-2e-11, 
             double threshold=5000., double sat=50000.,
             double gain=2, int maxproc=4, char refsub='t',
             int fitnonlin=1, int fitexpdecay=1, int returnivar=1):

    """
    
    """

    cdef double x, y, factor, xtop, xbot, rdnse
    cdef int i, j, k, jj, kk, nread, ny, nx, nchan, ichan

    nread, ny, nx = [cts.shape[0], cts.shape[1], cts.shape[2]]
    if nx != 2048 or ny != 2048:
        print "Error: fit_ramp assumes a 2048x2048 input array."
        return None
    nchan = nx/64
        
    ####################################################################
    # Weights for up-the-ramp, used for pixels that aren't close to 
    # saturation.
    ####################################################################

    factor = 12.0/(nread**3 - nread)
    weights_np = factor*(np.arange(1, nread + 1) - (nread + 1)/2.0)
    cdef double [:] weights = weights_np

    ####################################################################
    # Array of count rates, the output of this function, and mean
    # values of the reference pixels in each read and each channel.
    ####################################################################

    ctrates_np = np.zeros((ny, nx))
    cdef double [:, :] ctrates = ctrates_np
    ref_np = np.zeros((nread, nchan))
    cdef double [:, :] ref = ref_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(nread, schedule='dynamic'):

            ############################################################
            # Reference level to subtract: mean of reference pixels
            ############################################################

            for j in range(4):
                for k in range(nx):
                    ichan = k/64
                    if refsub == 'a':
                        ref[i, ichan] = ref[i, ichan] + cts[i, j, k] + cts[i, ny - 1 - j, k]
                    elif refsub == 't':
                        ref[i, ichan] = ref[i, ichan] + cts[i, ny - 1 - j, k]
                    elif refsub == 'b':
                        ref[i, ichan] = ref[i, ichan] + cts[i, j, k]
                    else:
                        pass
            for j in range(nchan):
                if refsub == 'a':
                    ref[i, j] = ref[i, j]/(64*8)
                else:
                    ref[i, j] = ref[i, j]/(64*4)

    ####################################################################
    # Fit and remove the exponential decay of the reference voltage in
    # the first read.
    ####################################################################

    if read0 == 1 and nread > 2 and fitexpdecay:
        fit_expdecay(cts, ref_np, mask, tol=tol, read0=read0, 
                     maxresid=100, maxproc=maxproc)

    ####################################################################
    # Fit the ramp
    ####################################################################

    with nogil, parallel(num_threads=maxproc):
        for j in prange(ny, schedule='dynamic'):
            for i in range(nread):
                for k in range(nx):
                    ichan = k/64
                    x = weights[i]*(cts[i, j, k] - ref[i, ichan])
                    ctrates[j, k] = ctrates[j, k] + x

    if fitnonlin:
        fit_nonlin(cts, ctrates, ref, tol=tol, read0=read0, 
                   a=a, b=b, c=c, threshold=threshold, sat=sat, maxproc=maxproc)

    ####################################################################
    # Subtract a linear fit between the reference pixels on either end
    # of the readout channel.
    ####################################################################
        
    with nogil, parallel(num_threads=maxproc):
        for ichan in prange(nchan, schedule='dynamic'):
            xtop = 0
            xbot = 0
            for j in range(4):
                for k in range(64):
                    kk = 64*ichan + k
                    xtop = xtop + ctrates[ny - j - 1, kk]
                    xbot = xbot + ctrates[j, kk]
            xtop = xtop/(64*4)
            xbot = xbot/(64*4)

            for j in range(ny):
                for k in range(64):
                    kk = 64*ichan + k
                    x = (xtop - xbot)*(j*1./ny) + xbot
                    ctrates[j, kk] = ctrates[j, kk] - x
             
    if not returnivar:
        return ctrates_np
                    
    ####################################################################
    # Compute the inverse variance using the measured read noise and
    # the input factor times the ADU count rate.
    ####################################################################

    ivar_np = np.empty((ny, nx))
    cdef double [:, :] ivar = ivar_np
    with nogil, parallel(num_threads=maxproc):
        for ichan in prange(nchan, schedule='dynamic'):  # Channel, 0-31
            rdnse = 0
            for j in range(4):
                for k in range(64):
                    kk = k + ichan*64
                    x = ctrates[j, kk]
                    rdnse = rdnse + x*x/(64*8)
                    x = ctrates[ny - j - 1, kk]
                    rdnse = rdnse + x*x/(64*8)
            for j in range(ny):
                for k in range(64):
                    kk = k + ichan*64
                    y = ctrates[j, kk]
                    if mask[j, kk] == 0:
                        ivar[j, kk] = 0
                    elif y <= 0 or j < 4 or j >= ny - 4:
                        ivar[j, kk] = 1./rdnse
                    elif y > 0 and y*(nread + read0) < sat:
                        x = 6./5*(nread*nread + 1.)/(nread*(nread*nread - 1.))
                        #ivar[j, kk] = 1./(rdnse + phnsefac*y/nread)
                        ivar[j, kk] = 1./(rdnse + x*y/gain)
                    else:
                        x = sat/y - read0
                        if x < 1:
                            x = 1
                        ivar[j, kk] = 1./(rdnse + 2./gain*(sat - y*read0)/x**2)

    ####################################################################
    # Set ivar to zero for pixels where one read contributes an
    # inordinate number of counts (at least four times the derived
    # count rate) that cannot be explained by read noise fluctuations.
    # Note: we could try to fix these pixels, but it's a little
    # dangerous as it could introduce biases.
    ####################################################################

    stds_np = np.zeros((nchan))
    cdef double [:] stds = stds_np

    ####################################################################
    # First find the standard deviation of the reference pixels in
    # each channel.
    ####################################################################
 
    with nogil, parallel(num_threads=maxproc):
        for ichan in prange(nchan, schedule='dynamic'):
            for j in range(4):
                for i in range(nread - 1):
                    for k in range(64):
                        jj = ny - j - 1
                        kk = k + ichan*64
                        x = cts[i + 1, jj, kk] - cts[i, jj, kk]
                        x = x - (ref[i + 1, ichan] - ref[i, ichan])
                        stds[ichan] = stds[ichan] + x*x/(64*4*(nread - 1))

    stds_np[:] = np.sqrt(stds_np)
    
    with nogil, parallel(num_threads=maxproc):
        for j in prange(ny, schedule='dynamic'):
            for i in range(nread - 1):
                for k in range(nx):
                    ichan = k/64
                    x = cts[i + 1, j, k] - cts[i, j, k]
                    x = x - (ref[i + 1, ichan] - ref[i, ichan])
                    if x > 7*stds[ichan]:  # >7sigma in read noise
                        y = cts[nread - 1, j, k] - cts[0, j, k]
                        y = y - (ref[nread - 1, ichan] - ref[0, ichan])
                        y = y/(nread - 1.)
                        if y < ctrates[j, k]:
                            y = ctrates[j, k]
                        if x > 4*y:  # >4 times expected count rate
                            ivar[j, k] = 0

    ####################################################################
    # Set the count rate to zero for masked pixels.
    ####################################################################

    with nogil, parallel(num_threads=maxproc):
        for j in prange(ny, schedule='dynamic'):
            for k in range(nx):
                if ivar[j, k] == 0:
                    ctrates[j, k] = 0

    return ctrates_np, ivar_np
