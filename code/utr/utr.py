#!/usr/bin/env python

import numpy as np
try:
    from image import Image
except:
    from charis.image import Image
import fitramp
import logging
import re
import time
import multiprocessing
from astropy.io import fits

log = logging.getLogger('main')

def getreads(filename, header=fits.PrimaryHDU().header, read_idx=[1,None]):
    """
    Get reads from fits file and put them in the correct 
    format for the up-the-ramp functions. 

    Inputs:
    1. filename:  string, name of the fits file

    Optional inputs:
    1. header:    FITS header object for recording the reads used.  
    2. read_idx:  list, [first index, last index] to extract from
                  the hdulist: index = 1 is the first read, use 
                  last index = None to use all reads after the 
                  first index.  Default [1, None].

    Returns:
    1. reads:     3D float32 ndarray, shape = (nreads, ny, nx)

    The input header, if given, will contain keywords for the first
    and last reads used.

    """

    log.info("Getting reads from " + filename)

    hdulist = fits.open(filename)
    shape = hdulist[1].data.shape
    if read_idx[1] > read_idx[0]:
        idx1 = read_idx[1] + 1
    else:
        idx1 = read_idx[1]
    reads = np.zeros((len(hdulist[read_idx[0]:idx1]), shape[0], shape[1]),
                     np.float32)

    header.append(('firstrd', read_idx[0], 'First HDU of original file used'), end=True)

    for i, r in enumerate(hdulist[read_idx[0]:idx1]):
        header['lastrd'] = (i + read_idx[0], 'Last HDU of original file used')
        reads[i] = r.data
    return reads



def calcramp(filename, mask=None, gain=2., noisefac=0,
             header=fits.PrimaryHDU().header, read_idx=[1,None], 
             maxcpus=multiprocessing.cpu_count(),
             fitnonlin=True, fitexpdecay=True):

    """
    Function calcramp computes the up-the-ramp count rates and their
    inverse variance using the cython function fit_ramp.

    Input:
    1. filename  string, name of the file containing the reads, to be
                 opened by astropy.io.fits.open().  Reads are assumed 
                 to reside in HDU 1, 2, ..., N.
    Optional inputs:
    1. mask      boolean array, nonzero for good pixels and zero for 
                 bad pixels.  Default None (no masking)
    2. gain      Float, electrons/ADU, used to compute shot noise on 
                 count rates.  Default 2.
    3. noisefac  Float, extra noise to add to derived count rates.  
                 This noise will be added in quadrature but will be 
                 assumed to be a fixed fraction of the count rate;
                 it is appropriate for handing imperfect PSFlet 
                 models.  Default 0.
    4. header    FITS header to hold information on the ramp.  Create
                 a new header if this is not given.
    5. read_idx  Reads to use to compute the ramp.  Default [1, None],
                 i.e., use the first and all subsequent reads.
    6. maxcpus   Maximum number of threads for OpenMP parallelization
                 in Cython.  Default multiprocessing.cpu_count().
    7. fitnonlin  Boolean, fit an approximately measured nonlinear 
                 response to each pixel's count rate?  Adds very 
                 little to the computational cost.  Default True.
    8. fitexpdecay  Boolean, fit for the exponential decay of the 
                 reference voltage in the first read (if using the
                 first read)?  Strongly recommended for CHARIS data.
                 Default True.  Only possible if there are at least 
                 three reads.

    Returns:    
    1. Image     an Image class containing the derived count rates,
                 their inverse variance, and a FITS header.


    This function is largely a wrapper for the cython function
    fit_ramp, which is calls after first calling getreads().  

    """

    header.append(('comment', ''), end=True)
    header.append(('comment', '*'*60), end=True)
    header.append(('comment', '*'*19 + ' Ramp, Masking, Noise ' + '*'*19), end=True)
    header.append(('comment', '*'*60), end=True)    
    header.append(('comment', ''), end=True)

    reads = getreads(filename, header, read_idx)

    try:
        read0 = header['firstrd']
    except:
        read0 = 0

    maskarr = np.ones(reads[0].shape, np.uint16)
    if mask is not None:
        maskarr[:] = mask != 0

    header.append(('pixmask', mask is not None, 'Mask known bad/hot pixels?'), end=True)

    data, ivar = fitramp.fit_ramp(reads, maskarr, tol=1e-5, read0=read0,
                                  gain=gain, maxproc=maxcpus,
                                  fitnonlin=fitnonlin, fitexpdecay=fitexpdecay,
                                  returnivar=True)

    if noisefac > 0:
        ivar[:] = 1./(1./(ivar + 1e-100) + (noisefac*data)**2)

    header.append(('gain', gain, 'Assumed detector gain for Poisson variance'), end=True)
    header['noisefac'] = (noisefac, 'Added noise (as fraction of abs(ct rate))')

    header['fitdecay'] = (fitexpdecay, 'Fit exponential decay of ref. volt. in read 1?')
    header['nonlin'] = (fitnonlin, 'Fit nonlinear pixel response?')
    

    return Image(data=data, ivar=ivar, header=header)




def _interp_coef(nreads, sig_rn, cmin, cmax, cpad=500, interp_meth='linear'):
    """
    Build interpolation functions for the up-the-ramp coefficients, 
    which are the weights that give the count (c) and intercept (a)
    in the equation y_i = a + i*c, where i is an integer from 1 to 
    nreads. This function returns interpolation objects for the 
    coefficients needed to calculate a and c. 

    Inputs:
    1. nreads:      int, the number of reads in the ramp
    2. sig_rn:      float, the std of the read noise in electrons
    3. cmin:        float, the minimum possible count rate in electrons
    4. cmax:        float, the maximum possible count rate in electrons

    Optional inputs:
    1. cpad:        int, pad for count rate, which may be necessary to
                    ensure that the interpolation range is large enough
    2. interp_meth: string, interpolation method. Possible methods 
                    are the same as for scipy's interp1d.

    Returns: 
    1. ia_coef:     Interpolation object for the coefficients needed to 
                    calculate a: ia_coef(count) = [w1, w2, w3...w_nreads]
    2. ic_coef:     Interpolation object for the coefficients needed to 
                    calculate c: ic_coef(count) = [w1, w2, w3...w_nreads]
    3. ic_ivar:     Interpolation object for count rate inverse variance

    Note:
    The weights are given by the generalized least squares solution 
    for a linear model. It turns out that, if we know the read noise
    and the number of reads, the weights are only a function of the 
    count. The purpose of this function is to precalculate the
    weights to speed up the up-the-ramp calculation. 
    """
    from scipy import interpolate

    i_arr = np.arange(1, nreads+1)
    x = np.array([np.ones(i_arr.size), i_arr]).T

    ###################################################################
    # Calculate the components of the covariance matrix: cov_shot = 
    # shot noise component (divided by the count), cov_read = the read
    # noise component. Thus, cov = cov_shot*count + cov_read. 
    ###################################################################

    uptri = np.triu(np.repeat(np.arange(1, nreads+1)[:,None], nreads, axis=1))
    cov_shot = uptri + uptri.T - np.tril(uptri)
    cov_read = np.diag(np.ones(nreads))*sig_rn**2

    ###################################################################
    # Calculate the up-the-ramp coefficients for many count values, 
    # store them, and interpolate using scipy's interp1d function.
    # note: a = intercept and c = count (all in units of electrons)
    ###################################################################

    a_coef = []
    c_coef = []
    c_ivar = []
    cvals = np.arange(np.floor(cmin)-cpad, np.ceil(cmax)+1)
    for c in cvals:
        cov = cov_shot*c + cov_read
        invcov = np.linalg.inv(cov)
        var = np.linalg.inv(np.dot(x.T, invcov).dot(x))
        coef = np.dot(var, np.dot(x.T, invcov))
        a_coef.append(coef[0])
        c_coef.append(coef[1])
        c_ivar.append(1.0/var[1,1])
    a_coef = np.array(a_coef)
    c_coef = np.array(c_coef)
    c_ivar = np.array(c_ivar)
    ia_coef = interpolate.interp1d(cvals, a_coef, axis=0, kind=interp_meth)
    ic_coef = interpolate.interp1d(cvals, c_coef, axis=0, kind=interp_meth)
    ic_ivar = interpolate.interp1d(cvals, c_ivar, axis=0, kind=interp_meth)

    return ia_coef, ic_coef, ic_ivar


def utr_rn(reads=None, filename=None, gain=2, return_im=False, header=fits.PrimaryHDU().header, biassub='all', phnoise=1.3, **kwargs):
    """
    Sample reads up-the-ramp in the read noise limit. We assume the counts 
    in each pixel obey the linear relation y_i = a + i*b*dt = a + i*c, 
    where i is an integer from 1 to nreads, and c = b*dt is the count. The 
    user can either pass the reads directly to this function or give the 
    data directory and file name containing the reads. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (nreads, 2048, 2112) or
                   (nreads, 2048, 2048), i.e. with or without the reference 
                   channel
    2. filename:   string, fits file name. Only needed when the reads 
                   are not given. Should include the full path to file.

    Optional inputs:
    1. sig_rn:     float, the std of the read noise in ADU 
    2. return_im:  bool, if True, return an Image class object

    Returns:
    1. c_arr:      2D ndarry of Image class object, best-fit 
                   count in each pixel
    """

    if reads is None:
        reads = getreads(filename, header, **kwargs)

    nreads = reads.shape[0]

    ###################################################################
    # If we are read noise limited, then the count (c = b*dt) is given
    # by 12/(N^3 - N)*sum((i - (N+1)/2)*y_i). We simply sum the reads, 
    # weighting each by (i - (N+1)/2). 
    ###################################################################

    factor = 12.0/(nreads**3 - nreads)
    weights = np.arange(1,nreads+1) - (nreads+1)/2.0
    c_arr = np.zeros(reads[0].shape)
    for i in range(nreads):
        c_arr += factor*weights[i]*reads[i]

    var = np.zeros((c_arr.shape))
    if biassub is not None:
        numchan = c_arr.shape[1]/64
        for j in xrange(numchan):
            if biassub=='top':
                refpix = c_arr[-4:, j*64:(j+1)*64]
            elif biassub=='bottom':
                refpix = c_arr[:4, j*64:(j+1)*64]
            elif biassub=='all':
                top = c_arr[-4:, j*64:(j+1)*64]
                bottom = c_arr[:4, j*64:(j+1)*64]
                refpix = np.concatenate([top, bottom])
            else:
                raise ValueError("Bias subtraction method must be one of 'top', 'bottom', or 'all'.")
            
            header['biassub'] = (biassub, 'Reference pixels used to correct ref voltage')
            sortedref = np.sort(refpix, axis=None)
            c_arr[:, j*64:(j+1)*64] -= np.mean(sortedref[1:-1])
            var[:, j*64:(j+1)*64] = np.var(sortedref[1:-1])
        
    
    #allrefpix = np.concatenate([c_arr[:4], c_arr[-4:], c_arr[4:-4, :4].T, 
    #                            c_arr[4:-4, -4:].T], axis=1)
    # Directly measure the read noise
    #readnoise = np.var(np.sort(allrefpix, axis=None)[5:-5])
    #header['readnois'] = (np.sqrt(readnoise), 'Effective read noise in the full ramp, ADU')
    #var = np.ones(c_arr.shape)*readnoise
    # Now add photon noise.  The factor of 1.3 is approximate and is from
    # the asymptotic performance of up-the-ramp sampling.  Divide by
    # nreads because we are using units of ADU per read.
    cts = c_arr[4:-4, 4:-4]
    var[4:-4, 4:-4] += phnoise*cts*(cts > 0)/gain/nreads
    ivar = 1./var

    if return_im:
        #ivar = (factor*sig_rn)**2*np.sum(weights**2)
        #ivar = (1.0/ivar)*np.ones(c_arr.shape)
        return Image(data=c_arr, ivar=ivar, header=header) 
    else:
        return c_arr

def utr(reads=None, filename=None, sig_rn=20.0, gain=2.0, 
        biassub='all', interp_meth='linear', calc_chisq=False, phnoise=1.3,
        header=fits.PrimaryHDU().header, **kwargs):
    """
    Sample reads up-the-ramp taking both shot noise and read noise 
    into account. We assume the counts in each pixel obey the linear 
    relation y_i = a + i*b*dt = a + i*c, where i is an integer from 
    1 to nreads, and c = b*dt is the count. The user can either pass 
    the reads directly to this function or give the data directory 
    and file name containing the reads. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (2048, 2112, nreads) or
                   (2048, 2048, nreads), i.e. with or without the reference 
                   channel
    2. filename:   string, fits file name. Only needed when the reads 
                   are not given. Should include the full path to file.

    Optional inputs:
    1. sig_rn:        float, the std of the read noise in ADU 
    2. gain:          float, the detector gain (electrons/ADU) 
    3. interp_meth:   string, the interpolation method to use when 
                      interpolating over the up-the-ramp coefficients.
                      Possible methods are the same as for scipy's interp1d.

    Returns:
    1. im             Image class object containing the count in electrons, 
                      ivar, and flags (not yet) for every pixel in the image. 
    """
    
    if reads is None:
        reads = getreads(filename, header, **kwargs)

    nreads = reads.shape[0]

    ###################################################################
    # Sample up-the-ramp in the read noise limit to get an estimate of 
    # the count for the covariance matrix calculation. Convert everthing
    # from ADU to electrons
    ###################################################################

    im = utr_rn(reads, header=header, biassub=biassub, gain=gain, phnoise=phnoise, return_im=True) # count rate (ADU) in RN limit
    #ivar_arr = np.ones(c_rn_arr.shape)

    ###################################################################
    # Generate interpolation objects for the up-the-ramp coefficients,
    # calculate the count (c) and intercept (a) for every pixel. The
    # calculation is done per row to conserve memory 
    ###################################################################

    if False: #calc_chisq:

        c_rn_arr *= gain
        sig_rn *= gain
        reads *= gain
        interp_objs = _interp_coef(nreads, sig_rn, c_rn_arr.min(),\
                                       c_rn_arr.max(), interp_meth=interp_meth)
        
        c_arr = np.zeros(c_rn_arr.shape)
        a_arr = np.zeros(c_rn_arr.shape)
        ivar_arr = np.zeros(c_rn_arr.shape)
        for row in xrange(c_rn_arr.shape[0]):
            a_coef = interp_objs[0](c_rn_arr[row,:])
            a_arr[row,:] = np.sum(a_coef*reads[:,row,:], axis=2)
            c_coef = interp_objs[1](c_rn_arr[row,:])
            c_arr[row,:] = np.sum(c_coef*reads[:,row,:], axis=2)
            ivar_arr[row,:] = interp_objs[2](c_rn_arr[row,:])
        del a_coef
        del c_coef

    ###################################################################
    # Calculate chi-squared for every pixel. The flags will be generated
    # here. Currently, we are calculating this in the read noise limit.
    # We are still thinking about how to optimize the general case. 
    ###################################################################

    if False: #calc_chisq:
        chisq = np.zeros(c_arr.shape)
        for i in range(nreads):
            chisq += (reads[:,:,i] - a_arr - c_arr*(i+1))**2
        chisq /= sig_rn**2 
        im = Image(data=c_arr, ivar=ivar_arr, chisq=chisq, header=header)
    else:
        pass #im = Image(data=c_rn_arr, ivar=ivar_arr, header=header)

    try:
        origname = re.sub('.*CRSA', '', re.sub('.fits', '', filename))
        im.header['origname'] = (origname, 'Original file ID number')
    except:
        pass

    return im
