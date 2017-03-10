#!/usr/bin/env python

import numpy as np
from scipy import interpolate, signal, stats
try:
    from charis.image import Image
except:
    from image import Image

import logging
import matutils
import multiprocessing
import time
from astropy.io import fits

log = logging.getLogger('main')

def _smoothandmask(datacube, good):
    """
    Set bad spectral measurements to an inverse variance of zero.  The
    threshold for effectively discarding data is an inverse variance
    less than 10% of the smoothed inverse variance, i.e., a
    measurement much worse than the surrounding ones.  This rejection
    is done separately at each wavelength.  

    Then use a smoothed, inverse-variance-weighted map to replace the
    values of the masked spectral measurements.  Note that this last
    step is purely cosmetic as the inverse variances are, in any case,
    zero.

    Parameters
    ----------
    datacube: image instance
            containing 3D arrays data and ivar
    good:     2D array
            nonzero = good lenslet

    Returns
    -------
    datacube: input datacube modified in place

    """

    ivar = datacube.ivar
    cube = datacube.data

    x = np.arange(7) - 3
    x, y = np.meshgrid(x, x)
    widewindow = np.exp(-(x**2 + y**2))
    narrowwindow = np.exp(-2*(x**2 + y**2))
    widewindow /= np.sum(widewindow)
    narrowwindow /= np.sum(narrowwindow)

    for i in range(cube.shape[0]):
        ivar_smooth = signal.convolve2d(ivar[i], widewindow, mode='same')
        ivar[i] *= ivar[i] > ivar_smooth/10.
        
        mask = signal.convolve2d(cube[i]*ivar[i], narrowwindow, mode='same')
        mask /= signal.convolve2d(ivar[i], narrowwindow, mode='same') + 1e-100
        indx = np.where(np.all([ivar[i] == 0, good], axis=0))
        cube[i][indx] = mask[indx]

    return datacube

def _trimmed_mean(arr, n=2, axis=None, maskval=0):

    """
    Return the trimmed mean of an input array.

    Parameters
    ----------
    arr:     ndarray
        Data to trim
    n:       integer
        Number of points to trim at each end
    axis:    int
        The axis along which to compute the trimmed mean.  
        If None, compute the trimmed mean on the 
        flattened array.  Default None.
    maskval: float
        Value to mask in trimmed mean calculation.  NaN
        and inf values are already masked.  Default 0.

    Returns
    -------
    trimmed_mean: ndarray
        Returns the trimmed mean
    """

    arr_sorted = arr.copy()
    shape = arr_sorted.shape

    ########################################################
    # Sort with masked values, NaN, inf at the end,
    # Trim the lowest n unmasked entries
    ########################################################

    arr_sorted[np.where(np.logical_not(np.isfinite(arr_sorted)))] = np.inf
    if maskval is not None:
        arr_sorted[np.where(arr == maskval)] = np.inf
    arr_sorted = np.sort(arr_sorted, axis=axis)
    if axis > 0:
        arr_sorted = np.take(arr_sorted, np.arange(n, shape[axis]), axis=axis)
    else:
        arr_sorted = arr_sorted[n:]

    ########################################################
    # Move masked values to the beginning, 
    # trim the largest n unmasked entries
    ########################################################

    arr_sorted[np.where(np.isinf(arr_sorted))] *= -1
    if axis > 0:
        arr_sorted = np.take(arr_sorted, np.arange(0, shape[axis] - n), axis=axis)
    elif n > 0:
        arr_sorted = np.sort(arr_sorted, axis=axis)[:-n]
    else:
        arr_sorted = np.sort(arr_sorted, axis=axis)

    ########################################################
    # Replace mask with zero and return trimmed mean.
    # No data --> return zero
    ########################################################

    if maskval is not None:
        norm = np.sum(np.isfinite(arr_sorted), axis=axis)
        arr_sorted[np.where(np.isinf(arr_sorted))] = 0
        return np.sum(arr_sorted, axis=axis)/(norm + 1e-100)
    else:
        return np.mean(arr_sorted, axis=axis)


def _get_corrnoise(resid, ivar):

    """
    Private function that returns the correlated noise
    
    Parameters
    ----------
    resid: ndarray
        Residuals of the psflet fit
    ivar: ndarray
        Inverse variance of the data
        
    Returns
    -------
    corrnoise: ndarray
        Correlated noise map
    """

    mask = np.zeros(resid.shape)
    corrnoise = np.zeros(resid.shape)
    dx = 64
    
    var_ratios = np.zeros((resid.shape[0], resid.shape[1]))
    for i in range(0, resid.shape[1]//dx):
        ivar_ref = np.median(ivar[:4, i*dx:(i + 1)*dx])
        var_ratios[:, i*dx:(i + 1)*dx] = ivar[:, i*dx:(i + 1)*dx]/ivar_ref

    ##################################################################
    # Default threshold: the variance is equal parts read noise and
    # other sources.  However, always use at least 40% of the pixels
    # to ensure a reasonable average.
    ##################################################################

    thresh = min(1/np.sqrt(2), stats.scoreatpercentile(var_ratios, 60))

    for i in range(resid.shape[1]//dx):
        ivar_ref = np.median(ivar[:4, i*dx:(i + 1)*dx])
        mask[:, i*dx:(i + 1)*dx] = ivar[:, i*dx:(i + 1)*dx] > thresh*ivar_ref

    masked = resid*mask
    stripe = np.zeros((resid.shape[1]//(2*dx), resid.shape[0], dx))

    ##################################################################
    # Do the even channels (j=0) and odd channels (j=dx) separately.
    # Correlated read noise is very different between even and odd
    # channels.  Compute the trimmed mean of the <=16 unmasked pixels
    # in each set.  Each channel has its own coupling to this shared
    # noise, which is estimated by corr.
    ##################################################################

    for j in [0, dx]:
        for i in range(0, resid.shape[1], 2*dx):
            stripe[i//(2*dx)] = masked[:, i+j:i+j+dx]
        noisemed = _trimmed_mean(stripe, axis=0, n=1, maskval=0)
        for i in range(0, resid.shape[1], 2*dx):
            corr = _trimmed_mean(noisemed*masked[:, i+j:i+j+dx], n=10, maskval=0)
            corr /= _trimmed_mean(noisemed**2, n=10, maskval=0)
            stripe[i//(2*dx)] /= corr
        noisemed = _trimmed_mean(stripe, axis=0, n=1, maskval=0)
        for i in range(0, resid.shape[1], 2*dx):
            corr = _trimmed_mean(noisemed*masked[:, i+j:i+j+dx], n=10, maskval=0)
            corr /= _trimmed_mean(noisemed**2, n=10, maskval=0)
            corrnoise[:, i+j:i+j+dx] = corr*noisemed

    return corrnoise

def _recalc_ivar(data, ivar):

    """
    Private function to recalculate the inverse variance
    """

    dx = 64
    var = 1/(ivar + 1e-100)
    for i in range(32):
        rdnoise_old = np.sqrt(np.median(var[:4, i*dx:(i + 1)*dx]))
        rdnoise_new = np.std(np.sort(data[:4, i*dx:(i + 1)*dx])[1:-1])
        var[:, i*dx:(i + 1)*dx] += 0.5*(rdnoise_new**2 - rdnoise_old**2)
    return (1/var)*(ivar > 0)

def _get_corrnoise_lowf(fit, resid, ivar, sig=250):

    """
    Subtract a highly smoothed map of the residuals to get
    rid of pow over large spatial scales (long timescales).
    """

    corrnoise = np.zeros(fit.shape)
    dx = 64
    
    x = np.arange(4*sig + 1) - 2*sig
    window = np.exp(-x**2/(2*sig**2))
    window /= np.sum(window)
    allstripes = np.zeros((32, fit.shape[0]))

    for i in range(32):
        
        #################################################################
        # We'll use the pixels where the fitted PSF intensity is modest
        # compared to the read noise so that the residual should be 
        # dominated by read noise.  Don't use reference pixels.
        #################################################################

        mask = fit[:, i*dx:(i + 1)*dx]**2 < 15/np.mean(ivar[-4:, i*dx:(i + 1)*dx])
        mask *= ivar[:, i*dx:(i + 1)*dx] > 0

        #################################################################
        # Don't use spectra for which any PSFlet lies near the detector
        # edge.  Spectral length is about 30 pixels.
        #################################################################

        mask[:40] = mask[-40:] = 0
        if i == 0:
            mask[:, :10] = 0
        elif i == 31:
            mask[:, -10:] = 0

        stripe = resid[:, i*dx:(i + 1)*dx]*mask            
        allstripes[i] = _trimmed_mean(stripe, axis=1, maskval=0, n=1)

        npts = np.sum(mask, axis=1)
        stripe = np.sum(stripe, axis=1)
            
        stripe = np.convolve(stripe, window, mode='same')
        stripe /= np.convolve(npts, window, mode='same') + 2
        
        for j in range(dx):
            corrnoise[:, i*dx + j] = stripe

    return corrnoise


def _fit_cutout(subim, psflets, bounds, x=None, y=None, mode='lstsq'):
    """
    Fit a series of PSFlets to an image, recover the best-fit coefficients.
    This is currently little more than a wrapper for np.linalg.lstsq, but 
    could be more complex if/when we regularize the problem or adopt some
    other approach.

    Parameters
    ----------
    subim:   2D nadarray
        Microspectrum to fit
    psflets: 3D ndarray
        First dimension is wavelength. psflets[0] must match the shape of subim.
    mode:    string
        Method to use. Currently limited to lstsq, ext, and apphot.
        lstsq highly recommended.
    
    Returns
    -------
    coef:    list of floats
        The best-fit coefficients (i.e. the microspectrum).

    Notes
    -----
    This routine may also return the covariance matrix in the future.
    It will depend on the performance of the algorithms and whether/how we
    implement regularization.
    """

    try:
        if not subim.shape == psflets[0].shape:
            raise ValueError("subim must be the same shape as each psflet.")
    except:
        raise ValueError("subim must be the same shape as each psflet.")
    
    if mode == 'lstsq':
        subim_flat = np.reshape(subim, -1)
        psflets_flat = np.reshape(psflets, (psflets.shape[0], -1))
        coef = np.linalg.lstsq(psflets_flat.T, subim_flat)[0]
    elif mode == 'ext':
        coef = np.zeros(psflets.shape[0])
        for i in range(psflets.shape[0]):
            coef[i] = np.sum(psflets[i]*subim)/np.sum(psflets[i])
    elif mode == 'apphot':
        coef = np.zeros((subim.shape[0]))
        for i in range(subim.shape[0]):
            coef[i] = np.sum(subim[i])
    else:
        raise ValueError("mode " + mode + " to fit microspectra is not currently implemented.")

    return coef


def _get_cutout(im, x, y, psflets, dx=3):
    
    """
    Cut out a microspectrum for fitting.  Return the inputs to 
    linalg.lstsq or to whatever regularization scheme we adopt.
    Assumes that spectra are dispersed in the -y direction.

    Parameters
    ----------
    im:      Image object
        Image containing data to be fit
    x0:      float
        x location of lenslet centroid at shortest wavelength
    y0:      float
        y location of lenslet centroid at shortest wavelength
    psflets: list of 2D ndarrays
        Each of which should have the same shape as image.
    dx:     int
        Horizontal length to cut out, default 3. 
        This is the length to cut out in the +/-x direction; 
        the lengths cut out in the +y direction 
        (beyond the shortest and longest wavelengths) are also dx.

    Returns
    -------
    subim:   1D array
        A flattened subimage to be fit
    psflet_subarr: 2D ndarray
        First dimension is wavelength second dimension is spatial,
        and is the same shape as the flattened subimage.

    Notes
    -----
    Both subim and psflet_subarr are scaled by the inverse
    standard deviation if it is given for the input Image.  This 
    will make the fit chi2 and properly handle bad/masked pixels.

    """

    y0, y1 = [int(np.amin(y)) - dx, int(np.amax(y)) + dx + 1]
    x0, x1 = [int(np.amin(x)) - dx, int(np.amax(x)) + dx + 1]

    subim = im.data[y0:y1, x0:x1]
    if im.ivar is not None:
        isig = np.sqrt(im.ivar[y0:y1, x0:x1])
        subim *= isig

    subarrshape = tuple([len(psflets)] + list(subim.shape))
    psflet_subarr = np.zeros(subarrshape)
    for i in range(len(psflets)):
        psflet_subarr[i] = psflets[i][y0:y1, x0:x1]
        if im.ivar is not None:
            psflet_subarr[i] *= isig

    return subim, psflet_subarr, [y0, y1, x0, x1]


def _tag_psflets(shape, x, y, good):
    """
    Create an array with the index of each lenslet at a given
    wavelength.  This will make it very easy to remove the best-fit
    spectra accounting for nearest-neighbor crosstalk.

    Parameters
    ----------
    shape:  tuple
        Shape of the image and psflet arrays.
    x:      ndarray
        x indices of the PSFlet centroids at a given wavelength
    y:      ndarray
        y indices of the PSFlet centroids
    good:  boolean ndarray
        True if the PSFlet falls on the detector

    Returns
    -------
    psflet_indx: ndarray
        Has the requested input shape (matching the PSFlet image). 
        The array contains the indices of the closest lenslet to each
        pixel at the wavelength of x and y

    Notes
    -----
    The output, psflet_indx, is to be used as follows:
    coefs[psflet_indx] will give the scaling of the monochromatic PSF-let
    frame.

    """
    
    psflet_indx = np.zeros(shape, np.int)
    oldshape = x.shape
    x_int = (np.reshape(x + 0.5, -1)).astype(int)
    y_int = (np.reshape(y + 0.5, -1)).astype(int)
    good = np.reshape(good, -1)

    for i in range(x_int.shape[0]):
        if good[i]:
            psflet_indx[y_int[i] - 6:y_int[i] + 7, x_int[i] - 6:x_int[i] + 7] = i

    good = np.reshape(good, oldshape)

    return psflet_indx


def fit_spectra(im, psflets, lam, x, y, good, header=fits.PrimaryHDU().header,
                flat=None, refine=False, smoothandmask=True, returnresid=False,
                suppressrdnse=False, return_corrnoise=False,
                maxcpus=multiprocessing.cpu_count()):

    """
    Fit the microspectra to produce a data cube.  The heavy lifting is
    done with a call to _fit_cutout; it is here that we can implement
    regularlizations, aperture photometry, etc., etc.  There is also a 
    framework to iteratively solve for the coefficients accounting for
    nearest-neighbor crosstalk.

    Parameters
    ----------
    im:      Image instance
        im.data is the data to be fit.
    psflets: list of Image instances
        each one having data of the same shape as im.data. 
        These are the monochromatic spots.
    lam:     1D array of floats
        Array of wavelengths extracted.  Should match the first dimension of psflets.
    x:       list of ndarrays
        x position of each lenslet spot at each wavelength
    y:       list of ndarrays
        y position of each lenslet spot at each wavelength
    good:    list of boolean arrays
        true if lenslet spot lies within the H2RG
    
    header:  FITS header instance
        Ordered dictionary or FITS header class,
        to store some information about the reduction.
    flat:    ndarray
        Lenslet flatfield.  Do not flatfield if this array is not given.  Default None.
    refine:  boolean
        Iterate solution to remove crosstalk? Approximately doubles runtime.  Default True.
    smoothandmask: boolean
        Set inverse variance of particularly noisy lenslets to zero and
        replace their values with interpolations for cosmetics
        (ivar will still be zero)? Default True.
    returnresid: boolean
        Return a residual image in addition the data cube?  Default False.
    suppressrdnse: boolean
    maxcpus:  int
        Maximum number of threads for OpenMP parallelization in Cython.
        Default multiprocessing.cpu_count().
    
    Returns
    -------
    datacube: Image instance
        An Image instance containing the data cube.

    Notes
    -----
    the 'x', 'y', and 'good' inputs are assumed to be the outputs from the function
    locatePSFlets.


    """
    
    ###################################################################
    # Fit the spectrum by minimizing chi squared
    ###################################################################

    coefshape = tuple([len(x)] + list(x[0].shape))
    
    x = np.asarray(x)
    y = np.asarray(y)

    xint = np.reshape((x + 0.5).astype(np.int64), (x.shape[0], -1))
    yint = np.reshape((y + 0.5).astype(np.int64), (y.shape[0], -1))
    goodint = np.reshape(np.prod(good.astype(np.int64), axis=0), -1)
    indx = np.where(goodint)[0]
    if im.ivar is not None:
        isig = np.sqrt(im.ivar).astype(np.float64)
    else:
        isig = np.ones(im.data.shape)
    if flat is not None:
        lenslet_ok = (flat > 0).astype(np.float64)

    ###################################################################
    # Need to make copies of the data arrays because FITS is
    # big-endian, but this not necessarily the native order on the
    # machine and can cause errors with cython code.
    ###################################################################

    data = np.empty(im.data.shape)
    data[:] = im.data
    if psflets.dtype != 'float64':
        psflets2 = np.empty(psflets.shape)
        psflets2[:] = psflets
        psflets = psflets2

    ###################################################################
    # Call cython implementations of _get_cutouts and _fit_cutouts.
    # Factor of 5 or so speedup on a 16 core machine.
    ###################################################################

    A, b, size = matutils.allcutouts(data, isig, xint, yint, indx, psflets, maxproc=maxcpus)
    nlens = xint.shape[1]

    ###################################################################
    # Get the covariance matrix for free. 
    # Discard off-diagonal elements for now, both to save space and 
    # because I don't know how to use them in practice.
    ###################################################################

    coefs, cov = matutils.lstsq(A, b, indx, size, nlens, returncov=1, maxproc=maxcpus)
    coefs = coefs.T.reshape(coefshape)
    cov = cov[:, np.arange(cov.shape[1]), np.arange(cov.shape[1])]
    cov = cov.T.reshape(coefshape)

    ###################################################################
    # Zero covariance = no data.  Set to infinity so that ivar = 0.
    ###################################################################

    cov[np.where(cov == 0)] = np.inf

    ###################################################################
    # Subtract the best fit spectrum to include crosstalk.
    # Run the above routine again to get the perturbations to the
    # intial guesses of the coefficients.
    ###################################################################
    
    if refine or returnresid or suppressrdnse:

        ###############################################################
        # Compute residual from fit.
        ###############################################################

        for i in range(len(psflets)):
            psflet_indx = _tag_psflets(psflets[i].shape, x[i], y[i], good[i])
            coefs_flat = np.reshape(coefs[i], -1)
            if flat is not None:
                coefs_flat *= np.reshape(lenslet_ok, -1)
            data -= psflets[i]*coefs_flat[psflet_indx]

        ###############################################################
        # Estimate shared noise and subtract it off.  Recompute the
        # data cube, inverse variance, and residuals.
        ###############################################################

        if suppressrdnse:
            corrnoise = _get_corrnoise(data, im.ivar)
            data[:] = im.data - corrnoise
            im.ivar = _recalc_ivar(data, im.ivar)
            isig = np.sqrt(im.ivar)
            A, b, size = matutils.allcutouts(data, isig, xint, yint, indx, psflets, maxproc=maxcpus)
            coefs, cov = matutils.lstsq(A, b, indx, size, nlens, returncov=1, maxproc=maxcpus)
            coefs = coefs.T.reshape(coefshape)
            cov = cov[:, np.arange(cov.shape[1]), np.arange(cov.shape[1])]
            cov = cov.T.reshape(coefshape)
            cov[np.where(cov == 0)] = np.inf
            for i in range(len(psflets)):
                psflet_indx = _tag_psflets(psflets[i].shape, x[i], y[i], good[i])
                coefs_flat = np.reshape(coefs[i], -1)
                if flat is not None:
                    coefs_flat *= np.reshape(lenslet_ok, -1)
                data -= psflets[i]*coefs_flat[psflet_indx]

            dcorrnoise = _get_corrnoise(data, im.ivar)
            data -= dcorrnoise
            corrnoise += dcorrnoise
            
            fit = im.data - data - corrnoise
            dcorrnoise = _get_corrnoise_lowf(fit, data, im.ivar, sig=250)
            data -= dcorrnoise
            corrnoise += dcorrnoise
            
            if return_corrnoise:
                return corrnoise

        else:
            corrnoise = 0

        if returnresid:
            resid = Image(data=data, ivar=im.ivar)
          
        ###############################################################
        # Compute the perturbation to the data cube after subtracting
        # nearest-neighbor crosstalk.  
        ###############################################################

        if refine:
            A, b, size = matutils.allcutouts(data, isig, xint, yint, indx, 
                                             psflets, maxproc=maxcpus)
            dcoefs = matutils.lstsq(A, b, indx, size, nlens, maxproc=maxcpus).T.reshape(coefshape)
            if flat is not None:
                coefs += dcoefs*lenslet_ok
            else:
                coefs += dcoefs

            if returnresid:
                data[:] = im.data - corrnoise
                for i in range(len(psflets)):
                    psflet_indx = _tag_psflets(psflets[i].shape, x[i], y[i], good[i])
                    coefs_flat = np.reshape(coefs[i], -1)                
                    data -= psflets[i]*coefs_flat[psflet_indx]
                resid = Image(data=data, ivar=im.ivar)
    
    header['cubemode'] = ('Chi^2 Fit to PSFlets', 'Method used to extract data cube')

    header['reducern'] = (suppressrdnse, 'Suppress read noise using low ct rate pixels?')
    header['refine'] = (refine, 'Iterate solution to remove crosstalk?')
    header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
    header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
    if len(lam) > 1:
        header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
    header['nlam'] = (len(lam), 'Number of extracted wavelengths')

    datacube = Image(data=coefs, ivar=1./cov, header=header)

    if flat is not None:
        datacube.data /= flat + 1e-10
        datacube.ivar *= flat**2

    if smoothandmask:
        datacube = _smoothandmask(datacube, np.reshape(goodint, tuple(list(coefshape)[1:])))
    
    datacube.header['maskivar'] = (smoothandmask, 'Set poor ivar to 0, smoothed I for cosmetics')

    if returnresid:
        return datacube, resid
    else:
        return datacube


def optext_spectra(im, PSFlet_tool, lam, delt_x=7, flat=None, sig=0.7, 
                   smoothandmask=True, header=fits.PrimaryHDU().header,
                   maxcpus=multiprocessing.cpu_count()):
    """
    Function optext_spectra performs an "optimal" extraction of the
    microspectra (a truly optimal extraction would require more
    accurate dispersions; this will be implemented).  Aperture
    photometry may also be performed by setting sig=1e10 (or something
    similarly large); the aperture will then be delt_x.

    Parameters
    ----------
    im:     Image instance
        Containing the 2D count rates and inverse variances
    PSFlet_tool: PSFLet_tool instance
        An instance of the PSFlet_tool class containing the wavelength solution. 
        The locations of the microspectra and the wavelengths corresponding to 
        whole pixels along the dispersion direction will be used in this routine.
    lam:         list of floats
        List of floating point numbers or 1D array. The array of wavelengths onto which
        to interpolate the microspectra.
    delt_x:      int
        Odd, positive integer, the aperture width for extraction.  Default 7 (5 or 7 recommended).
    flat:        2D ndarray
        2D floating point ndarray, lenslet flat. Default None (don't flat-field).
    sig:         float
        1D root variance of lenslet PSF. Default 0.7 (currently cannot be a function of 
        wavelength).  Setting sig >> delt_x is equivalent to aperture photometry.
    smoothandmask: boolean
        Set inverse variance of particularly noisy lenslets to zero and replace their values 
        with interpolations for cosmetics (ivar will still be zero)?  Default True.
    header:      FITS header
        Ordered dictionary or FITS header class, to store some information about the reduction.
    maxcpus      int
        Maximum number of threads for OpenMP parallelization in Cython. 
        Default multiprocessing.cpu_count().
    
    Returns
    -------
    datacube:    Image instance
        An Image instance containing the data cube.

    Notes
    -----
    In the near future a position- and wavelength-dependent PSFlet
    width will be included from the measured PSFlets.

    """

    loglam = np.log(lam)

    ########################################################################
    # x-locations of the centers of the microspectra.  The y locations
    # are integer pixels, and the wavelengths in PSFlet_tool.lam_indx
    # are given at the centers of the pixels.  Dispersion is in the
    # y-direction.  Make copies of all arrays to ensure that they are
    # in native byte order as required by Cython.
    ########################################################################

    xindx = np.zeros(PSFlet_tool.xindx.shape)
    xindx[:] = PSFlet_tool.xindx
    yindx = np.zeros(PSFlet_tool.yindx.shape)
    yindx[:] = PSFlet_tool.yindx
    loglam_indx = np.log(PSFlet_tool.lam_indx + 1e-100)
    nlam = np.zeros(PSFlet_tool.nlam.shape, np.int32)
    nlam[:] = PSFlet_tool.nlam
    Nmax = max(PSFlet_tool.nlam_max, lam.shape[0])

    data = np.zeros(im.data.shape)
    data[:] = im.data
    ivar = np.zeros(im.ivar.shape)
    ivar[:] = im.ivar

    coefs, tot_ivar = matutils.optext(data, ivar, xindx, yindx, 
                                      loglam_indx, nlam, loglam, Nmax, 
                                      delt_x=delt_x, sig=sig, maxproc=maxcpus)

    header['cubemode'] = ('Optimal Extraction', 'Method used to extract data cube')
    header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
    header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
    header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
    header['nlam'] = (lam.shape[0], 'Number of extracted wavelengths')

    datacube = Image(data=coefs, ivar=tot_ivar, header=header)

    if flat is not None:
        datacube.data /= flat + 1e-10
        datacube.ivar *= flat**2

    if smoothandmask:
        good = np.any(datacube.data != 0, axis=0)
        datacube = _smoothandmask(datacube, good)

    return datacube

