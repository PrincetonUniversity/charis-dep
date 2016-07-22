#!/usr/bin/env python

import numpy as np
from scipy import interpolate, signal
from image import Image
import copy
from collections import OrderedDict
import matutils
import multiprocessing
import time

def _smoothandmask(cube, ivar, good):
    """
    
    """
    x = np.arange(7) - 3
    x, y = np.meshgrid(x, x)
    widewindow = np.exp(-(x**2 + y**2))
    narrowwindow = np.exp(-3*(x**2 + y**2))
    widewindow /= np.sum(widewindow)
    narrowwindow /= np.sum(narrowwindow)

    for i in range(cube.shape[0]):
        ivar_thresh = signal.convolve2d(ivar[i], widewindow, mode='same')
        ivar[i] *= ivar[i] > ivar_smooth/4.
        
        mask = signal.convolve2d(cube[i]*ivar[i], narrowwindow, mode='same')
        mask /= signal.convolve2d(ivar[i], narrowwindow, mode='same')
        indx = np.where(np.all([ivar == 0, good], axis=0))
        cube[i][indx] = mask[indx]

    return None

def _fit_cutout(subim, psflets, bounds, x=None, y=None, mode='lstsq'):
    """
    Fit a series of PSFlets to an image, recover the best-fit coefficients.
    This is currently little more than a wrapper for np.linalg.lstsq, but 
    could be more complex if/when we regularize the problem or adopt some
    other approach.

    Inputs:
    1. subim:   2D nadarray, microspectrum to fit
    2. psflets: 3D ndarray, first dimension is wavelength.  psflets[0] 
                must match the shape of subim.
    3. mode:    string, method to use.  Currently limited to lstsq (a 
                simple least-squares fit using linalg.lstsq), this can
                be expanded to include an arbitrary approach.
    
    Returns:
    1. coef:    the best-fit coefficients (i.e. the microspectrum).

    Note: this routine may also return the covariance matrix in the future.
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

    Inputs:
    1. im:      Image object containing data to be fit
    2. x0:      float, x location of lenslet centroid at shortest 
                wavelength
    3. y0:      float, y location of lenslet centroid at shortest 
                wavelength
    4. psflets: list of 2D ndarrays, each of which should have the
                same shape as image.
         
    Optional inputs:
    5. dy:      vertical length to cut out, default 30.  E.g.
                cut out from y0-dy to y0+dx (inclusive).
    6. dx:      horizontal length to cut out, default 3.  This is
                the length to cut out in the +/-x direction; the 
                length cut out in the +y direction (beyond the 
                shortest wavelength) is also dx.

    Returns: 
    1. subim:   a flattened subimage to be fit
    2. psflet_subarr: a 2D ndarray, first dimension is wavelength,
                second dimension is spatial, and is the same shape
                as the flattened subimage.

    Note: both subim and psflet_subarr are scaled by the inverse
    standard deviation if it is given for the input Image.  This 
    will make the fit chi2 and properly handle bad/masked pixels.

    """

    ###################################################################
    # Note: x0 - dy is intentional--the horizontal spacing of 
    # spectra is similar to the vertical spacing and is distinct
    # from the spectral length.
    ###################################################################

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

    Inputs:
    1. shape:  tuple, shape of the image and psflet arrays.
    2. x:      ndarray, x indices of the PSFlet centroids at a given
               wavelength
    3. y:      ndarray, y indices of the PSFlet centroids
    4. good:   ndarray, boolean True if the PSFlet falls on the detector

    Output:
    1. psflet_indx: ndarray of the requested input shape (matching the
               PSFlet image).  The array contains the indices of the
               closest lenslet to each pixel at the wavelength of x and y

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


def fit_spectra(im, psflets, lam, x, y, good, header=OrderedDict(), 
                refine=True, maxcpus=None):

    """
    Fit the microspectra to produce a data cube.  The heavy lifting is
    done with a call to _fit_cutout; it is here that we can implement
    regularlizations, aperture photometry, etc., etc.  There is also a 
    framework to iteratively solve for the coefficients accounting for
    nearest-neighbor crosstalk.

    Inputs:
    1. im:      Image class, im.data is the data to be fit.
    2. psflets: list of Image classes, each one having data of the same 
                shape as im.data.  These are the monochromatic spots.
    3. lam:     array of wavelengths extracted.  Should match the first
                dimension of psflets.
    4. x:       list of ndarrays, x position of each lenslet spot at 
                each wavelength
    5. y:       list of ndarrays, y position of each lenslet spot at 
                each wavelength
    6. good:    list of boolean arrays, true if lenslet spot lies 
                within the H2RG
    Optional input:
    1. header:  ordered dictionary or FITS header class, to store some
                information about the reduction.
    
    Note: the 'x', 'y', and 'good' inputs are assumed to be the
          outputs from the function locatePSFlets.

    Output:
    1. coefs:   An Image class containing the data cube.

    Note: these routines remain a work in progress.  

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
        isig = np.sqrt(im.ivar)
    else:
        isig = np.ones(im.data.shape)

    ###################################################################
    # Need to make copies of the data arrays because FITS is
    # big-endian, but this not necessarily the native order on the
    # machine and can cause errors with cython code.
    ###################################################################

    data = np.empty(im.data.shape)
    data[:] = im.data
    psflets2 = np.empty(psflets.shape)
    psflets2[:] = psflets

    ###################################################################
    # Call cython implementations of _get_cutouts and _fit_cutouts.
    # Factor of 5 or so speedup on a 16 core machine.
    ###################################################################

    ncpus = multiprocessing.cpu_count()
    if maxcpus is not None:
        maxcpus = min(maxcpus, ncpus)
    else:
        maxcpus = ncpus
    
    A, b, size = matutils.allcutouts(data, isig, xint, yint, indx, psflets2, maxproc=maxcpus)
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
    #coefs = matutils.lstsq(A, b, indx, size, nlens, maxproc=maxcpus).T.reshape(coefshape)

    ###################################################################
    # Subtract the best fit spectrum to include crosstalk.
    # Run the above routine again to get the perturbations to the
    # intial guesses of the coefficients.
    ###################################################################
    
    if refine:
        for i in range(len(psflets)):
            psflet_indx = _tag_psflets(psflets[i].shape, x[i], y[i], good[i])
            coefs_flat = np.reshape(coefs[i], -1)
            data -= psflets[i]*coefs_flat[psflet_indx]

        A, b, size = matutils.allcutouts(data, isig, xint, yint, indx, psflets2, maxproc=maxcpus)
        coefs += matutils.lstsq(A, b, indx, size, nlens, maxproc=maxcpus).T.reshape(coefshape)

    header['cubemode'] = ('leastsq', 'Method used to extract data cube')
    header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
    header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
    header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
    header['nlam'] = (lam.shape[0], 'Number of extracted wavelengths')
    datacube = Image(data=coefs, ivar=1./cov, header=header)

    _smoothandmask(datacube.data, datacube.ivar, 
                   np.reshape(goodint, tuple(list(coefshape)[1:])))

    return datacube


def fitspec_intpix(im, PSFlet_tool, lam, delt_x=7, header=OrderedDict()):
    """
    """

    xindx = PSFlet_tool.xindx
    yindx = PSFlet_tool.yindx
    Nmax = PSFlet_tool.nlam_max

    x = np.arange(im.data.shape[1])
    y = np.arange(im.data.shape[0])
    x, y = np.meshgrid(x, y)

    coefs = np.zeros(tuple([max(Nmax, lam.shape[0])] + list(xindx.shape)[:-1]))
    
    xarr, yarr = np.meshgrid(np.arange(delt_x), np.arange(Nmax))

    loglam = np.log(lam)

    for i in range(xindx.shape[0]):
        for j in range(yindx.shape[1]):
            _x = xindx[i, j, :PSFlet_tool.nlam[i, j]]
            _y = yindx[i, j, :PSFlet_tool.nlam[i, j]]
            _lam = PSFlet_tool.lam_indx[i, j, :PSFlet_tool.nlam[i, j]]

            if not (np.all(_x > x[0, 10]) and np.all(_x < x[0, -10]) and 
                    np.all(_y > y[10, 0]) and np.all(_y < y[-10, 0])):
                continue


            i1 = int(np.mean(_x) - delt_x/2.)
            dx = _x[yarr[:len(_lam)]] - x[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            #var = _var[yarr[:len(_lam)]] - x[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            sig = 0.7
            weight = np.exp(-dx**2/2./sig**2)
            data = im.data[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            if im.ivar is not None:
                ivar = im.ivar[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            else:
                ivar = np.ones(data.shape)

            coefs[:len(_lam), i, j] = np.sum(weight*data*ivar, axis=1)[::-1]
            coefs[:len(_lam), i, j] /= np.sum(weight**2*ivar, axis=1)[::-1]

            tck = interpolate.splrep(np.log(_lam[::-1]), coefs[:len(_lam), i, j], s=0, k=3)
            coefs[:loglam.shape[0], i, j] = interpolate.splev(loglam, tck, ext=1)

    header['cubemode'] = ('optext', 'Method used to extract data cube')
    header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
    header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
    header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
    header['nlam'] = (lam.shape[0], 'Number of extracted wavelengths')
    datacube = Image(data=coefs[:loglam.shape[0]], header=header)
    return datacube

