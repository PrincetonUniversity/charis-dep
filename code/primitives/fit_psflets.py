#!/usr/bin/env python

import numpy as np
from image import Image


def _fit_cutout(subim, psflets, mode='lstsq'):
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
    else:
        raise ValueError("mode " + mode + " to fit microspectra is not currently implemented.")

    return coef


def _get_cutout(im, x0, y0, psflets, dy=30, dx=3):
    
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

    subim = im.data[y0 - dy:y0 + dx + 1, x0 - dx:x0 + dx + 1]
    if im.ivar is not None:
        isig = np.sqrt(im.ivar[y0 - dy:y0 + dx + 1, x0 - dx:x0 + dx + 1])
        subim *= isig

    subarrshape = tuple([len(psflets)] + list(subim.shape))
    psflet_subarr = np.zeros(subarrshape)
    for i in range(len(psflets)):
        psflet_subarr[i] = psflets[i].data[y0 - dy:y0 + dx + 1, x0 - dx:x0 + dx + 1]
        if im.ivar is not None:
            psflet_subarr[i] *= isig

    return subim, psflet_subarr


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
               closest lenslet to each pixel at the avelength of x and y

    The output, psflet_indx, is to be used as follows:
    coefs[psflet_indx] will give the scaling of the monochromatic PSF-let
    frame.

    """
    
    psflet_indx = np.zeros(shape, np.int)
    oldshape = x.shape
    x_int = (np.reshape(x + 0.5, -1)).astype(int)
    y_int = (np.reshape(y + 0.5, -1)).astype(int)
    good = np.reshape(good, -1)

    for i in range(x.shape[0]):
        if good[i]:
            psflet_indx[y_int[i] - 6:y_int[i] + 7, x_int[i] - 6:x_int[i] + 7] = i

    good = np.reshape(good, oldshape)

    return psflet_indx


def fit_spectra(im, psflets, x, y, good):

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
    3. x:       list of ndarrays, x position of each lenslet spot at 
                each wavelength
    4. y:       list of ndarrays, y position of each lenslet spot at 
                each wavelength
    
    Note: the 'x', 'y', and 'good' inputs are assumed to be the
          outputs from the function locatePSFlets.

    Output:
    1. coefs:   An Image class containing the data cube.

    Note: these routines are very much a work in progress.  

    """
    
    ###################################################################
    # Fit the spectrum
    ###################################################################

    coefs = np.zeros(tuple([len(x)] + list(x[0].shape)))
    resid = im.data.copy()
    
    for i in range(x[0].shape[0]):
        for j in range(x[0].shape[1]):
            if good[0][i, j] and good[-1][i, j]:
                subim, psflet_subarr = _get_cutout(im, x[0][i, j] + 0.5,
                                                   y[0][i, j], psflets)
                coefs[:, i, j] = _fit_cutout(subim, psflet_subarr)

    ###################################################################
    # Subtract the best fit spectrum to include crosstalk
    # Run the above routine again to get the perturbations to the
    # intial guesses of the coefficients.
    ###################################################################
    
    for i in range(len(psflets)):
        psflet_indx = _tag_psflets(psflets[i].data.shape, x[i], y[i], good[i])
        coefs_flat = np.reshape(coefs[i], -1)
        resid -= psflets[i].data*coefs_flat[psflet_indx]
        
    datacube = Image(data=coefs)

    return datacube
