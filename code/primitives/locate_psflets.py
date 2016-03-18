#!/usr/bin/env python

import numpy as np
from scipy import signal, ndimage, optimize
from image import Image
import tools

log = tools.getLogger('main')

def _initcoef(order, scale=15.2, phi=np.arctan2(2,-1), x0=0, y0=0):
    """
    private function _initcoef in locate_psflets

    Create a set of coefficients including a rotation matrix plus zeros.

    Inputs: 
    1. order: int, the polynomial order of the grid distortion

    Optional inputs:
    2. scale: float, the linear separation in pixels of the PSFlets.
              Default 13.88.
    3. phi:   float, the pitch angle of the lenslets.  Default atan(2)
    4. x0:    float, x offset to apply to the central pixel. Default 0
    5. y0:    float, x offset to apply to the central pixel. Default 0

    Returns: 
    1. coef: a list of length (order+1)*(order+2) to be optimized.
    
    The list of coefficients has space for a polynomial fit of the
    input order (i.e., for order 3, up to terms like x**3 and x**2*y,
    but not x**3*y).  It is all zeros in the output apart from the 
    rotation matrix given by scale and phi.    
    """

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except:
            raise ValueError("Polynomial order must be integer")

    n = (order + 1)*(order + 2)
    coef = np.zeros((n))

    coef[0] = x0
    coef[1] = scale*np.cos(phi)
    coef[order + 1] = -scale*np.sin(phi)
    coef[n/2] = y0
    coef[n/2 + 1] = scale*np.sin(phi)
    coef[n/2 + order + 1] = scale*np.cos(phi)
     
    return list(coef)


def _transform(x, y, order, coef):
    """
    private function _transform in locate_psflets

    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Inputs:
    1. x:     ndarray of floats, rectilinear grid
    2. y:     ndarray of floats, rectilinear grid
    3. order: int, order of the polynomial fit
    4. coef:  list of the coefficients.  Must match the length
              required by order = (order+1)*(order+2)
   
    Outputs:
    1. _x:    ndarray, transformed coordinates
    2. _y:    ndarray, transformed coordinates

    """
    
    try:
        if not len(coef) == (order + 1)*(order + 2):
            raise ValueError("Number of coefficients incorrect for polynomial order.")
    except:
        raise AttributeError("order must be integer, coef should be a list.")
    
    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except:
            raise ValueError("Polynomial order must be integer")

    _x = np.zeros(x.shape)
    _y = np.zeros(y.shape)
    
    i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _x += coef[i]*x**ix*y**iy
            i += 1
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _y += coef[i]*x**ix*y**iy
            i += 1
 
    return [_x, _y]


def _corrval(coef, x, y, filtered, order, trimfrac=0.1):

    """
    private function _corrval in locate_psflets

    Return the negative of the sum of the middle XX% of the PSFlet
    spot fluxes (disregarding those with the most and the least flux
    to limit the impact of outliers).  Analogous to the trimmed mean.

    Inputs:
    1. coef:     list, coefficients for polynomial transformation
    2. x:        ndarray, coordinates of lenslets
    3. y:        ndarray, coordinates of lenslets
    4. filtered: ndarray, image convolved with gaussian PSFlet
    5. order:    int, order of the polynomial fit

    Optional inputs: 
    6. trimfrac: float, fraction of outliers (high & low combined) to
                 trim Default 0.1 (5% trimmed on the high end, 5% on
                 the low end)

    Output:
    1. score:    float, negative sum of PSFlet fluxes, to be minimized
    """

    #################################################################
    # Use np.nan for lenslet coordinates outside the CHARIS FOV, 
    # discard these from the calculation before trimming.
    #################################################################

    _x, _y = _transform(x, y, order, coef)
    vals = ndimage.map_coordinates(filtered, [_y, _x], mode='constant', cval=np.nan)
    vals_ok = vals[np.where(np.isfinite(vals))]

    iclip = int(vals_ok.shape[0]*trimfrac/2)
    vals_sorted = np.sort(vals_ok)
    score = -1*np.sum(vals_sorted[iclip:-iclip])
    return score


def locatePSFlets(inImage, polyorder=2, sig=0.7, coef=None, trimfrac=0.1):
    """
    function locatePSFlets takes an Image class, assumed to be a
    monochromatic grid of spots with read noise and shot noise, and
    returns the esimated positions of the spot centroids.  This is
    designed to constrain the domain of the PSF-let fitting later in
    the pipeline.

    Input:
    1. imImage: Image class, assumed to be a monochromatic grid of spots

    Optional Input:
    2. polyorder float, order of the polynomial coordinate transformation
                 Default 2.
    3. sig:      float, standard deviation of convolving Gaussian used
                 for estimating the grid of centroids.  Should be close
                 to the true value for the PSF-let spots.  Default 0.7.
    4. coef:     list, initial guess of the coefficients of polynomial
                 coordinate transformation
    5. trimfrac: float, fraction of lenslet outliers (high & low
                 combined) to trim in the minimization.  Default 0.1
                 (5% trimmed on the high end, 5% on the low end)

    Output:
    1. x:       2D ndarray with the estimated spot centroids in x.
    2. y:       2D ndarray with the estimated spot centroids in y.
    3. good:    2D boolean ndarray, true for lenslets with spots inside
                the detector footprint
    4. coef:    list of best-fit polynomial coefficients

    Notes: the coefficients, if not supplied, are initially set to the 
    known pitch angle and scale.  A loop then does a quick check to find
    reasonable offsets in x and y.  With all of the first-order polynomial
    coefficients set, the optimizer refines these and the higher-order
    coefficients.  This routine seems to be relatively robust down to
    per-lenslet signal-to-noise ratios of order unity (or even a little 
    less).

    Important note: as of now (09/2015), the number of lenslets to grid
    is hard-coded as 1/10 the dimensionality of the final array.  This is
    sufficient to cover the detector for the fiducial lenslet spacing.    
    """

    #############################################################
    # Convolve with a Gaussian, centroid the filtered image.
    #############################################################
    
    x = np.arange(-1*int(3*sig + 1), int(3*sig + 1) + 1)
    x, y = np.meshgrid(x, x)
    gaussian = np.exp(-(x**2 + y**2)/(2*sig**2))

    if inImage.ivar is None:
        filtered = signal.convolve2d(inImage.data, gaussian, mode='same')
    else:
        filtered = signal.convolve2d(inImage.data*inImage.ivar, gaussian, mode='same')
        filtered /= signal.convolve2d(inImage.ivar, gaussian, mode='same')

    #############################################################
    # x, y: Grid of lenslet IDs, Lenslet (0, 0) is the center.
    #############################################################

    gridfrac = 20  
    ydim, xdim = inImage.data.shape
    x = np.arange(-(ydim//gridfrac), ydim//gridfrac + 1)
    x, y = np.meshgrid(x, x)
    
    #############################################################
    # Set up polynomial coefficients, convert from lenslet 
    # coordinates to coordinates on the detector array.  
    # Then optimize the coefficients.
    # We want to start with a decent guess, so we use a grid of 
    # offsets.  Seems to be robust down to SNR/PSFlet ~ 1
    # Create slice indices for subimages to perform the intial
    # fits on. The new dimensionality in both x and y is 2*subsize
    #############################################################

    if coef is None:
        subsize = ydim//4
        imslice = np.s_[ydim/2 - subsize:ydim/2 + subsize,
                        xdim/2 - subsize:xdim/2 + subsize]
        xslice = np.s_[x[0].size/2 - subsize//(gridfrac/2):
                       x[0].size/2 + subsize//(gridfrac/2) + 1,
                       x[0].size/2 - subsize//(gridfrac/2):
                       x[0].size/2 + subsize//(gridfrac/2) + 1]

        log.info("Initializing PSFlet location transformation coefficients")
        bestval = 0
        for ix in np.arange(0, 10, 0.5):
            for iy in np.arange(0, 10, 0.5):
                coef = _initcoef(polyorder, x0=subsize + ix, y0=subsize + iy)
                newval = _corrval(coef, x[xslice], y[xslice], filtered[imslice], polyorder, trimfrac)
                if newval < bestval:
                    bestval = newval
                    coefbest = coef[:]
        coef = coefbest
        
        log.info("Performing initial optimization of PSFlet location transformation coefficients for frame " + inImage.filename)
        res = optimize.minimize(_corrval, coef, args=(x[xslice], y[xslice], filtered[imslice], polyorder, trimfrac), method='Powell')
        coef_opt = res.x

        coef_opt[0] += (ydim/2-subsize)
        coef_opt[(polyorder + 1)*(polyorder + 2)/2] += (xdim/2-subsize)

    #############################################################
    # If we have coefficients from last time, we assume that we
    # are now at a slightly higher wavelength, so try out offsets
    # that are slightly to the right to get a good initial guess.
    #############################################################

    else:
        bestval = 0
        coefsave = coef[:]
        for ix in np.arange(0, 5, 0.5):
            for iy in np.arange(-1, 1.5, 0.5):
                coef = coefsave[:]
                coef[0] += ix
                coef[(polyorder + 1)*(polyorder + 2)/2] += iy
                newval = _corrval(coef, x, y, filtered, polyorder, trimfrac)
                if newval < bestval:
                    bestval = newval
                    coefbest = coef[:]
        coef = coefbest

    log.info("Performing final optimization of PSFlet location transformation coefficients for frame " + inImage.filename)
    res = optimize.minimize(_corrval, coef_opt, args=(x, y, filtered, polyorder, trimfrac), method='Powell')
    coef_opt = res.x

    if not res.success:
        log.info("Optimizing PSFlet location transformation coefficients may have failed for frame " + inImage.filename)
    _x, _y = _transform(x, y, polyorder, coef_opt)

    #############################################################
    # Boolean: do the lenslet PSFlets lie within the detector?
    #############################################################   

    good = (_x > 5)*(_x < xdim - 5)*(_y > 5)*(_y < ydim - 5)

    return [_x, _y, good, coef_opt] 
