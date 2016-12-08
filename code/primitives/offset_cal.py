#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from scipy import linalg, signal, ndimage
import matutils
import multiprocessing


def calc_offset(psflets, image, offsets, dx=64, 
                maxcpus=multiprocessing.cpu_count()):
    """
    Function calc_offset: compute the position-dependent sub-pixel
    shift in the PSFlet spot grid using cross-correlation.

    Inputs:
    1. psflets:  ndarray, (nlam, ny, Nx), with Nx = upsamp*nx,
                 i.e. oversampled by a factor upsamp in the last 
                 dimension (perpendicular to the CHARIS dispersion
                 direction).
    2. image:    ndarray, (ny, nx)
    3. offsets:  ndarray, integer offsets at which to compute 
                 cross-correlation, shifting psflets by upsamp with
                 respect to image.

    Optional inputs:
    1. dx:       integer, (dx,dx) size of chunks on which to compute
                 cross-correlation.  Default 64 (so that 32x32
                 offsets are computed for a 2048x2048 image).
    2. maxcpus:  number of threads for OpenMP parallelization of 
                 cython code.  Default multiprocessing.cpu_count().

    Output:
    1. psflets:  ndarray, (nlam, ny, nx), PSFlet templates shifted 
                 by a position-dependent value to maximize the 
                 cross-correlation with the input image.

    """

    mask = (image.ivar > 0).astype(np.uint16)

    if psflets.dtype != 'float32':  # Ensure correct dtype, native byte order
        psflets2 = np.empty(psflets.shape, np.float32)
        psflets2[:] = psflets
        psflets = psflets2
        
    #####################################################################
    # Calculate the cross-correlation of the PSFlet images and the
    # actual data at the specified offsets.  Do this in dim/dx
    # subregions of the image to get a position-dependent shift.
    #####################################################################

    ny, nx = image.data.shape
    shiftarr = np.zeros((int(np.ceil(ny*1./dx)), int(np.ceil(nx*1./dx))))

    outarr = np.zeros((psflets.shape[0], ny, nx))

    for i in range(0, nx, dx):

        corrvals_all = matutils.crosscorr(psflets, image.data, image.ivar,
                                          offsets, maxproc=maxcpus,
                                          m1=i, m2=i+dx)

        for j in range(0, ny, dx):
            
            corrvals = np.sum(corrvals_all[:, j:j+dx], axis=1)

            #############################################################
            # Calculate offset to maximize the cross-correlation by
            # fitting a parabola to the five points bracketing the
            # best value
            #############################################################

            icen = np.arange(offsets.shape[0])[np.where(corrvals == np.amax(corrvals))]
            imin = int(max(0, icen - 2))
            imax = int(min(offsets.shape[0], icen + 3))
            corrvals = corrvals[imin:imax]
            
            arr = np.ones((imax - imin, 3))
            arr[:, 1] = offsets[imin:imax]
            arr[:, 2] = offsets[imin:imax]**2
            coef = linalg.lstsq(arr, corrvals)[0]
            
            shift = -coef[1]/(2*coef[2])
            shiftarr[j//dx, i//dx] = shift

    #####################################################################
    # Return the interpolated array at the requested offset
    #####################################################################
    
    if shiftarr.shape[0] >= 3:
        shiftarr[1:-1, 1:-1] = signal.medfilt2d(shiftarr, 3)[1:-1, 1:-1]
        shiftarr[0, 1:-1] = signal.medfilt(shiftarr[0], 3)[1:-1]
        shiftarr[-1, 1:-1] = signal.medfilt(shiftarr[-1], 3)[1:-1]
        shiftarr[1:-1, 0] = signal.medfilt(shiftarr[:, 0], 3)[1:-1]
        shiftarr[1:-1, -1] = signal.medfilt(shiftarr[:, -1], 3)[1:-1]

    x = (1.*np.arange(ny))/dx - 0.5
    x *= x > 0
    x[np.where(x > shiftarr.shape[0] - 1)] = shiftarr.shape[0] - 1
    x, y = np.meshgrid(x, x)

    fullshiftarr = ndimage.map_coordinates(shiftarr, [y, x], order=3)

    psflets = matutils.interpcal(psflets, image.data, mask, fullshiftarr, 
                                 maxproc=maxcpus)
    return psflets

