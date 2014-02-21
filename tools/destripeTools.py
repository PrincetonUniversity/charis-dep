# Author: Tim Brandt
# Email: tbrandt@astro.princeton.edu
# Date: January 2011
# 
# Summary:  Compute the best-fit offset between two stripes, subtract 
# from the first stripe supplied.  
# 

import numpy as np
import sys

def horizontal(flux, stripe, minx=0, xdist=4):

    """    
    Function horizontal takes three arguments:
    1.  A 2048 x 2048 array of flux values
    2.  The stripe to correct (0 - 31)

    Optional arguments:
    3.  x pixel from which to start counting (default = 0)
    4.  Width of x pixel range to be used (default = 4)

    horizontal finds the best-fit difference (bias) between two stripes.  
    It then subtracts that bias from the first stripe supplied.
    
    NOTE: This code is an almost identical copy of that in the destripe_utils 
    directory of the ACORNS-ADI pipeline, written by Timothy D. Brandt and 
    described in Brandt 2013.
    """
    
    dimy, dimx = flux.shape
    row = stripe * 64

    ##################################################################
    # Even/odd rows are read in opposite directions
    ##################################################################

    diffdist_l = flux[row:row + 64, minx:minx + xdist]
    diffdist_r = flux[row:row + 64, dimx - xdist - minx:dimx - minx]

    diffdist = np.hstack((diffdist_l, diffdist_r))

    ##################################################################
    # Gaussian data ==> use mean.  Exclude > nsig sigma outliers as bad
    # pixels.  Loop over the data niter times to get a good guess.
    ##################################################################
    
    mu = 0
    nsig = 3
    niter = 4
    sig = 1e4
    for i in range(niter):
        pts = np.extract(np.abs(diffdist - mu) < nsig * sig, diffdist)
        mu = np.mean(pts)
        sig = np.std(pts)
    
    for i in range(row, row + 64):
        flux[i] -= mu
    
    return 

def interpbadpix(flux, n=5):

    """
    Function interpbadpix takes one argument:
    1.  A 2048 x 2048 array of flux values, with bad pixels flagged by NaN

    Optional argument:
    2.  Box size from which to select value (default 5, for a total 
        of 25 pixels)

    interpbadpix 'cleans' bad pixels using the median of surrounding
    good pixels.  It uses an n x n box (default 5x5) to select good
    pixels.  Bad pixels must be flagged by NaN.
    
    NOTE: This code is an almost identical copy of that in the destripe_utils 
    directory of the ACORNS-ADI pipeline, written by Timothy D. Brandt and 
    described in Brandt 2013.
    """
    
    dimy, dimx = flux.shape
    flux = np.reshape(flux, -1)

    ##################################################################
    # Map the bad pixels, create and populate an array to store
    # their neighbors
    ##################################################################

    badpix = np.logical_not(np.isfinite(flux))

    fluxmed = np.ndarray((n**2 - 1, np.sum(badpix)), np.float32)
    indx = np.extract(badpix, np.arange(flux.shape[0]))
    
    for i in range(0, n):
        if i <= n // 2 and i > 0:
            di = -i * dimy
        else:
            di = (i - n // 2) * dimy

        if i > 0:
            fluxmed[i - 1] = flux[indx + di]
            
        for j in range(1, n):
            if j <= n // 2:
                dj = -j
            else:
                dj = j - n // 2
            fluxmed[(i + 1) * (n - 1) + j - 1] = flux[indx + di + dj]

    ##################################################################
    # This is significantly faster than scipy.stats.nanmedian (I don't
    # know how that works).  Sort arrays, NaN are at the end.  Then
    # count the non-NaN values for each pixel and take their median.
    ##################################################################

    fluxmed = np.sort(fluxmed, axis=0)
    imax = np.sum(np.logical_not(np.isnan(fluxmed)), axis=0) 

    for i in range(1, np.amax(imax) + 1):
        indxnew = np.where(imax == i)
        if len(indxnew) > 0:
            flux[indx[indxnew]] = np.median(fluxmed[:i, indxnew], axis=0)

    flux = np.reshape(flux, (dimy, dimx))
    
    return

def verticalmed(flux, flat, r_ex=0):

    """
    Function verticalmed takes two arguments:
    1.  A 2048 x 2048 array of flux values
    2.  A 2048 x 2048 flat-field array

    Optional argument:
    3.  Exclusion radius for calculting the median of the horizontal stripes
          (default zero, recommended values from 0 to 800)
          See Kandori-san's IDL routine for the equivalent.
        
    verticalmed takes the median of the horizontal stripes to calculate a
    vertical template, as in Kandori-san's routine.  The routine ignores a
    circular region about the array center if r_ex > 0, and also ignores
    all pixels flagged with NaN.
    
    NOTE: This code is an almost identical copy of that in the destripe_utils 
    directory of the ACORNS-ADI pipeline, written by Timothy D. Brandt and 
    described in Brandt 2013.
    """

    ###############################################################
    # Construct radius array, copy flux to mask
    ###############################################################

    dimy, dimx = flux.shape
    x = np.linspace(-dimx, dimx, dimx) / 2
    y = np.linspace(-dimy, dimy, dimy) / 2
    x, y = np.meshgrid(x, y)
    r2 = x**2 + y**2

    if r_ex > 0:
        flux2 = np.ndarray(flux.shape, np.float32)
        flux2[:] = flux
        np.putmask(flux2, r2 < r_ex**2, np.nan)
    else:
        flux2 = flux

    ###############################################################
    # Estimate background level
    ###############################################################

    backgnd = np.ndarray(flux2.shape)
    backgnd[:] = flux2 / flat
    backgnd = np.sort(np.reshape(backgnd, -1))
    ngood = np.sum(np.isfinite(backgnd))
    level = np.median(backgnd[:ngood])
    flux2 -= level * flat

    ###############################################################
    # Sort the flux values.  NaN values will be at the end for
    # numpy versions >= 1.4.0; otherwise this routine may fail
    ###############################################################
    
    tmp = np.ndarray((32, dimy // 32, dimx), np.float32)
    for i in range(1, 33, 2):
        tmp[i] = flux2[64 * i:64 * i + 64]
        tmp[i - 1] = flux2[64 * i - 64:64 * i]
        
    tmp = np.sort(tmp, axis=0)
    oldshape = tmp[0].shape
    tmp = np.reshape(tmp, (tmp.shape[0], -1))
    
    oddstripe = np.zeros(tmp[0].shape, np.float32)

    ###############################################################
    # imax = number of valid (!= NaN) references for each pixel.
    # Calculate the median using the correct number of elements,
    # doing it only once for each pixel.
    ###############################################################
    
    imax = np.sum(np.logical_not(np.isnan(tmp)), axis=0)
    for i in range(np.amin(imax), np.amax(imax) + 1):
        indxnew = np.where(imax == i)
        if len(indxnew) > 0:
            oddstripe[indxnew] = np.median(tmp[:i, indxnew], axis=0)

    ###############################################################
    # Set the median of the pattern to be subtracted equal to the 
    # median of the difference between the science and reference
    # pixels.
    ###############################################################

    oddstripe -= np.median(oddstripe)
    oddstripe += 0.5 * (np.median(flux[:4]) + np.median(flux[-4:]))
    
    oddstripe = np.reshape(oddstripe, oldshape)
    evenstripe = oddstripe[::-1]

    return [oddstripe, evenstripe]

def verticalref(flux, refstripe=None, smoothwidth=300, bias_only=True):

    """
    Function verticalref takes two arguments:
    1.  A 2048 x 2048 array of flux values

    Optional arguments:
    2.  The stripe to smooth, constructing a reference stripe.  Ignored if
        bias_only=True, required otherwise.
    3.  The width of the Gaussian filter (default 300) -- see paper
    4.  Use only the reference pixels at the top and bottom of the detector
        (default True)

    verticalref convolves the input stripe with a Gaussian to suppress
    high frequency pickup.  It returns the smoothed stripe.
    
    NOTE: This code is an almost identical copy of that in the destripe_utils 
    directory of the ACORNS-ADI pipeline, written by Timothy D. Brandt and 
    described in Brandt 2013.
    """
    
    nsig = 5
    returntime = 3
    dimy, dimx = flux.shape
    if not bias_only:
        refrow = refstripe * 64

    ##################################################################
    # Extract the stripe as a 1-D array in the order it was read out.  
    # Take care to leave extra blank pixels for the start of a new row.
    ##################################################################

    pixels = np.ndarray((64 + returntime, dimx), np.float32)

    if refstripe % 2 == 0 and not bias_only:
        pixels[:] = flux[refrow:refrow + 64 + returntime]
    elif not bias_only:
        pixels[:] = flux[refrow + 63:refrow - 1 - returntime:-1]
    else:
        pixels[:] = np.nan
        pixels[:4] = 0.5 * (flux[:4] + flux[dimy - 1:dimy - 5:-1])
    
    pixels[64:64 + returntime, 4:dimx - 4] = np.nan

    ##################################################################
    # Mask deviant pixels, blank ('nan') pixels
    ##################################################################

    if bias_only:
        imax = 4
    else:
        imax = pixels.shape[0]

    ##################################################################
    # Sigma-reject to get the useful pixels.
    ##################################################################

    mu = 0
    sig = 1e4
    niter = 4

    for i in range(niter):
        pts = np.extract(np.abs(pixels[:imax] - mu) < nsig * sig, 
                         pixels[:imax])
        mu = np.mean(pts)
        sig = np.std(pts)
            
    pixmask = np.zeros(pixels.shape, np.float32)
    np.putmask(pixmask[:imax], np.abs(pixels[:imax] - mu) < nsig * sig, 1)
    np.putmask(pixels[:imax], pixmask[:imax] < 1, 0)

    pixelrow = np.reshape(pixels, -1, order='F')
    pixmask = np.reshape(pixmask, -1, order='F')
            
    ##################################################################
    # Convolve the pixel mask with a normalized Gaussian
    ##################################################################

    window = 3 * smoothwidth - np.arange(6 * smoothwidth + 1)
    window = np.exp(-(window * 1.0)**2 / (2 * smoothwidth**2))
        
    if bias_only:
        pixsmooth = np.zeros(pixelrow.shape, np.float32)
        pixnorm = np.zeros(pixelrow.shape, np.float32)
        for i in range(dimx):
            j = i * (64 + returntime)
            dj = window.shape[0] // 2 + 2
            i1 = max(0, j - dj)
            i2 = min(pixelrow.shape[0], j + dj)
            j1 = max(0, dj - j)
            j2 = window.shape[0] + 3 - max(0, j + dj - pixelrow.shape[0])
            
            pixsmooth[i1:i2] += np.convolve(pixelrow[j:j + 4], window)[j1:j2]
            pixnorm[i1:i2] += np.convolve(pixmask[j:j + 4], window)[j1:j2]
    else:
        pixsmooth = np.convolve(pixelrow, window, 'same')
        pixnorm = np.convolve(pixmask, window, 'same')
        
    pixsmooth /= pixnorm
    
    ##################################################################
    # Reshape and return the smoothed array
    ##################################################################

    pixsmooth = np.reshape(pixsmooth, (64 + returntime, -1), order = 'F')
    if refstripe % 2 == 1 or refstripe is None and bias_only:
        vstripe = pixsmooth[63::-1]
    else:
        vstripe = pixsmooth[0:64]
        
    return vstripe
    

