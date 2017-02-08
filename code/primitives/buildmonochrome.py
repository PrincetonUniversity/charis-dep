#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from scipy import signal, ndimage


def gethires(x, y, image, upsample=5, nsubarr=5, npix=13, renorm=True):
    """
    Build high resolution images of the undersampled PSF using the
    monochromatic frames.

    Inputs:
    1. 
    """

    ###################################################################
    # hires_arr has nsubarr x nsubarr high-resolution PSFlets.  Smooth
    # out the result very slightly to reduce the impact of poorly
    # sampled points.  The resolution on these images, which will be
    # passed to a multidimensional spline interpolator, is a factor of
    # upsample higher than the pixellation of the original image.
    ###################################################################

    hires_arr = np.zeros((nsubarr, nsubarr, upsample*(npix + 1), upsample*(npix + 1)))
    _x = np.arange(3*upsample) - (3*upsample - 1)/2.
    _x, _y = np.meshgrid(_x, _x)
    r2 = _x**2 + _y**2
    window = np.exp(-r2/(2*0.3**2*(upsample/5.)**2))
    
    ###################################################################
    # yreg and xreg denote the regions of the image.  Each region will
    # have roughly 20,000/nsubarr**2 PSFlets from which to construct
    # the resampled version.  For 5x5 (default), this is roughly 800.
    ###################################################################   

    for yreg in range(nsubarr):
        i1 = yreg*image.data.shape[0]//nsubarr
        i2 = i1 + image.data.shape[0]//nsubarr
        i1 = max(i1, npix)
        i2 = min(i2, image.data.shape[0] - npix)

        for xreg in range(nsubarr):
            j1 = xreg*image.data.shape[1]//nsubarr
            j2 = j1 + image.data.shape[1]//nsubarr
            j1 = max(j1, npix)
            j2 = min(j2, image.data.shape[1] - npix)

            ############################################################
            # subim holds the high-resolution images.  The first
            # dimension counts over PSFlet, and must hold roughly the
            # total number of PSFlets divided by upsample**2.  The
            # worst possible case is about 20,000/nsubarr**2.
            ############################################################

            k = 0
            subim = np.zeros((20000/nsubarr**2, upsample*(npix + 1), upsample*(npix + 1)))

            ############################################################
            # Now put the PSFlets in.  The pixel of index
            # [npix*upsample//2, npix*upsample//2] is the centroid.
            # The counter k keeps track of how many PSFlets contribute
            # to each resolution element.
            ############################################################

            for i in range(x.shape[0]):
                if x[i] > j1 and x[i] < j2 and y[i] > i1 and y[i] < i2:
                    xval = x[i] - 0.5/upsample
                    yval = y[i] - 0.5/upsample

                    ix = (1 + int(xval) - xval)*upsample
                    iy = (1 + int(yval) - yval)*upsample

                    if ix == upsample:
                        ix -= upsample
                    if iy == upsample:
                        iy -= upsample

                    iy1, ix1 = [int(yval) - npix//2, int(xval) - npix//2]
                    cutout = image.data[iy1:iy1 + npix + 1, ix1:ix1 + npix + 1]
                    subim[k, iy::upsample, ix::upsample] = cutout
                    k += 1

            meanpsf = np.zeros((upsample*(npix + 1), upsample*(npix + 1)))
            weight = np.zeros((upsample*(npix + 1), upsample*(npix + 1)))

            ############################################################
            # Take the trimmed mean (middle 60% of the data) for each
            # PSFlet to avoid contamination by bad pixels.  Then
            # convolve with a narrow Gaussian to mitigate the effects
            # of poor sampling.
            ############################################################

            for ii in range(3):

                window1 = np.exp(-r2/(2*1**2*(upsample/5.)**2))
                window2 = np.exp(-r2/(2*1**2*(upsample/5.)**2))
                if ii < 2:
                    window = window2
                else:
                    window = window1                    

                if ii > 0:
                    for kk in range(k):
                        mask = 1.*(subim[kk] != 0)
                        if np.sum(mask) > 0:
                            A = np.sum(subim[kk]*meanpsf*mask)
                            A /= np.sum(meanpsf**2*mask)

                            if A > 0.5 and A < 2:
                                subim[kk] /= A
                            else:
                                subim[kk] = 0

                            chisq = np.sum(mask*(meanpsf - subim[kk])**2)
                            chisq /= np.amax(meanpsf)**2

                            subim[kk] *= (chisq < 1e-2*upsample**2)
                            #mask2 = np.abs(meanpsf - subim[kk])/(np.abs(meanpsf) + 0.01*np.amax(meanpsf)) < 1
                            #subim[kk] *= mask2
                            subim[kk] *= subim[kk] > -1e-3*np.amax(meanpsf)

                subim2 = subim.copy()
                for i in range(subim.shape[1]):
                    for j in range(subim.shape[2]):

                        _i1 = max(i - upsample//4, 0)
                        _i2 = min(i + upsample//4 + 1, subim.shape[1] - 1)
                        _j1 = max(j - upsample//4, 0)
                        _j2 = min(j + upsample//4 + 1, subim.shape[2] - 1)
                        
                        data = subim2[:k, _i1:_i2, _j1:_j2][np.where(subim2[:k, _i1:_i2, _j1:_j2] != 0)]
                        if data.shape[0] > 10:
                            data = np.sort(data)[3:-3]
                            std = np.std(data) + 1e-10
                            mean = np.mean(data)
                        
                            subim[:k, i, j] *= np.abs(subim[:k, i, j] - mean)/std < 3.5
                        elif data.shape[0] > 5:
                            data = np.sort(data)[1:-1]
                            std = np.std(data) + 1e-10
                            mean = np.mean(data)
                        
                            subim[:k, i, j] *= np.abs(subim[:k, i, j] - mean)/std < 3.5
                        
                        data = subim[:k, i, j][np.where(subim[:k, i, j] != 0)]
                        #data = np.sort(data)
                        npts = data.shape[0]
                        if npts > 0:
                            meanpsf[i, j] = np.mean(data)
                            weight[i, j] = npts

                meanpsf = signal.convolve2d(meanpsf*weight, window, mode='same')
                meanpsf /= signal.convolve2d(weight, window, mode='same')

                val = meanpsf.copy()
                for jj in range(10):
                    tmp = val/signal.convolve2d(meanpsf, window, mode='same')
                    meanpsf *= signal.convolve2d(tmp, window[::-1, ::-1], mode='same')
                    
            
            ############################################################
            # Normalize all PSFs to unit flux when resampled with an
            # interpolator.
            ############################################################

            if renorm:
                meanpsf *= upsample**2/np.sum(meanpsf)
            hires_arr[yreg, xreg] = meanpsf
            
    return hires_arr


def make_polychrome(lam1, lam2, hires_arrs, lam_arr, psftool, allcoef,
                     xindx, yindx, upsample=5, nlam=10, trans=None):
    """
    """

    padding = 10
    image = np.zeros((2048 + 2*padding, 2048 + 2*padding))
    x = np.arange(image.shape[0])
    x, y = np.meshgrid(x, x)
    npix = hires_arrs[0].shape[2]//upsample

    dloglam = (np.log(lam2) - np.log(lam1))/nlam
    loglam = np.log(lam1) + dloglam/2. + np.arange(nlam)*dloglam

    for lam in np.exp(loglam):

        if trans is not None:
            indx = np.where(np.abs(np.log(trans[:, 0]/lam)) < dloglam/2.)
            meantrans = np.mean(trans[:, 1][indx])

        ################################################################
        # Build the appropriate average hires image by averaging over
        # the nearest wavelengths.  Then apply a spline filter to the
        # interpolated high resolution PSFlet images to avoid having
        # to do this later, saving a factor of a few in time.
        ################################################################

        hires = np.zeros((hires_arrs[0].shape))
        if lam <= np.amin(lam_arr):
            hires[:] = hires_arrs[0]
        elif lam >= np.amax(lam_arr):
            hires[:] = hires_arrs[-1]
        else:
            i1 = np.amax(np.arange(len(lam_arr))[np.where(lam > lam_arr)])
            i2 = i1 + 1
            hires = hires_arrs[i1]*(lam - lam_arr[i1])/(lam_arr[i2] - lam_arr[i1])
            hires += hires_arrs[i2]*(lam_arr[i2] - lam)/(lam_arr[i2] - lam_arr[i1])

        for i in range(hires.shape[0]):
            for j in range(hires.shape[1]):
                hires[i, j] = ndimage.spline_filter(hires[i, j])

        ################################################################
        # Run through lenslet centroids at this wavelength using the
        # fitted coefficients in psftool to get the centroids.  For
        # each centroid, compute the weights for the four nearest
        # regions on which the high-resolution PSFlets have been made.
        # Interpolate the high-resolution PSFlets and take their
        # weighted average, adding this to the image in the
        # appropriate place.
        ################################################################

        xcen, ycen = psftool.return_locations(lam, allcoef, xindx, yindx)
        xcen += padding
        ycen += padding
        xcen = np.reshape(xcen, -1)
        ycen = np.reshape(ycen, -1)
        for i in range(xcen.shape[0]):
            if not (xcen[i] > npix//2 and xcen[i] < image.shape[0] - npix//2 and 
                    ycen[i] > npix//2 and ycen[i] < image.shape[0] - npix//2):
                continue
                
            # central pixel -> npix*upsample//2
            iy1 = int(ycen[i]) - npix//2
            iy2 = iy1 + npix
            ix1 = int(xcen[i]) - npix//2
            ix2 = ix1 + npix
            yinterp = (y[iy1:iy2, ix1:ix2] - ycen[i])*upsample + upsample*npix/2
            xinterp = (x[iy1:iy2, ix1:ix2] - xcen[i])*upsample + upsample*npix/2
            # Now find the closest high-resolution PSFs
            
            x_hires = xcen[i]*1./image.shape[1]
            y_hires = ycen[i]*1./image.shape[0]
            
            x_hires = x_hires*hires_arrs[0].shape[1] - 0.5
            y_hires = y_hires*hires_arrs[0].shape[0] - 0.5
            
            totweight = 0
            
            if x_hires <= 0:
                i1 = i2 = 0
            elif x_hires >= hires_arrs[0].shape[1] - 1:
                i1 = i2 = hires_arrs[0].shape[1] - 1
            else:
                i1 = int(x_hires)
                i2 = i1 + 1

            if y_hires < 0:
                j1 = j2 = 0
            elif y_hires >= hires_arrs[0].shape[0] - 1:
                j1 = j2 = hires_arrs[0].shape[0] - 1
            else:
                j1 = int(y_hires)
                j2 = j1 + 1
            
            ##############################################################
            # Bilinear interpolation by hand.  Do not extrapolate, but
            # instead use the nearest PSFlet near the edge of the
            # image.  The outer regions will therefore have slightly
            # less reliable PSFlet reconstructions.  Then take the
            # weighted average of the interpolated PSFlets.
            ##############################################################

            weight22 = max(0, (x_hires - i1)*(y_hires - j1))
            weight12 = max(0, (x_hires - i1)*(j2 - y_hires))
            weight21 = max(0, (i2 - x_hires)*(y_hires - j1))
            weight11 = max(0, (i2 - x_hires)*(j2 - y_hires))
            totweight = weight11 + weight21 + weight12 + weight22
            weight11 /= totweight*nlam
            weight12 /= totweight*nlam
            weight21 /= totweight*nlam
            weight22 /= totweight*nlam

            if trans is not None:
                weight11 *= meantrans
                weight12 *= meantrans
                weight21 *= meantrans
                weight22 *= meantrans

            image[iy1:iy2, ix1:ix2] += weight11*ndimage.map_coordinates(hires[j1, i1], [yinterp, xinterp], prefilter=False)
            image[iy1:iy2, ix1:ix2] += weight12*ndimage.map_coordinates(hires[j1, i2], [yinterp, xinterp], prefilter=False)
            image[iy1:iy2, ix1:ix2] += weight21*ndimage.map_coordinates(hires[j2, i1], [yinterp, xinterp], prefilter=False)
            image[iy1:iy2, ix1:ix2] += weight22*ndimage.map_coordinates(hires[j2, i2], [yinterp, xinterp], prefilter=False)
     
    image = image[padding:-padding, padding:-padding]
    return image

