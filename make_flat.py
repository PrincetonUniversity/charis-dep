import collections
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Box2DKernel, convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, mad_std

from charis.embed_shell import ipsh


def sigma_filter(img, box=5, nsigma=3, iterate=False, return_mask=False, max_iter=20, _iters=0, _mask=None):
    '''
    Performs sigma-clipping over an image

    Adapted from the IDL function with the same name in the astron library.

    Parameters
    ----------
    img : array
        The input image

    box : int, optional
        Box size for the sigma-clipping. Default is 5 pixel

    nsigma : float, optional
        Sigma value. Default if 3.

    iterate : bool, optional
        Controls if the filtering is iterative. Default is False

    return_mask : bool
        If True, returns a mask to identify the clipped values. Default is False

    max_iter : int, optional
        Maximum number of iterations. Default is 20

    _iters : int (internal)
        Internal counter to keep track during iterative sigma-clipping

    _mask : array_like (internal)
        Keep track of bad pixels over the iterations

    Returns
    -------
    return_value : array
        Input image with clipped values

    '''

    # clip bad pixels
    box2 = box**2

    kernel = Box2DKernel(box)
    img_clip = (convolve(img, kernel)*box2 - img) / (box2-1)

    imdev = (img - img_clip)**2
    fact = nsigma**2 / (box2-2)
    imvar = fact*(convolve(imdev, kernel)*box2 - imdev)

    # following solution is faster but does not support bad pixels
    # see avigan/VLTPF#49
    # img_clip = (ndimage.uniform_filter(img, box, mode='constant')*box2 - img) / (box2-1)

    # imdev = (img - img_clip)**2
    # fact = nsigma**2 / (box2-2)
    # imvar = fact*(ndimage.uniform_filter(imdev, box, mode='constant')*box2 - imdev)

    wok = np.nonzero(imdev < imvar)
    nok = wok[0].size

    # copy good pixels in clipped image
    if (nok > 0):
        img_clip[wok] = img[wok]

    # create _mask at first iteration
    if _mask is None:
        _mask = np.zeros_like(img, dtype=np.bool)

    # identify clipped pixels
    _mask[img != img_clip] = True

    # iterations
    nchange = img.size - nok
    if (iterate is True):
        _iters = _iters+1
        if (_iters >= max_iter) or (nchange == 0):
            if return_mask:
                return img_clip, _mask
            else:
                return img_clip

        return sigma_filter(img_clip, box=box, nsigma=nsigma, iterate=iterate,
                            return_mask=return_mask, _iters=_iters, _mask=_mask)

    if return_mask:
        return img_clip, _mask
    else:
        return img_clip


def fix_badpix(img, bpm, npix=8, weight=False, dmax=10):
    '''Corrects the bad pixels, marked in the bad pixel mask.

    It will fill in bad pixels by finding the NPIX nearest good
    pixels, toss the highest and lowest ones of the bunch, and then
    arithmatically average. Additional it will weight adjacent pixels
    by inverse of their distances in averaging process if the option
    is selected.

    Important warning: to make computation faster, the weighing is not
    applied for bad pixels located within a few pixels from the edges
    of the image.

    Parameters
    ----------
    img : array_like
        Input 2D image

    bpm : array_like, optional
        Input bad pixel map. Good pixels have a value of 0, bad pixels
        a value of 1.

    npix : int, optional
        The number of adjacent good pixels used for the estimation bad
        pixel value. Default value is 8

    weight : bool, optional
        Weigh good pixel by inverse of their distance in the averaging
        process. Default is False

    dmax : int
        Maximum distance from the bad pixel over which good pixels are
        searched for. Default is 10 pixels

    Return
    ------
    img_clean : array_like
        Cleaned image

    '''
    # new arrays
    img = img.copy()
    bpm = (bpm != 0)

    # bad pixels
    bp = np.where(bpm)
    nbp = bp[0].size
    if nbp == 0:
        return img

    # usefull parameters
    ddmin = 2
    ddmax = dmax
    shape = img.shape

    # create default distance array
    dd = ddmin
    xx, yy = np.meshgrid(np.arange(2*dd+1)-dd, np.arange(2*dd+1)-dd)
    dist_default = np.sqrt(xx**2 + yy**2)

    bpm = np.logical_not(bpm)

    for cx, cy in zip(bp[1], bp[0]):
        # default search box is 2*dd+1 pixel
        dd = ddmin

        # determine search region
        found = False
        while not found:
            x0 = max(cx-dd, 0)
            x1 = min(cx+dd+1, shape[-1])
            y0 = max(cy-dd, 0)
            y1 = min(cy+dd+1, shape[-2])

            bpm_sub = bpm[y0:y1, x0:x1]
            img_sub = img[y0:y1, x0:x1]

            if bpm_sub.sum() < npix:
                dd = dd + 2
            else:
                found = True

            if dd > ddmax:
                break

        # distance to adjacent good pixels
        if dd == ddmin:
            # get default array if dd unchanged
            dist = dist_default
        else:
            # otherwise recompute one
            xx, yy = np.meshgrid(np.arange(2*dd+1)-dd, np.arange(2*dd+1)-dd)
            dist = np.sqrt(xx**2 + yy**2)

        # no weighing if we at the edges
        if (bpm_sub.shape != (2*dd+1, 2*dd+1)):
            dist = np.ones_like(bpm_sub)

        # keep good pixels
        good_pix = img_sub[bpm_sub]
        good_dist = dist[bpm_sub]

        # sort them by distance
        ii = np.argsort(good_dist)
        good_pix = good_pix[ii]
        good_dist = good_dist[ii]

        # make sure we have some data to work with
        if len(good_dist) < npix:
            continue

        # get values of relevant pixels
        mm = np.where(good_dist <= good_dist[npix-1])
        good_pix = good_pix[mm]
        good_dist = good_dist[mm]

        ii = np.argsort(good_pix)
        good_pix = good_pix[ii]
        good_dist = good_dist[ii]

        # calculate new pixel value, tossing the highest and lowest
        # pixels of the bunch, then weighting by the inverse of the
        # distances if desired
        if weight:
            final_dist = good_dist[1:-1]
            new_val = np.sum(good_pix[1:-1] / final_dist)
            new_val = new_val / np.sum(1/final_dist)
        else:
            new_val = np.mean(good_pix[1:-1])

        img[cy, cx] = new_val

    return img


def compute_bad_pixel_map(bpm_files, dtype=np.uint8):
    '''
    Compute a combined bad pixel map provided a list of files

    Parameters
    ----------
    bpm_files : list
        List of names for the bpm files

    dtype : data type
        Data type for the final bpm

    logger : logHandler object
        Log handler for the reduction. Default is root logger

    Returns
    -------
    bpm : array_like
        Combined bad pixel map
    '''

    print('> compute master bad pixel map from {} files'.format(len(bpm_files)))

    # get shape
    shape = fits.getdata(bpm_files[0]).shape

    # star with empty bpm
    bpm = np.zeros((shape[-2], shape[-1]), dtype=np.uint8)

    # fill if files are provided
    for f in bpm_files:
        data = fits.getdata(f)
        bpm = np.logical_or(bpm, data)

    bpm = bpm.astype(dtype)

    return bpm


def compute_detector_flat(raw_flat_files, bpm_files=[], mask_vignetting=True):
    '''
    Compute a master detector flat and associated bad pixel map

    Parameters
    ----------
    raw_flat_files : list
        List of 2 raw flat files

    bpm_files : list
        List of bad pixel map files

    mask_vignetting : bool
        Apply a mask on the flats to compensate the optical
        vignetting. The areas of the detector that are vignetted are
        replaced by a value of 1 in the flats. Default is True

    logger : logHandler object
        Log handler for the reduction. Default is root logger

    Returns
    -------
    flat : array
        Master detector flat

    bpm : array

        Bad pixel map from flat

    '''

    # read bad pixel maps
    bpm_in = compute_bad_pixel_map(bpm_files, dtype=np.uint8)
    # IFU mask one for non-vignetted region
    ifu_mask = fits.getdata(
        '/home/masa4294/science/51eri_rawdata/ifs/ifu_mask.fits').astype('bool')

    # read data
    print('> read data')
    ff0, hdr0 = fits.getdata(raw_flat_files[0], header=True)
    ff1, hdr1 = fits.getdata(raw_flat_files[1], header=True)

    # flatten if needed
    if ff0.ndim == 3:
        ff0 = np.median(ff0, axis=0)

    if ff1.ndim == 3:
        ff1 = np.median(ff1, axis=0)

    # create master flat
    print('> create master flat')
    DIT0 = hdr0['HIERARCH ESO DET SEQ1 DIT']
    DIT1 = hdr1['HIERARCH ESO DET SEQ1 DIT']

    if DIT0 > DIT1:
        flat = ff0 - ff1
    else:
        flat = ff1 - ff0
    # ipsh()
    # bad pixels correction
    # print('> bad pixels correction (1/2)')
    # flat = fix_badpix(flat, bpm_in, npix=12, weight=True)

    # flat_clean, flat_bpm1 = sigma_filter(flat, box=5, nsigma=3, iterate=True, return_mask=True)
    # flat_clean, flat_bpm2 = sigma_filter(flat_clean, box=7, nsigma=3, iterate=True, return_mask=True)
    # normalized flat
    print('> normalize')
    mean_value, median_value, _ = sigma_clipped_stats(
        flat[ifu_mask], sigma=3, maxiters=5, cenfunc='median',
        stdfunc=mad_std)
    print("Mean: {} Median: {}".format(mean_value, median_value))
    # flat_clean /= median_value
    flat /= median_value
    # additional round of bad pixels correction
    print('> bad pixels correction (2/2)')
    bpm = (flat <= 0.88) | (flat >= 1.12)
    # ipsh()
    bpm = bpm.astype('int')
    # flat_clean = fix_badpix(flat_clean, bpm, npix=12, weight=True)
    # ipsh()
    # final products
    # print('> compute final flat')
    # mean_value, median_value, _ = sigma_clipped_stats(
    #     flat_clean[ifu_mask], sigma=3, maxiters=5, cenfunc='median',
    #     stdfunc=mad_std)
    # flat_clean /= median_value
    # bpm2 = (flat_clean <= 0.88) | (flat_clean >= 1.12)
    # bpm2 = bpm.astype(np.uint8)
    # apply IFU mask to avoid "edge effects" in the final images,
    # where the the lenslets are vignetted
    if mask_vignetting:
        print('> apply mask vignetting')
        flat[ifu_mask == False] = 1.
        bpm[ifu_mask == False] = 0
    flat[flat <= 0.88] = 0.
    flat[flat >= 1.12] = 0.
    bpm[0:10, :] = False
    bpm[:, 0:10] = False
    bpm[-10:, :] = False
    bpm[:, -10:] = False

    return flat, bpm


def sph_ifs_fix_badpix(img, bpm):
    '''
    Clean the bad pixels in an IFU image

    Extremely effective routine to remove bad pixels in IFS data. It
    goes through all bad pixels and fit a line beween the first good
    pixels encountered along the same column as the bad pixel,
    i.e. along the spectral axis of each micro-spectrum. Works very
    well because as zeroth-order the spectrum is very smooth and can
    be approximated by a line over one (or a few) bad pixels.

    Parameters
    ----------
    img : array_like
        The image to be cleaned

    bpm : array_like
        Bad pixel map

    logger : logHandler object
        Log handler for the reduction. Default is root logger

    Returns
    -------
    img_clean : array_like
        The cleaned image
    '''

    # copy the original image
    # print('> copy input image')
    img_clean = img.copy()

    # extension over which the good pixels will be looked for along
    # the spectral direction starting from the bad pixel
    ext = 10

    # remove edges in bad pixel map
    bpm[:ext+1, :] = 0
    bpm[:, :ext+1] = 0
    bpm[-ext-1:, :] = 0
    bpm[:, -ext-1:] = 0

    # use NaN for identifying bad pixels directly in the image
    img_clean[bpm == 1] = np.nan

    # static indices for searching good pixels and for the linear fit
    idx = np.arange(2*ext+1)
    idx_lh = np.arange(ext)+1

    # loop over bad pixels
    # print('> loop over bad pixels')
    badpix = np.where(bpm == 1)
    for y, x in zip(badpix[0], badpix[1]):
        # extract sub-region along the spectral direction
        sub = img_clean[y-ext:y+ext+1, x]

        # sub-regions "above" and "below" the bad pixel
        sub_low = np.flip(img_clean[y-ext:y, x], axis=0)
        sub_hig = img_clean[y+1:y+1+ext, x]

        # if any of the two is completely bad: skip
        # occurs only in the vignetted areas
        if np.all(np.isnan(sub_low)) or np.all(np.isnan(sub_hig)):
            continue

        # indices of the first good pixels "above" and "below" the bad pixel
        imin_low = idx_lh[~np.isnan(sub_low)].min()
        imin_hig = idx_lh[~np.isnan(sub_hig)].min()

        # linear fit
        xl = idx[ext-imin_low]
        yl = sub[ext-imin_low]

        xh = idx[ext+imin_hig]
        yh = sub[ext+imin_hig]

        a = (yh - yl) / (xh - xl)
        b = yh - a*xh

        fit = a*idx + b

        # replace bad pixel with the fit
        img_clean[y-imin_low+1:y+imin_hig, x] = fit[ext-imin_low+1:ext+imin_hig]

    # put back original value in regions that could not be corrected
    mask = np.isnan(img_clean)
    img_clean[mask] = img[mask]

    return img_clean


def make_cube(files):
    images = []
    for file in files:
        images.append(fits.getdata(file))
    images = np.array(images)
    images = images.reshape(-1, 2048, 2048)
    return images


raw_flat_files = [
    '/home/masa4294/science/51eri_rawdata/ifs/SPHER.2017-09-28T16.55.48.472IFS_DETECTOR_FLAT_FIELD_RAW.fits',
    '/home/masa4294/science/51eri_rawdata/ifs/SPHER.2017-09-28T16.56.27.337IFS_DETECTOR_FLAT_FIELD_RAW.fits'
]

bpm_files = [
    '/home/masa4294/science/51eri_rawdata/ifs/flat_creation/sci_bg/bpm.fits'
]

flat, bpm = compute_detector_flat(
    raw_flat_files=raw_flat_files, bpm_files=bpm_files, mask_vignetting=True)

# fits.writeto('arthur_flat_slim.fits', flat, overwrite=True)
# fits.writeto('arthur_bpm_slim.fits', bpm, overwrite=True)

sky_bg_files = [
    '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/SPHER.2017-11-30T01.17.50.720IFS_DARK_RAW.fits',
    '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/SPHER.2017-11-30T01.23.29.894IFS_DARK_RAW.fits']

sky_bg = make_cube(sky_bg_files)
sky_bg = np.median(sky_bg[3:], axis=0)

fits.writeto('final_sky_bg.fits', sky_bg, overwrite=False)

data_files = [
    '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/SPHER.2017-11-30T01.00.02.332IFS_SCIENCE_DR_RAW.fits',
    '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/SPHER.2017-11-30T01.01.13.992IFS_SCIENCE_DR_RAW.fits',
    '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/SPHER.2017-11-30T01.03.33.497IFS_SCIENCE_DR_RAW.fits',
    '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/SPHER.2017-11-30T01.04.43.377IFS_SCIENCE_DR_RAW.fits'
]

data = make_cube(data_files)
data = np.median(data, axis=0)

# skybg = fits.getdata(
#     '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/background.fits')
# '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/SPHER.2017-11-30T01.23.29.894IFS_DARK_RAW.fits')

# data = fits.getdata(
#     '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/center_minus_medianbg.fits')

# data = np.mean(data, axis=0)
flat = fits.getdata('arthur_flat_slim.fits')
flat_bpm = fits.getdata('arthur_bpm_slim.fits').astype('bool')
dark_bpm = fits.getdata(bpm_files[0]).astype('bool')
ifu_mask = fits.getdata(
    '/home/masa4294/science/51eri_rawdata/ifs/ifu_mask.fits').astype('bool')
ifu_mask2 = fits.getdata(
    '/home/masa4294/science/python/projects/charis-dep/charis/calibrations/SPHERE/background_scaling_mask.fits').astype('bool')
# ifu_mask2 = ~ifu_mask2

flat_bpm[~ifu_mask] = dark_bpm[~ifu_mask]
# Dont mask non-illuminated pixels
flat_bpm[0:10, :] = False
flat_bpm[:, 0:10] = False
flat_bpm[-10:, :] = False
flat_bpm[:, -10:] = False

# specpos_image = fits.getdata('/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW.fits')
# specpos_image = np.median(specpos_image, axis=0)
# specpos_bg = fits.getdata('/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/specpos_bg/specpos_bg.fits')
# specpos = (specpos_image - specpos_bg) / flat
# fits.writeto('/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/bpm_and_flat_test/bgsub_specpos_flat.fits', specpos, overwrite=True)
data_bgsub = data - skybg
data_bgsub_flat = (data - skybg) / flat
data_bgsub_flat[~np.isfinite(data_bgsub_flat)] = 0.


def detect_bad_pixel_in_specpos(image, bpm_in, sigma=2.5, boxsize=5, ifu_mask=None, niter=5):
    if ifu_mask is None:
        ifu_mask = np.ones_like(image).astype('bool')
    if bpm_in is None:
        bpm_in = np.zeros_like(image).astype('bool')

    print("Initial numbers of bad pixels: {}".format(np.sum(bpm_in)))
    bpm = bpm_in.copy()
    n_bad_init = np.sum(bpm_in)
    badpixelmasks = [bpm_in]
    n_bad = [n_bad_init]
    image = sph_ifs_fix_badpix(image, bpm=bpm)

    for iteration in range(niter):
        _, bpm = sigma_filter(
            image, box=boxsize, nsigma=sigma,
            iterate=False, return_mask=True,
            max_iter=20, _iters=0, _mask=None)
        # Deselect bad pixels on the edge of FoV
        print(np.sum(bpm))
        # if iteration > 0:
        badpixelmasks.append(bpm)
        bpm = np.logical_or.reduce(badpixelmasks)
        # not_to_fix = ifu_mask #np.logical_or.reduce([ifu_mask, bpm_so_far])
        # bpm[not_to_fix] = False
        # ext = 10
        # bpm[:ext+1, :] = False
        # bpm[:, :ext+1] = False
        # bpm[-ext-1:, :] = False
        # bpm[:, -ext-1:] = False
        image = sph_ifs_fix_badpix(image, bpm=bpm)
        # n_bad.append(np.sum(
        #     np.logical_and(
        #         badpixelmasks[iteration+1],
        #         ~badpixelmasks[iteration])))
        n_bad.append(np.sum(bpm))

        print("Iter: {} New bad pixels found: {}".format(
            iteration, n_bad[iteration+1] - n_bad[iteration]))
        if n_bad[iteration+1] - n_bad[iteration] == 0:
            print("Converged after {}".format(iteration+1))
            break

    image = sph_ifs_fix_badpix(image, bpm=bpm)
    # mask_of_new = np.logical_or.reduce(badpixelmasks[1:])
    mask_of_new = np.logical_and(bpm, ~bpm_in)
    return image, bpm, mask_of_new, badpixelmasks, n_bad


combined_bpm = np.logical_or(flat_bpm, dark_bpm)
fixed_image, combined_mask, mask_of_new, _, _ = detect_bad_pixel_in_specpos(
    image=data_bgsub_flat, bpm_in=combined_bpm, sigma=2.7,
    boxsize=5, ifu_mask=ifu_mask2, niter=10)

fixed_image_std, combined_mask_std, mask_of_new_std, _, _ = detect_bad_pixel_in_specpos(
    image=data_bgsub_flat, bpm_in=combined_bpm, sigma=3.,
    boxsize=5, ifu_mask=ifu_mask2, niter=10)

fixed_image_orig, combined_mask_orig, mask_of_new_orig, _, _ = detect_bad_pixel_in_specpos(
    image=data, bpm_in=combined_bpm, sigma=2.7,
    boxsize=5, ifu_mask=ifu_mask2, niter=10)

fits.writeto('stable_bpm_2.7sig_v3.fits', (combined_mask).astype('int'), overwrite=True)
fits.writeto('final_bpm_for_pipeline_v3.fits', (~combined_mask).astype('int'), overwrite=True)
fits.writeto('mask_of_new_2.7sig_v3.fits', (mask_of_new).astype('int'), overwrite=True)
fits.writeto('fixed_image_2.7sig_v3.fits', fixed_image, overwrite=True)
fits.writeto('stable_bpm_3sig.fits', (combined_mask_std).astype('int'), overwrite=True)
fits.writeto('mask_of_new_3sig.fits', (mask_of_new_std).astype('int'), overwrite=True)
fits.writeto('fixed_image_3sig.fits', fixed_image_std, overwrite=True)
fits.writeto('original_image.fits', data, overwrite=True)
fits.writeto('basic_red_image.fits', data_bgsub_flat, overwrite=True)
fits.writeto('fixed_image_orig.fits', fixed_image_orig, overwrite=True)
fits.writeto('stable_bpm_orig.fits', (combined_mask_orig).astype('int'), overwrite=True)
fits.writeto('mask_of_new_orig.fits', (mask_of_new_orig).astype('int'), overwrite=True)

# fixed_specpos = sph_ifs_fix_badpix(specpos, flat_bpm)
# _, mask_specpos = sigma_filter(
#     fixed_specpos, box=5, nsigma=2.5, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)

# specpos_fixed2 = sph_ifs_fix_badpix(specpos_fixed, bpm=mask_specpos2)

# data_bgsub_filt, mask_bgsub = sigma_filter(
#     data_bgsub, box=5, nsigma=3, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)
# data_bgsub_flat_filt, mask_bgsub_flat = sigma_filter(
#     data_bgsub_flat, box=5, nsigma=3, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)
#
# masked_data = np.logical_or(mask_bgsub, mask_bgsub_flat)
#
# mask_conservative = np.logical_or(
#     np.logical_and(masked_data, dark_bpm),
#     np.logical_and(masked_data, flat_bpm)
# )
#
# mask_combined = np.logical_or(mask_conservative, flat_bpm)
#
# # Fill in bg based mask in vignetted region
# # mask_combined = flat_bpm
# mask_combined[~ifu_mask] = dark_bpm[~ifu_mask]
#
# # Dont mask non-illuminated pixels
# mask_combined[0:4, :] = False
# mask_combined[:, 0:4] = False
# mask_combined[-4:, :] = False
# mask_combined[:, -4:] = False
#
# mask_combined = ~mask_combined
#
# fits.writeto('flat_based_mask_for_charis.fits', mask_combined.astype('int'), overwrite=True)
#
# # print(np.sum(mask_bgsub))
# print(np.sum(mask_bgsub_flat))
# print(np.sum(bpm.astype('bool')) - np.sum(~ifu_mask))
#
#
# # fits.writeto('mask_sci_minus_sky.fits', mask.astype('int'), overwrite=True)
# # fits.writeto('filtere_sci_minus_sky.fits', data_filtered, overwrite=True)
# # fits.writeto('mask_sci_minus_sky.fits', mask.astype('int'), overwrite=True)
# # fits.writeto('filtere_sci_minus_sky.fits', data_filtered, overwrite=True)
#
#
# # resid = fits.getdata('/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/bpm_and_flat_test/both_bpm_and_flat/SPHER.2017-11-30T01.03.33.497IFS_SCIENCE_DR_RAW_resid.fits')
# #
# # resid_clean, mask_resid = sigma_filter(
# #     resid, box=7, nsigma=5, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)
#
# # specpos_resi = fits.getdata('/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/bpm_and_flat_test/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW_resid.fits')
# specpos_resi = fits.getdata('/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/bpm_and_flat_test/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW_resid.fits')
# # resid_clean, mask_resid_sig5 = sigma_filter(
# #     specpos_resi, box=7, nsigma=5, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)
# #
# # resid_clean, mask_resid_sig4 = sigma_filter(
# #     specpos_resi, box=7, nsigma=4, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)
# #
# # resid_clean, mask_resid_sig3 = sigma_filter(
# #     specpos_resi, box=7, nsigma=3, iterate=True, return_mask=True, max_iter=20, _iters=0, _mask=None)
#
# _, median_value, std_dev_value = sigma_clipped_stats(
#     specpos_resi[ifu_mask], sigma=3, maxiters=5, cenfunc='median',
#     stdfunc=mad_std)
# threshold_negative = median_value - std_dev_value * 3
#
# mask_negative = specpos_resi < threshold_negative
#
# fits.writeto('resid_mask_box7_sig5_flatonly.fits', mask_resid_sig5.astype('int'), overwrite=True)
# fits.writeto('resid_mask_box7_sig4_flatonly.fits', mask_resid_sig4.astype('int'), overwrite=True)
# fits.writeto('resid_mask_box7_sig3_flatonly.fits', mask_resid_sig3.astype('int'), overwrite=True)
#
# mask_combined_with_resid = np.logical_or(~mask_combined, mask_resid_sig4)
# mask_combined_with_resid = ~mask_combined_with_resid
