import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from charis.embed_shell import ipsh

from pdb import set_trace


def make_lenslet_flat(file, mask=None, save=False):
    data = fits.getdata(file)

    basename = os.path.splitext(os.path.basename(file))[0]
    dirname = os.path.dirname(file)

    if mask is None:
        mask = np.zeros((data.shape[-2], data.shape[-1]), dtype='bool')
        x1, x2 = 60, 150
        y1, y2 = 45, 150
        mask[y1:y2, x1:x2] = True

    # median_values_old = np.median(data[:, mask], axis=1)
    _, median_values, _ = sigma_clipped_stats(
        data[:, mask], sigma=3, maxiters=5, axis=1)

    normalized_data = data / median_values[:, None, None]
    collapsed_data = np.median(
        np.sort(normalized_data[3:-3], axis=0)[2:-2], axis=0)

    mask_small = np.logical_and(collapsed_data < 0.8, collapsed_data != 0)
    mask_large = collapsed_data > 1.08
    mask = np.logical_or(mask_small, mask_large)
    # np.sum(mask) == 839 lenslets are bad

    mask_hard = np.zeros_like(collapsed_data).astype('bool')
    mask_hard[0:27] = True
    mask_hard[172:] = True
    mask_hard[:, 174:] = True
    mask_hard[:, :32] = True
    # mask_combined = np.logical_or(mask_hard, mask)
    mask_combined = mask_hard
    # ipsh()
    collapsed_data[mask_combined] = 1.
    good_lenslets = np.logical_and(
        collapsed_data > 0, collapsed_data != 1.)
    norm_factor = np.median(collapsed_data[collapsed_data > 0])
    _, norm_factor_clip, _ = sigma_clipped_stats(
        collapsed_data[good_lenslets], sigma=3, maxiters=5)
    print("Final norm. factor: {}".format(norm_factor))
    print("Final norm. factor sigma clipped: {}".format(norm_factor_clip))

    collapsed_data /= norm_factor

    # plt.imshow(mask_combined, origin='lower')
    # plt.show()
    # plt.imshow(collapsed_data, origin='lower')
    # plt.show()

    outputdir = os.path.join(dirname, 'calibrated_lensletflat')

    if save is True:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fits.writeto(
            os.path.join(outputdir, basename + '_median_lensletvalues.fits'),
            median_values, overwrite=True)
        fits.writeto(
            os.path.join(outputdir, basename + '_normalized_lenslet_flat_cube.fits'),
            normalized_data, overwrite=True)
        fits.writeto(
            os.path.join(outputdir, basename + '_lenslet_flat.fits'),
            collapsed_data, overwrite=True)

    return median_values, normalized_data, collapsed_data


# wo_corr_filename = '/home/masa4294/science/51eri_rawdata/ifs/oversampled/tests/lenslet_flat_wo_xtalkcorr.fits'
# wo_corr_filename = '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW_cube.fits'
# filename = '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/caldata/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW_cube.fits'
# filename = '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/bpm_and_flat_test/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW_cube.fits'
filename = '/home/masa4294/science/sphere_calibration/test_data/HD2133_wd/scidata/bpm_and_flat_test/SPHER.2017-11-30T16.39.05.342IFS_SPECPOS_RAW_cube.fits'
# w_corr_filename = '/home/masa4294/science/51eri_rawdata/ifs/oversampled/tests/lenslet_flat_w_xtalkcorr.fits'

median_values, normalized_data, collapsed_data = make_lenslet_flat(
    filename, mask=None, save=False)

good_lenslets = np.logical_and(
    collapsed_data > 0., collapsed_data != 1.)
plt.imshow(good_lenslets, origin='bottom', interpolation='nearest')

median_values, normalized_data, collapsed_data = make_lenslet_flat(
    filename, mask=good_lenslets, save=False)


# w_median_values, w_normalized_data, w_collapsed_data = make_lenslet_flat(
#     w_corr_filename, save=False)
# set_trace()

plt.imshow(collapsed_data, origin='lower', interpolation='nearest')
plt.show()

# fits.writeto('51eri_lensletflat.fits', wo_normalized_data)
fits.writeto('wd_lensletflat_collapsed_newbpm_v2.fits', collapsed_data, overwrite=True)
fits.writeto('wd_lensletflat_normalized_newbpm_v2.fits', normalized_data, overwrite=True)


plt.plot(median_values / np.sum(median_values), label='')
plt.legend()
plt.show()
