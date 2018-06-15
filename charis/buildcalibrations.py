#!/usr/bin/env python

from __future__ import print_function, absolute_import

from builtins import input
from builtins import str
from builtins import range
import copy
import glob
import logging
import multiprocessing
import os
import pdb
import re
import shutil
import sys
import time

import numpy as np
import pkg_resources
from astropy import units as u
from astropy.io import fits
from scipy import interpolate, ndimage

try:
    import instruments
    import primitives
    import utr
    from image import Image
    from parallel import Task, Consumer
except ImportError:
    from charis import instruments
    from charis import primitives
    from charis import utr
    from charis.image import Image
    from charis.parallel import Task, Consumer

log = logging.getLogger('main')


def buildcalibrations(inImage, inLam, mask, indir, instrument, outdir="./",
                      order=3, upsample=True, header=None,
                      ncpus=multiprocessing.cpu_count(),
                      nlam=10):
    """
    Build the calibration files needed to extract data cubes from
    sequences of CHARIS reads.

    Inputs:
    1. inImage:  Image object, should include count rate and ivar for
                 a narrow-band flatfield calibration image.
    2. inLam:    wavelength in nm of inImage
    3. mask:     bad pixel mask, =0 for bad pixels
    4. indir:    directory where master calibration files live
    5. instrument: instrument object

    Optional inputs:
    outdir:   directory in which to place
    order:    int, order of polynomial fit to position(lambda).
                 Default 3 (strongly recommended).
    header:   FITS header, to which will be appended the shifts
                 and rotation angle between the stored and the fitted
                 wavelength solutions.  Default None.
    ncpus:    number of threads for multithreading.
                 Default multiprocessing.cpu_count()

    nlam: int, number of monochromatic PSFlets per integrated PSFlet

    Returns None, writes calibration files to outdir.

    """

    tstart = time.time()
    lower_wavelength_limit, upper_wavelength_limit = instrument.wavelength_range.value
    R = instrument.resolution
    npix_y, npix_x = inImage.data.shape

    #################################################################
    # Fit the PSFlet positions on the input image, compute the shift
    # in the mean position (coefficients 0 and 10) and in the linear
    # component of the fit (coefficients 1, 4, 11, and 14).  The
    # comparison point is the location solution for this wavelength in
    # the existing calibration files.
    #################################################################

    log.info("Loading wavelength solution from " + indir + "/lamsol.dat")
    lam = np.loadtxt(os.path.join(indir, "lamsol.dat"))[:, 0]
    allcoef = np.loadtxt(os.path.join(indir, "lamsol.dat"))[:, 1:]
    psftool = primitives.PSFLets()
    oldcoef = []
    for cal_lam in inLam:
        oldcoef += [psftool.monochrome_coef(cal_lam, lam, allcoef, order=order).tolist()]
    print('Generating new wavelength solution')
    _, _, _, newcoef = primitives.locatePSFlets(inImage, instrument, polyorder=3, coef=oldcoef, fitorder=1)

    psftool.geninterparray(lam, allcoef, order=order)
    dcoef = np.asarray(newcoef[0]) - oldcoef[0]

    psftool.interp_arr[0] += dcoef
    psftool.genpixsol(lam, allcoef, instrument, order=order, lam1=lower_wavelength_limit / 1.05,
                      lam2=upper_wavelength_limit * 1.05)
    psftool.savepixsol(outdir=outdir)

    #################################################################
    # Record the shift in the spot locations.  Pull the linear orders
    # first to ensure consistency in the indices.
    #################################################################

    oldlin = primitives.pullorder(oldcoef[0])
    newlin = primitives.pullorder(newcoef[0])
    phi1 = np.mean([np.arctan2(oldlin[2], oldlin[1]),
                    np.arctan2(-oldlin[4], oldlin[5])])
    phi2 = np.mean([np.arctan2(newlin[2], newlin[1]),
                    np.arctan2(-newlin[4], newlin[5])])
    dx, dy, dphi = [newlin[0] - oldlin[0], newlin[3] - oldlin[3], phi2 - phi1]
    print(dx, dy, dphi)
    if header is not None:
        header['cal_dx'] = (dx, 'x-shift from archival spot positions (pixels)')
        header['cal_dy'] = (dy, 'y-shift from archival spot positions (pixels)')
        header['cal_dphi'] = (dphi, 'Rotation from archival spot positions (radians)')

    #################################################################
    # Load the high-resolution PSFlet images and associated
    # wavelengths.
    #################################################################

    hires_list = np.sort(glob.glob(os.path.join(indir, 'hires_psflets_lam*.fits')))
    hires_arrs = [fits.open(filename)[0].data for filename in hires_list]
    lam_hires = [float(re.sub('.*lam', '', re.sub('.fits', '', filename)))
                 for filename in hires_list]
    psflet_res = 9  # Oversampling of high-resolution PSFlet images

    #################################################################
    # Width of high-resolution PSFlets, in pixels.  First compute the
    # width from the images perpendicular to the dispersion direction
    # at the central pixel along the dispersion direction.
    #################################################################

    shape = hires_arrs[0].shape
    sigarr = np.zeros((len(hires_list), shape[0], shape[1]))
    _x = np.arange(shape[3]) / 9.
    _x -= _x[_x.shape[0] // 2]

    for i in range(sigarr.shape[0]):
        for j in range(sigarr.shape[1]):
            for k in range(sigarr.shape[2]):
                row = hires_arrs[i][j, k, shape[2] // 2]
                sigarr[i, j, k] = np.sum(row * _x**2)
                sigarr[i, j, k] /= np.sum(row)

        sigarr[i] = np.sqrt(sigarr[i])

    #################################################################
    # Now interpolate the width at the locations and wavelengths of
    # the microspectra for optimal extraction.  First interpolate in
    # location, then interpolate in wavelength for each lenslet.
    #################################################################

    mean_x = psftool.lenslet_ix[:, :, psftool.lenslet_ix.shape[-1] // 2]
    mean_y = psftool.lenslet_iy[:, :, psftool.lenslet_iy.shape[-1] // 2]

    longsigarr = np.zeros((len(lam_hires), mean_x.shape[0], mean_x.shape[1]))

    ix = mean_x * hires_arrs[0].shape[1] / npix_x - 0.5
    iy = mean_y * hires_arrs[0].shape[0] / npix_y - 0.5

    for i in range(sigarr.shape[0]):
        longsigarr[i] = ndimage.map_coordinates(sigarr[i], [iy, ix], order=3, mode='nearest')
    fullsigarr = np.zeros((psftool.lenslet_ix.shape))
    for i in range(mean_x.shape[0]):
        for j in range(mean_x.shape[1]):
            fit = interpolate.interp1d(np.asarray(lam_hires), longsigarr[:, i, j],
                                       bounds_error=False, fill_value='extrapolate')
            fullsigarr[i, j] = fit(psftool.lam_indx[i, j])
    out = fits.HDUList(fits.PrimaryHDU(fullsigarr.astype(np.float32)))
    out.writeto('PSFwidths.fits', overwrite=True)

    #################################################################
    # Wavelengths at which to return the PSFlet templates
    #################################################################

    Nspec = int(np.log(upper_wavelength_limit * 1. / lower_wavelength_limit) * R + 1.5)
    loglam_endpts = np.linspace(np.log(lower_wavelength_limit), np.log(upper_wavelength_limit), Nspec)
    loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1]) / 2.
    lam_endpts = np.exp(loglam_endpts)
    lam_midpts = np.exp(loglam_midpts)

    #################################################################
    # Compute the PSFlets integrated over small ranges in wavelength,
    # accounting for atmospheric+filter transmission.  Do this
    # calculation in parallel.
    #################################################################

    lenslet_ix, lenslet_iy = instrument.lenslet_ix, instrument.lenslet_iy

    #################################################################
    # Oversampling in x in final calibration frame.  If >1, fitting a
    # subpixel shift is possible in cube extraction.
    #################################################################

    if upsample:
        upsamp = 5
    else:
        upsamp = 1

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    consumers = [Consumer(tasks, results)
                 for i in range(ncpus)]
    for w in consumers:
        w.start()

    for i in range(upsamp * (Nspec - 1)):
        ilam = i // upsamp
        dx = (i % upsamp) * 1. / upsamp
        tool = copy.deepcopy(psftool)
        tool.interp_arr[0, 0] -= dx
        tasks.put(Task(i, primitives.make_polychrome,
                       (lam_endpts[ilam], lam_endpts[ilam + 1], hires_arrs,
                        lam_hires, tool, allcoef, lenslet_ix, lenslet_iy,
                        psflet_res, nlam, instrument.transmission)))
    for i in range(ncpus):
        tasks.put(None)

    polyimage = np.empty((Nspec - 1, npix_y, npix_x * upsamp), np.float32)

    print('Generating narrowband template images')
    for i in range(upsamp * (Nspec - 1)):
        frac_complete = (i + 1) * 1. / (upsamp * (Nspec - 1))
        N = int(frac_complete * 40)
        print('-' * N + '>' + ' ' * (40 - N) + ' %3d%% complete\r' % (int(100 * frac_complete)), end='')
        index, result = results.get()
        ilam = index // upsamp
        dx = (index % upsamp)
        polyimage[ilam, :, dx::upsamp] = result
    print('')

    #################################################################
    # Save the positions of the PSFlet centers to cut out the
    # appropriate regions in the least-squares extraction
    #################################################################

    xpos = []
    ypos = []
    good = []
    buffer_size = 8
    for i in range(Nspec - 1):
        _x, _y = psftool.return_locations(lam_midpts[i], allcoef, lenslet_ix, lenslet_iy)
        _good = (_x > buffer_size) * (_x < npix_x - buffer_size) * (_y > buffer_size) * (_y < npix_y - buffer_size)
        xpos += [_x]
        ypos += [_y]
        good += [_good]
    if upsamp > 1:
        np.save(outdir + 'polychromefullR%d.npy' % (R), polyimage)

    out = fits.HDUList(fits.PrimaryHDU(polyimage[:, :, ::upsamp].astype(np.float32)))
    out.writeto(outdir + 'polychromeR%d.fits' % (R), overwrite=True)

    outkey = fits.HDUList(fits.PrimaryHDU(lam_midpts))
    outkey.append(fits.PrimaryHDU(np.asarray(xpos)))
    outkey.append(fits.PrimaryHDU(np.asarray(ypos)))
    outkey.append(fits.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(outdir + 'polychromekeyR%d.fits' % (R), overwrite=True)

    print("Total time elapsed: %.0f seconds" % (time.time() - tstart))
    return None
