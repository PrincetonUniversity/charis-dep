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
from tqdm import tqdm

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


def read_in_file(infile, instrument, calibration_wavelength=None,
                 ncpus=1, mask=None, calibdir=None, bgfiles=[],
                 outdir="./"):
    if calibdir is None:
        calibdir = instrument.calibration_path
    if mask is None:
        mask = fits.open(os.path.join(calibdir, 'mask.fits'))[0].data

    hdr = fits.PrimaryHDU().header
    hdr.clear()
    infilelist = glob.glob(infile)
    if len(infilelist) == 0:
        raise ValueError("No CHARIS file found for calibration.")

    hdr['calfname'] = (re.sub('.*/', '', infilelist[0]),
                       'Monochromatic image used for calibration')
    try:
        hdr['cal_date'] = (fits.open(infilelist[0])[0].header['mjd'],
                           'MJD date of calibration image')
    except:
        hdr['cal_date'] = ('unavailable', 'MJD date of calibration image')

    hdr['cal_band'] = (instrument.observing_mode, 'Band/mode of calibration image (J/H/K/Broadband)')
    if instrument.instrument_name == 'CHARIS':
        hdr['cal_lam'] = (calibration_wavelength.value[0], 'Wavelength of calibration image (nm)')

    ###############################################################
    # Mean background count rate, weighted by inverse variance
    ###############################################################

    print('Computing ramps from sequences of raw reads')
    if instrument.instrument_name == 'CHARIS':
        num = 0.
        denom = 1e-100
        ibg = 1
        for idx, bgfile in enumerate(bgfiles):
            bg = utr.calcramp(filename=bgfile, mask=mask, maxcpus=ncpus)
            num = num + bg.data * bg.ivar
            denom = denom + bg.ivar
            hdr['bkgnd%03d' % (ibg)] = (re.sub('.*/', '', bgfile),
                                        'Dark(s) used for background subtraction')
            ibg += 1
        if len(bgfiles) > 0:
            background = Image(data=num / denom, ivar=1. / denom,
                               instrument_name=instrument.instrument_name)
            background.write('background.fits')
        else:
            hdr['bkgnd001'] = ('None', 'Dark(s) used for background subtraction')

    # elif instrument.instrument_name == 'SPHERE':
    #     bgs = []
    #     for idx, bgfile in enumerate(bgfiles):
    #         bg = fits.getdata(bgfile)
    #         if len(bg.shape) == 3:
    #             bg = np.sort(bg.astype('float64'), axis=0)
    #                 bgs.append(np.mean(bg[1:-1], axis=0) * mask)
    #         hdr['bkgnd%03d' % (idx + 1)] = (re.sub('.*/', '', bgfile),
    #                                     'Dark(s) used for background subtraction')
    #     bg = np.mean(np.array(bgs), axis=0)
    #     if len(bgiles) > 0:
    #         inImage = Image(data=bg, ivar=mask.astype('float64'),
    #                         instrument_name=instrument.instrument_name)
    #         background.write('background.fits')
    #     else:
    #         hdr['bkgnd001'] = ('None', 'Dark(s) used for background subtraction')
    # else:
    #     raise ValueError('Instrument not defined.')

    ###############################################################
    # Monochromatic flatfield image
    ###############################################################

    num = 0
    denom = 1e-100
    if instrument.instrument_name == 'CHARIS':
        for filename in infilelist:
            im = utr.calcramp(filename=filename, mask=mask, maxcpus=ncpus)
            num = num + im.data * im.ivar
            denom = denom + im.ivar
            inImage = Image(data=num, ivar=mask * 1. / denom,
                            instrument_name=instrument.instrument_name)

    elif instrument.instrument_name == 'SPHERE':
        for filename in infilelist:
            data = fits.getdata(filename)
            if len(data.shape) == 3:
                data = np.sort(data.astype('float64'), axis=0)
                data = np.mean(data[1:-1], axis=0) * mask
            inImage = Image(data=data, ivar=mask.astype('float64'),
                            instrument_name=instrument.instrument_name)

    return inImage, hdr


def buildcalibrations(inImage, instrument, inLam, mask=None, calibdir=None,
                      order=None, upsample=True, header=None,
                      ncpus=multiprocessing.cpu_count(),
                      nlam=10, outdir="./", verbose=True):
    """
    Build the calibration files needed to extract data cubes from
    sequences of CHARIS reads.

    Inputs:
    1. inImage:  Image object, should include count rate and ivar for
                 a narrow-band flatfield calibration image.
    2. instrument: instrument object
    3. inLam:    wavelength in nm of inImage
    4. mask:     bad pixel mask, =0 for bad pixels
    5. calibdir:    directory where master calibration files live
                 default: None, use instrument class info


    Optional inputs:

    order:    int, order of polynomial fit to position(lambda).
                 Default None (taken from instrument class).
    header:   FITS header, to which will be appended the shifts
                 and rotation angle between the stored and the fitted
                 wavelength solutions.  Default None.
    ncpus:    number of threads for multithreading.
                 Default multiprocessing.cpu_count()

    nlam: int, number of monochromatic PSFlets per integrated PSFlet
    outdir:   directory in which to place

    Returns None, writes calibration files to outdir.

    """

    if order is None:
        order = instrument.wavelengthpolyorder

    if calibdir is None:
        calibdir = instrument.calibration_path

    if mask is None:
        mask = fits.open(os.path.join(calibdir, 'mask.fits'))[0].data

    tstart = time.time()
    lower_wavelength_limit, upper_wavelength_limit = instrument.wavelength_range.value
    R = instrument.resolution

    # inImage = read_in_file(infile=, instrument=instrument, calibdir=calibdir)
    npix_y, npix_x = inImage.data.shape

    #################################################################
    # Fit the PSFlet positions on the input image, compute the shift
    # in the mean position (coefficients 0 and 10) and in the linear
    # component of the fit (coefficients 1, 4, 11, and 14).  The
    # comparison point is the location solution for this wavelength in
    # the existing calibration files.
    #################################################################

    log.info("Loading wavelength solution from " + calibdir + "/lamsol.dat")
    lam = np.loadtxt(os.path.join(calibdir, "lamsol.dat"))[:, 0]
    allcoef = np.loadtxt(os.path.join(calibdir, "lamsol.dat"))[:, 1:]
    psftool = primitives.PSFLets()
    oldcoef = []
    for cal_lam in inLam:
        oldcoef += [psftool.monochrome_coef(cal_lam, lam, allcoef, order=order).tolist()]
    if verbose:
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
    if verbose:
        print('x, y, phi shift: %.6f, %.6f, %.6f' % (dx, dy, dphi))
    if header is not None:
        header['cal_dx'] = (dx, 'x-shift from archival spot positions (pixels)')
        header['cal_dy'] = (dy, 'y-shift from archival spot positions (pixels)')
        header['cal_dphi'] = (dphi, 'Rotation from archival spot positions (radians)')

    #################################################################
    # Load the high-resolution PSFlet images and associated
    # wavelengths.
    #################################################################

    hires_list = np.sort(glob.glob(os.path.join(calibdir, 'hires_psflets_lam*.fits')))
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
    out.writeto(os.path.join(outdir, 'PSFwidths.fits'), overwrite=True)

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

    Nspec = len(instrument.lam_midpts)

    for i in range(upsamp * Nspec):
        ilam = i // upsamp
        dx = (i % upsamp) * 1. / upsamp
        tool = copy.deepcopy(psftool)
        tool.interp_arr[0, 0] -= dx
        tasks.put(Task(i, primitives.make_polychrome,
                       (instrument.lam_endpts[ilam], instrument.lam_endpts[ilam + 1], hires_arrs,
                        lam_hires, tool, allcoef, lenslet_ix, lenslet_iy,
                        psflet_res, nlam, instrument.transmission)))
    for i in range(ncpus):
        tasks.put(None)

    polyimage = np.empty((Nspec, npix_y, npix_x * upsamp), np.float32)

    if verbose:
        print('Generating narrowband template images')
        for i in tqdm(range(upsamp * Nspec), miniters=ncpus):
            # if verbose:
            #     frac_complete = (i + 1) * 1. / (upsamp * (Nspec - 1))
            #     N = int(frac_complete * 40)
            #     print('-' * N + '>' + ' ' * (40 - N) + ' %3d%% complete\r' % (int(100 * frac_complete)), end='')
            index, result = results.get()
            ilam = index // upsamp
            dx = (index % upsamp)
            polyimage[ilam, :, dx::upsamp] = result
    else:
        for i in range(upsamp * Nspec):
            index, result = results.get()
            ilam = index // upsamp
            dx = (index % upsamp)
            polyimage[ilam, :, dx::upsamp] = result

    if verbose:
        print('')

    #################################################################
    # Save the positions of the PSFlet centers to cut out the
    # appropriate regions in the least-squares extraction
    #################################################################

    xpos = []
    ypos = []
    good = []
    buffer_size = 8
    for i in range(Nspec):
        _x, _y = psftool.return_locations(instrument.lam_midpts[i], allcoef, lenslet_ix, lenslet_iy)
        _good = (_x > buffer_size) * (_x < npix_x - buffer_size) * (_y > buffer_size) * (_y < npix_y - buffer_size)
        xpos += [_x]
        ypos += [_y]
        good += [_good]
    if upsamp > 1:
        np.save(os.path.join(outdir, 'polychromefullR%d.npy' % (R)), polyimage)

    out = fits.HDUList(fits.PrimaryHDU(polyimage[:, :, ::upsamp].astype(np.float32)))
    out.writeto(os.path.join(outdir, 'polychromeR%d.fits' % (R)), overwrite=True)

    outkey = fits.HDUList(fits.PrimaryHDU(instrument.lam_midpts))
    outkey.append(fits.PrimaryHDU(np.asarray(xpos)))
    outkey.append(fits.PrimaryHDU(np.asarray(ypos)))
    outkey.append(fits.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(os.path.join(outdir, 'polychromekeyR%d.fits' % (R)), overwrite=True)

    out = fits.HDUList(fits.PrimaryHDU(None, header))
    out.writeto(os.path.join(outdir, 'cal_params.fits'), overwrite=True)

    shutil.copy(os.path.join(calibdir, 'lensletflat.fits'), os.path.join(outdir, 'lensletflat.fits'))

    for filename in ['mask.fits', 'pixelflat.fits']:
        shutil.copy(os.path.join(calibdir, filename), os.path.join(outdir, filename))

    if verbose:
        print("Total time elapsed: %.0f seconds" % (time.time() - tstart))
    return None
