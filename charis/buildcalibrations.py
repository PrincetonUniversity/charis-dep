#!/usr/bin/env python

import copy
import glob
import logging
import multiprocessing
import os
import re
import time
from builtins import range

import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy import interpolate, ndimage
from tqdm import tqdm

from charis import primitives, utr
from charis.image import Image
from charis.parallel import Consumer, Task
from charis.tools import expected_spectrum

log = logging.getLogger('main')


def read_in_file(infile, instrument, calibration_wavelength=None,
                 ncpus=1, mask=None, bgfiles=[]):
    """Read and preprocess a monochromatic calibration flat-field image.

    Handles both CHARIS (up-the-ramp ramp fitting) and SPHERE (multi-DIT
    mean after outlier trimming) input formats. Returns the preprocessed
    ``Image`` and a minimal FITS header with calibration metadata, ready to
    pass directly to :func:`buildcalibrations`.

    Parameters
    ----------
    infile : str
        Path or glob pattern to the raw calibration FITS file. For CHARIS,
        a sequence of reads; for SPHERE, a cube of DITs.
    instrument : Instrument
        Instrument configuration object providing ``instrument_name``,
        ``calibration_path_instrument``, and ``gain``.
    calibration_wavelength : astropy.units.Quantity or None, optional
        Calibration wavelength(s) with units (e.g. ``[987.72] * u.nm``).
        Required for CHARIS (written to the output header as ``cal_lam``).
        Default None.
    ncpus : int, optional
        Number of CPU threads for the up-the-ramp combination (CHARIS only).
        Default 1.
    mask : ndarray of int or None, optional
        Bad-pixel mask with shape ``(ny, nx)``, where 0 marks bad pixels and
        1 marks good pixels. If None, loaded from
        ``instrument.calibration_path_instrument/mask.fits``. Default None.
    bgfiles : list of str, optional
        Paths to background (dark) frames to subtract (CHARIS only).
        Default ``[]`` (no background subtraction).

    Returns
    -------
    inImage : Image
        Preprocessed flat-field image with ``data`` (count rate, shape
        ``(ny, nx)``) and ``ivar`` (inverse variance, same shape).
    hdr : astropy.io.fits.Header
        Minimal FITS header containing calibration metadata: input filename,
        MJD observation date, observing band, and calibration wavelength.
    """

    if mask is None:
        mask = fits.getdata(
            os.path.join(instrument.calibration_path_instrument, 'mask.fits'))

    hdr = fits.PrimaryHDU().header
    hdr.clear()
    infilelist = glob.glob(infile)
    if len(infilelist) == 0:
        raise ValueError("No CHARIS file found for calibration.")

    hdr['calfname'] = (re.sub('.*/', '', infilelist[0]),
                       'Monochromatic image used for calibration')
    try:
        hdr['cal_date'] = (fits.getheader(infilelist[0])['mjd'],
                           'MJD date of calibration image')
    except Exception:
        hdr['cal_date'] = ('unavailable', 'MJD date of calibration image')

    hdr['cal_band'] = (instrument.observing_mode,
                       'Band/mode of calibration image (J/H/K/Broadband)')
    if instrument.instrument_name == 'CHARIS':
        hdr['cal_lam'] = (calibration_wavelength.value[0], 'Wavelength of calibration image (nm)')

    ###############################################################
    # Mean background count rate, weighted by inverse variance
    ###############################################################

    # NOTE: Handling of BG frames for CHARIS defunct, not used in further routines.
    if instrument.instrument_name == 'CHARIS':
        print('Computing ramps from sequences of raw reads')
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
        # NOTE: Actually only takes into account one file at the monent
        # Has to be changed to appending routine and ensure same dimension
        readnoise = 6.
        for filename in infilelist:
            data = fits.getdata(filename)
            if data.ndim == 3 and data.shape[0] > 1:
                data = np.sort(data.astype('float64'), axis=0)
                data = np.mean(data[1:-1], axis=0) * mask
            elif data.ndim == 3 and data.shape[0] == 1:
                data = data[0] * mask
            else:
                data = data * mask
            var = np.abs(data) * instrument.gain + readnoise**2
            var[mask == 0] = 1e20
            ivar = 1. / var

            inImage = Image(data=data, ivar=ivar,
                            instrument_name=instrument.instrument_name)

    return inImage, hdr


def buildcalibrations(inImage, instrument, inLam, mask=None,
                      order=None, upsample=True, header=None,
                      ncpus=multiprocessing.cpu_count(),
                      nlam=10, outdir="./",
                      stellar_temperature=None,
                      verbose=True):
    """Build the calibration files required to extract spectral data cubes.

    Fits PSFlet positions to a monochromatic flat-field calibration image,
    computes the wavelength-dependent lenslet position grid, and generates the
    polychrome PSFlet template images used by
    :func:`~charis.extractcube.getcube`.

    Parameters
    ----------
    inImage : Image
        Preprocessed monochromatic flat-field image (count rate and inverse
        variance). Typically the output of :func:`read_in_file`. For CHARIS
        this is a laser flat; for SPHERE it is an internal wavelength
        calibration lamp frame.
    instrument : Instrument
        Instrument configuration object (e.g. ``instruments.SPHERE('YH')``).
        Provides wavelength range, lenslet grid geometry, PSFlet resolution,
        and transmission curve. The output wavelength grid is derived from
        ``instrument.lam_midpts`` and ``instrument.lam_endpts``; override
        these attributes (on a copy) to build calibrations at a non-default
        spectral resolution.
    inLam : array-like of float
        Calibration wavelength(s) in nm corresponding to ``inImage``.
    mask : ndarray of int or None, optional
        Bad-pixel mask with shape ``(ny, nx)``, where 0 marks bad pixels and
        1 marks good pixels. If None, loaded from the instrument's static
        calibration directory. Default None.
    order : int or None, optional
        Polynomial order for the lenslet position fit as a function of
        wavelength. Default None uses ``instrument.wavelengthpolyorder``.
    upsample : bool, optional
        If True, build the oversampled PSFlet templates
        (``polychromefullR{R}.npy``, 5× in the cross-dispersion direction)
        in addition to the standard ``polychromeR{R}.fits``. The oversampled
        file is required for sub-pixel shift fitting (``fitshift=True`` in
        :func:`~charis.extractcube.getcube`). Building the oversampled file
        is slow at high spectral resolutions. Default True.
    header : astropy.io.fits.Header or None, optional
        FITS header to which calibration shift metadata (``cal_dx``,
        ``cal_dy``, ``cal_dphi``) are appended before being written to
        ``cal_params.fits``. Default None.
    ncpus : int, optional
        Number of parallel worker processes for building the polychrome
        template images. Default ``multiprocessing.cpu_count()``.
    nlam : int, optional
        Number of monochromatic PSFlet images integrated per wavelength
        channel when building the polychrome templates. Higher values give
        smoother band-integrated templates at the cost of computation time.
        Default 10.
    outdir : str, optional
        Directory where all calibration files are written. Default ``'./'``.
    stellar_temperature : astropy.units.Quantity or None, optional
        If provided, weight the polychrome template integration by a blackbody
        spectrum at this temperature convolved with the instrument transmission,
        rather than using a spectrally flat weighting. Default None.
    verbose : bool, optional
        Print progress messages and total elapsed time. Default True.

    Returns
    -------
    None
        All output is written to ``outdir``. Files produced:

        ``PSFloc.fits``
            Lenslet pixel positions as a function of wavelength
            (resolution-independent).
        ``PSFwidths.fits``
            Cross-dispersion PSFlet widths at each lenslet and wavelength,
            used by the optimal extraction (resolution-independent).
        ``polychromeR{R}.fits``
            Band-integrated PSFlet template images, shape ``(nlam, ny, nx)``,
            where ``R`` is ``instrument.resolution``.
        ``polychromekeyR{R}.fits``
            Multi-extension FITS containing the wavelength grid, lenslet x/y
            positions, and a good-lenslet boolean mask, keyed to resolution
            ``R``.
        ``polychromefullR{R}.npy`` *(only if* ``upsample=True`` *)*
            Oversampled (5×) PSFlet templates for sub-pixel shift fitting,
            shape ``(nlam, ny, 5*nx)``.
        ``cal_params.fits``
            FITS file storing the ``header`` supplied to this function,
            augmented with the measured shift offsets.

    Notes
    -----
    The wavelength solution in ``lamsol.dat`` in the static calibration
    directory is used as the reference. This function fits the PSFlet positions
    in ``inImage`` and records the shift (translation + rotation) relative to
    that reference. The updated position solution is used for all subsequent
    polychrome template generation.

    To build calibrations at a non-default spectral resolution, override
    ``instrument.lam_midpts`` and ``instrument.lam_endpts`` on a
    ``copy.copy`` of the instrument before calling this function, to avoid
    mutating the shared instrument object.
    """

    if order is None:
        order = instrument.wavelengthpolyorder

    calibration_path_mode = instrument.calibration_path_mode

    tstart = time.time()

    ncpus = min(ncpus, multiprocessing.cpu_count())

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

    log.info("Loading wavelength solution from " + calibration_path_mode + "/lamsol.dat")
    lam = np.loadtxt(os.path.join(calibration_path_mode, "lamsol.dat"))[:, 0]
    allcoef = np.loadtxt(os.path.join(calibration_path_mode, "lamsol.dat"))[:, 1:]
    psftool = primitives.PSFLets()
    oldcoef = []
    for cal_lam in inLam:
        oldcoef += [psftool.monochrome_coef(cal_lam, lam, allcoef, order=order).tolist()]
    if verbose:
        print('Generating new wavelength solution')
    _, _, _, newcoef = primitives.locatePSFlets(
        inImage, instrument, polyorder=3, coef=oldcoef, fitorder=1)

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

    hires_list = np.sort(glob.glob(os.path.join(
        calibration_path_mode, 'hires_psflets_lam*.fits')))
    hires_arrs = [fits.getdata(filename) for filename in hires_list]
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

    lenslet_ix, lenslet_iy = instrument.lenslet_ix, instrument.lenslet_iy

    #################################################################
    # Compute the PSFlets integrated over small ranges in wavelength,
    # accounting for atmospheric+filter transmission.  Do this
    # calculation in parallel.
    #################################################################

    if stellar_temperature is not None:
        print('Applying spectrum of {}.'.format(stellar_temperature.__str__()))
        transmission = expected_spectrum(
            stellar_temperature=stellar_temperature,
            wavelength=instrument.transmission[:, 0] * u.nm,
            transmission=instrument.transmission[:, 1])
    else:
        transmission = instrument.transmission

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
                        psflet_res, nlam, transmission)))
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
        _good = (_x > buffer_size) * (_x < npix_x - buffer_size) * \
            (_y > buffer_size) * (_y < npix_y - buffer_size)
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

    if verbose:
        print("Total time elapsed: %.0f seconds" % (time.time() - tstart))

    return None
