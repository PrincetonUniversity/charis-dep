#!/usr/bin/env python

#########################################################################
# A provisional routine for actually producing and returning data cubes.
#########################################################################

import copy
import glob
import json
import logging
import multiprocessing
import os
import re

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats

import charis
from charis import instruments, primitives, utr
from charis.image import Image
from charis.image.image_geometry import resample_image_cube
from charis.tools import (
    fit_background,
    sph_ifs_correct_spectral_xtalk,
    sph_ifs_fix_badpix,
)

log = logging.getLogger('main')


def getcube(dit=None, read_idx=[1, None], filename=None, calibdir=None,
            bgsub=True, bgpath=None, bg_scaling_without_mask=False,
            mask=True,
            gain=2, nonlinear_threshold=40000, noisefac=0,
            saveramp=False, R=30,
            individual_dits=False,
            method='lstsq', refine=True, crosstalk_scale=0.8,
            dc_xtalk_correction=False,
            linear_wavelength=False,
            suppressrn=True, fitshift=True, fitshift_nchunks=None,
            flatfield=True, smoothandmask=True,
            minpct=70, fitbkgnd=True, saveresid=False,
            maxcpus=multiprocessing.cpu_count(),
            instrument=None, resample=True,
            outdir="./",
            static_calibdir=None,
            preprocessing_only=False,
            verbose=True):
    """Construct and return a spectral data cube from a raw IFS detector frame.

    Reads the input FITS file, performs preprocessing (up-the-ramp combination
    for CHARIS or DIT handling for SPHERE, background subtraction, bad-pixel
    masking, flat fielding), loads calibration products, and extracts a 3-D
    data cube via least-squares or quasi-optimal extraction.

    Parameters
    ----------
    filename : str
        Path to the input FITS file containing the raw detector reads.
    calibdir : str
        Directory containing the calibration products (``PSFloc.fits``,
        ``polychromeR{R}.fits``, ``mask.fits``, etc.).
    dit : int or None, optional
        For SPHERE data with multiple DITs: zero-based index of the DIT to
        extract when ``individual_dits=True``. Ignored for CHARIS. Default None.
    read_idx : list of [int, int or None], optional
        First and last read indices to use in the up-the-ramp combination
        (CHARIS only). Default ``[1, None]``, i.e. use all reads from index 1.
    bgsub : bool, optional
        Subtract the thermal background loaded from ``calibdir/background.fits``.
        Default True.
    bgpath : str or None, optional
        Path to a custom background file. If None, uses ``calibdir/background.fits``.
        Default None.
    bg_scaling_without_mask : bool, optional
        If True, scale the background using all good pixels rather than only
        the region defined by ``calibdir/background_scaling_mask.fits``.
        Default False.
    mask : bool, optional
        Apply the bad-pixel mask ``calibdir/mask.fits``. Strongly recommended.
        Default True.
    gain : float, optional
        Detector gain in e⁻/DN, used to compute shot noise in the variance
        model. Default 2.
    nonlinear_threshold : int, optional
        Count level (DN) above which a full non-linearity fit is performed
        (SPHERE only). Default 40000.
    noisefac : float, optional
        Additional noise floor as a fraction of the count rate:
        ``var = readnoise + shotnoise + (noisefac * countrate)^2``.
        Values around 0.05 give a reduced chi-squared of ~1 in the lstsq fit.
        Default 0.
    saveramp : bool, optional
        Save the up-the-ramp combined 2-D image as an intermediate FITS file.
        Default False.
    R : int, optional
        Approximate spectral resolution lambda/delta(lambda) of the output cube.
        The pipeline selects the closest pre-built calibration
        (``polychromeR{R}.fits``). Resolutions above ~30 approach the pixel
        sampling limit and introduce strong inter-channel covariance. Default 30.
    individual_dits : bool, optional
        For SPHERE: extract the DIT specified by ``dit`` individually rather
        than averaging all DITs. Default False.
    method : {'lstsq', 'optext'}, optional
        Extraction algorithm. ``'lstsq'`` performs a full least-squares fit of
        the PSFlet templates. ``'optext'`` uses a quasi-optimal aperture
        extraction (faster, does not produce a residuals image). Default ``'lstsq'``.
    refine : bool, optional
        Perform a second lstsq pass to subtract nearest-neighbour lenslet
        crosstalk before the final extraction. Approximately doubles runtime.
        Also enables correlated read-noise suppression when ``suppressrn=True``.
        Default True.
    crosstalk_scale : float, optional
        Fractional amplitude of the crosstalk correction applied during the
        refinement step (``refine=True``). 1.0 applies the full predicted
        crosstalk; values slightly below 1.0 (e.g. 0.8–0.98) are more
        conservative and avoid over-subtraction. Default 0.8.
    dc_xtalk_correction : bool, optional
        Apply a spectral (DC) cross-talk correction via convolution before
        extraction (SPHERE only). Default False.
    linear_wavelength : bool, optional
        Use a linear (rather than logarithmic) wavelength grid for the
        ``'optext'`` extraction. Default False.
    suppressrn : bool, optional
        Estimate and subtract correlated read noise from the data using the
        residuals of the 50 % least-illuminated pixels. Requires
        ``refine=True``. Default True.
    fitshift : bool, optional
        Fit a position-dependent sub-pixel shift between the PSFlet templates
        and the data via cross-correlation before extraction. Improves accuracy
        when there is flexure or thermal drift between calibration and
        observation. Requires ``polychromefullR{R}.npy`` in ``calibdir``.
        Default True.
    fitshift_nchunks : int or None, optional
        Number of spatial chunks along each detector axis for the shift fit.
        The detector is divided into an N×N grid; each chunk measures its local
        shift independently. Larger values capture finer spatial variation at
        the cost of lower signal-to-noise per chunk. Default None uses 16 for
        CHARIS (128 px chunks on a 2048×2048 detector) and 1 for SPHERE (a
        single image-wide shift).
    flatfield : bool, optional
        Apply the pixel flat and lenslet flat corrections. Default True.
    smoothandmask : bool, optional
        After extraction, identify lenslets with anomalously low inverse
        variance (compared to their neighbours), set their ivar to zero, and
        replace their flux values with an inverse-variance-weighted local
        average (cosmetic only). Hides strong outliers; disable when computing
        quality metrics on the raw extraction. Default True.
    minpct : int, optional
        Minimum percentage of pixels that must be available to estimate the
        correlated read noise. If fewer pixels pass the threshold, read-noise
        suppression is skipped for that frame. Default 70.
    fitbkgnd : bool, optional
        Fit and subtract an undispersed background component in each
        microspectrum column during lstsq extraction. Default True.
    saveresid : bool, optional
        Write the 2-D residual image (preprocessed frame minus best-fit forward
        model) to ``{outdir}/{basename}_residuals.fits``. Only available with
        ``method='lstsq'``. Default False.
    maxcpus : int, optional
        Maximum number of CPU threads for OpenMP-parallelised Cython routines.
        Default ``multiprocessing.cpu_count()``.
    instrument : str or None, optional
        Override instrument auto-detection. Accepted values: ``'CHARIS'``,
        ``'SPHERE'``. Default None (detected from the FITS header).
    resample : bool, optional
        For SPHERE: resample the extracted cube from the native hexagonal
        lenslet grid to a regular rectangular grid. Default True.
    outdir : str, optional
        Directory where output FITS files are written. Default ``'./'``.
    static_calibdir : str or None, optional
        Override the default static calibration directory containing
        instrument-specific reference files shipped with the package.
        Default None.
    preprocessing_only : bool, optional
        Stop after preprocessing (background subtraction, bad-pixel correction,
        flat fielding) and write only the cleaned 2-D detector image. No cube
        extraction is performed. Default False.
    verbose : bool, optional
        Print progress and warning messages. Default True.

    Returns
    -------
    datacube : Image or tuple of (Image, Image)
        For CHARIS and SPHERE with ``resample=False``: a single ``Image``
        with ``datacube.data`` of shape ``(nlam, ny, nx)`` and
        ``datacube.ivar`` of the same shape.
        For SPHERE with ``resample=True``: a tuple
        ``(datacube_hex, datacube_resampled)`` where the first element is on
        the native hexagonal lenslet grid and the second on a resampled
        rectangular grid.
        If ``preprocessing_only=True``: a 2-D ``Image`` of the cleaned
        detector frame.

    Notes
    -----
    **Extraction methods:**

    *lstsq* — Solves for spectral coefficients at each lenslet and wavelength
    via SVD-based least squares using pre-built PSFlet template images. Returns
    the inverse-variance array (diagonal of the covariance matrix). Optionally
    saves a 2-D residual image (``saveresid=True``) and fits a background term
    per column (``fitbkgnd``).

    *optext* — Quasi-optimal aperture extraction. Each lenslet's spectrum is
    extracted with Gaussian pixel weights and interpolated onto a common
    wavelength grid. Natively samples a different wavelength set per lenslet,
    so the interpolation introduces inter-channel covariance. Faster than lstsq
    and does not require pre-built polychrome template images, but does not
    produce a residuals image.

    **Calibration file lookup:** The pipeline searches ``calibdir`` for
    ``polychromeR{R}.fits`` and ``polychromekeyR{R}.fits`` at the requested
    resolution ``R``. If not found it falls back to the nearest available
    resolution and logs a warning.
    """

    ################################################################
    # Initiate the header with critical data about the observation.
    # Then add basic information about the calibration data used to
    # extract a cube.
    ################################################################

    version = charis.__version__

    header = fits.getheader(filename)
    instrument, _, _ = instruments.instrument_from_data(
        header, calibration=False, static_calibdir=static_calibdir,
        interactive=False)

    if instrument.instrument_name == 'CHARIS':
        header = utr.metadata(filename, version=version)
    elif instrument.instrument_name == 'SPHERE':
        header = utr.metadata_SPHERE(filename, dit_idx=dit, version=version)
    else:
        raise ValueError("Only CHARIS and SPHERE instruments implemented.")

    try:
        calhead = fits.getheader(os.path.join(calibdir, 'cal_params.fits'))
        header.append(('comment', ''), end=True)
        header.append(('comment', '*' * 60), end=True)
        header.append(('comment', '*' * 21 + ' Calibration Data ' + '*' * 21), end=True)
        header.append(('comment', '*' * 60), end=True)
        header.append(('comment', ''), end=True)
        for key in calhead:
            header.append((key, calhead[key], calhead.comments[key]), end=True)
    except Exception:
        log.warn('Unable to append calibration parameters to FITS header.')

    ################################################################
    # Read in file and return an instance of the Image class with the
    # up-the-ramp combination of reads.  Subtract the thermal
    # background and apply a bad pixel mask.
    ################################################################

    calibration_path_instrument = instrument.calibration_path_instrument
    calibration_path_mode = instrument.calibration_path_mode

    maxcpus = min(maxcpus, multiprocessing.cpu_count())

    maskarr = None
    if mask:
        maskarr = fits.getdata(
            os.path.join(calibration_path_instrument, 'mask.fits'))
        bpm = np.logical_not(maskarr.astype('bool')).astype('int')

    if instrument.instrument_name == 'CHARIS':
        inImage = utr.calcramp(filename=filename, mask=maskarr, read_idx=read_idx,
                               header=header, gain=gain, noisefac=noisefac,
                               maxcpus=maxcpus)
        file_ending = ''

    elif instrument.instrument_name == 'SPHERE':
        data = fits.getdata(filename).astype('float64')
        readnoise = 6

        if maskarr is None:
            maskarr = np.ones((data.shape[-2], data.shape[-1]))
        if flatfield:
            pixelflat = fits.getdata(
                os.path.join(calibration_path_instrument, 'pixelflat.fits'))
            good_pixels = np.logical_and(pixelflat > 0.9, pixelflat < 1.1)
            maskarr[~good_pixels] = 0
        bpm = np.logical_not(maskarr.astype('bool')).astype('int')

        if nonlinear_threshold is not None:
            nonlinear = data > nonlinear_threshold
        else:
            nonlinear = np.zeros([data.shape[-2], data.shape[-1]]).astype('bool')

        if data.ndim == 3:
            if not individual_dits:
                ndit = len(data)
                data = np.mean(data, axis=0)
                file_ending = ''
            else:
                ndit = 1
                if dit >= data.shape[0]:
                    raise ValueError(
                        f"Requested DIT index {dit} but file only has {data.shape[0]} frames. "
                        "Header says NDIT={hdr['HIERARCH ESO DET NDIT']} but data shape disagrees."
                    )
                data = data[dit]
                file_ending = '_DIT_{:03d}'.format(dit)
        elif data.ndim == 2:
            file_ending = ''
            ndit = 1
        ivar = 1. / (abs(data) * gain * ndit + (data * noisefac)**2 + readnoise**2 * ndit)

        inImage = Image(data=data, ivar=ivar, header=header,
                        instrument_name=instrument.instrument_name)

    if flatfield and instrument.instrument_name == 'CHARIS':
        pixelflat = fits.getdata(
            os.path.join(calibration_path_instrument, 'pixelflat.fits'))

    if bgsub:
        if bgpath is not None:
            hdulist = fits.open(bgpath)
            bg = hdulist[0].data
            if bg is None:
                bg = hdulist[1].data
            hdulist.close()
            if len(bg.shape) == 3:
                bg = np.median(bg, axis=0)

        if instrument.instrument_name == 'SPHERE':
            # Region to match background counts for SPHERE IFS
            bgscalemask = fits.getdata(
                os.path.join(calibration_path_instrument, 'background_scaling_mask.fits')).astype('bool')
            if bgpath is not None:
                if bg_scaling_without_mask:
                    norm = inImage.data[maskarr == 1] / bg[maskarr == 1]
                else:
                    norm = inImage.data[bgscalemask & (bpm == 0)] / bg[bgscalemask & (bpm == 0)]

                norm = norm[np.isfinite(norm)].flatten()
                _, norm, _ = sigma_clipped_stats(
                    norm, sigma=2., sigma_lower=None, sigma_upper=None,
                    maxiters=5, cenfunc='median', stdfunc='std',
                    std_ddof=0)
                bg *= norm
                log.info("Background subtracted")
            else:
                log.info("Fitting background from template components")
                components = fits.getdata(
                    os.path.join(calibration_path_instrument, 'background_template.fits'))
                bg, bg_coef = fit_background(
                    image=inImage.data, components=components, bgmask=bgscalemask & (bpm == 0), outlier_percentiles=[2, 98])
                log.debug("Background template coefficients: %s", bg_coef)
            inImage.data -= bg

    if instrument.instrument_name == 'SPHERE':
        inImage.data = sph_ifs_fix_badpix(img=inImage.data, bpm=bpm)
        inImage.ivar = sph_ifs_fix_badpix(img=inImage.ivar, bpm=bpm)
        inImage.ivar[bpm.astype('bool')] = 0  # inImage.ivar[bpm.astype('bool')] * 1e-20
    
    if dc_xtalk_correction and instrument.instrument_name == 'SPHERE':
        inImage.data, convolved_image = sph_ifs_correct_spectral_xtalk(
            inImage.data, mask=~(bpm == 0))
        fits.writeto(
            re.sub(r'\.fits$','_convolved_image' + file_ending + '.fits',
                   os.path.join(outdir, os.path.basename(filename))),
            convolved_image, overwrite=True)

    header['bgsub'] = (bgsub, 'Subtract background count rate from a dark?')
    if saveramp:
        inImage.write(re.sub(r'\.fits$',f'_ramp{file_ending}.fits', os.path.join(
            outdir, os.path.basename(filename))))
        if instrument.instrument_name == 'SPHERE' and bgsub:
            fits.writeto(re.sub(r'\.fits$',f'_bg{file_ending}.fits', os.path.join(
                outdir, os.path.basename(filename))), bg, overwrite=True)

    ################################################################
    # If preprocessing_only, apply pixel flat to the image and
    # return early without cube extraction.
    ################################################################

    if preprocessing_only:
        if flatfield:
            good_pixel_mask = np.logical_not(bpm.astype('bool'))
            inImage.data[good_pixel_mask] = (
                inImage.data[good_pixel_mask] / pixelflat[good_pixel_mask])
            if instrument.instrument_name == 'SPHERE':
                inImage.data = sph_ifs_fix_badpix(img=inImage.data, bpm=bpm)
                inImage.ivar[good_pixel_mask] *= pixelflat[good_pixel_mask]**2
                inImage.ivar = sph_ifs_fix_badpix(img=inImage.ivar, bpm=bpm)
                inImage.ivar[bpm.astype('bool')] = 0

        header['preonly'] = (True, 'Preprocessing only, no cube extraction')
        inImage.header = header
        extrahdr = fits.getheader(filename)
        inImage.extraheader = extrahdr
        outname = re.sub(
            r'\.fits$', '_preprocessed' + file_ending + '.fits',
            os.path.join(outdir, os.path.basename(filename)))
        inImage.write(outname)
        return inImage

    ################################################################
    # Read in necessary calibration files and extract the data cube.
    # Optionally fit for a position-dependent offset
    ################################################################

    header.append(('comment', ''), end=True)
    header.append(('comment', '*' * 60), end=True)
    header.append(('comment', '*' * 22 + ' Cube Extraction ' + '*' * 21), end=True)
    header.append(('comment', '*' * 60), end=True)
    header.append(('comment', ''), end=True)

    if flatfield:
        lensletflat = fits.getdata(
            os.path.join(
                calibration_path_mode, 'lensletflat.fits')).astype('float64')
        good_pixel_mask = np.logical_not(bpm.astype('bool'))
    else:
        lensletflat = None
        good_pixel_mask = np.ones_like(inImage.data).astype('bool')

    header['flatfld'] = (flatfield, 'Flatfield the detector and lenslet array?')
    datacube = None

    if method == 'lstsq' or suppressrn or refine:
        try:
            keyfile = fits.open(os.path.join(calibdir, 'polychromekeyR%d.fits' % (R)))
            R2 = R
        except IOError:
            keyfilenames = glob.glob(calibdir + '*polychromekeyR*.fits')
            if len(keyfilenames) == 0:
                raise IOError("No key file found in " + calibdir)
            R2 = int(re.sub('.*keyR', '', re.sub(r'\.fits$','', keyfilenames[0])))
            keyfile = fits.open(os.path.join(calibdir, 'polychromekeyR%d.fits' % (R2)))
            if verbose:
                log.warning("Calibration files not found at requested resolution R=%d, using R=%d instead.", R, R2)

        if fitshift:
            try:
                psflets = np.load(os.path.join(calibdir, 'polychromefullR%d.npy' % (R2)))
            except FileNotFoundError:
                if verbose:
                    log.warning("Oversampled PSFlets for fitshift not found, reverting to not shifting.")
                fitshift = False
        if fitshift:
            offsets = instrument.offsets
            if fitshift_nchunks is None:
                # Default: CHARIS uses a 16×16 grid; all other instruments
                # use a single image-wide shift.
                nchunks = 16 if instrument.instrument_name == 'CHARIS' else 1
            else:
                nchunks = int(fitshift_nchunks)
            dx = max(1, inImage.data.shape[0] // nchunks)
            try:
                psflets = primitives.calc_offset(
                    psflets, inImage, offsets, dx=dx, maxcpus=maxcpus)
                print("Fit shift successful, PSFlet positions adjusted across the detector.")
            except Exception as e:
                if verbose:
                    log.warning("Fit shift failed, continuing without fitting shift: %s", e)
                fitshift = False
        if not fitshift:
            psflets = fits.getdata(os.path.join(calibdir, 'polychromeR%d.fits' % (R2)))

        header.append(('fitshift', fitshift, 'Fit a subpixel shift in PSFlet locations?'), end=True)
        lam_midpts = keyfile[0].data
        lam_psflets = lam_midpts.copy()
        x = keyfile[1].data
        y = keyfile[2].data
        good = keyfile[3].data
        keyfile.close()

        if flatfield:
            # Only apply flat field to pixels with non-anomalous flat field values
            psflets[:, good_pixel_mask] = psflets[:, good_pixel_mask] * pixelflat[good_pixel_mask]

        ############################################################
        # Do an initial least-squares fit to remove correlated read
        # noise if method = optext and suppressrn = True
        ############################################################

        if method != 'lstsq':
            residuals, coefs = primitives.fit_spectra(
                inImage, psflets, lam_midpts, x, y, good,
                instrument=instrument,
                header=inImage.header, lensletflat=lensletflat, refine=refine,
                suppressreadnoise=suppressrn, smoothandmask=smoothandmask,
                minpct=minpct, fitbkgnd=fitbkgnd, maxcpus=maxcpus,
                return_corrnoise=True)
            if suppressrn:
                corrnoise = residuals
                inImage.data -= corrnoise
            elif refine and not dc_xtalk_correction:
                # This is to remove crosstalk: we remove the
                # contribution of each lenslet to its nearest
                # neighbors, scaled by crosstalk_scale.  In this case
                # corrnoise actually refers to the residuals.  We take
                # a mixture of the residuals and the data, with the
                # mixture scaled by crosstalk_scale; the components
                # from the coefficients will be added back in later.
                if crosstalk_scale > 0:
                    coefs *= crosstalk_scale
                    inImage.data += crosstalk_scale*(residuals - inImage.data)
                log.info("Crosstalk scale: %s", crosstalk_scale)
        else:
            result = primitives.fit_spectra(
                inImage, psflets, lam_midpts, x, y, good,
                instrument=instrument,
                header=inImage.header, lensletflat=lensletflat, refine=refine,
                suppressreadnoise=suppressrn, smoothandmask=smoothandmask,
                minpct=minpct, fitbkgnd=fitbkgnd, returnresid=saveresid,
                maxcpus=maxcpus)
            if saveresid:
                datacube, resid = result
                resid.write(
                    re.sub(r'\.fits$','_residuals' + file_ending + '.fits',
                           os.path.join(outdir, os.path.basename(filename))))
            else:
                datacube = result

    if method == 'optext' or method == 'apphot3' or method == 'apphot5':
        loc = primitives.PSFLets(load=True, infiledir=calibdir)

        if linear_wavelength:
            lam_midpts = np.linspace(
                instrument.wavelength_range[0].value,
                instrument.wavelength_range[1].value,
                39)
        else:
            lam_midpts, _ = instrument.wavelengths(
                instrument.wavelength_range[0].value,
                instrument.wavelength_range[1].value,
                R)

        try:
            sig = fits.getdata(os.path.join(calibdir, 'PSFwidths.fits'))
        except IOError:
            log.warning("Failed to load PSFwidths.fits, defaulting to standard PSF width.")
            sig = 0.7

        if method == 'apphot3' or method == 'apphot5':
            sig = 1e10
            if method == 'apphot3':
                delt_x = 3
            else:
                delt_x = 5
        else:
            delt_x = 5

        if flatfield:
            inImage.data[good_pixel_mask] = inImage.data[good_pixel_mask] / \
                pixelflat[good_pixel_mask]
            if instrument.instrument_name == 'SPHERE':
                inImage.data = sph_ifs_fix_badpix(img=inImage.data, bpm=bpm)
                inImage.ivar[good_pixel_mask] *= pixelflat[good_pixel_mask]**2
                inImage.ivar = sph_ifs_fix_badpix(img=inImage.ivar, bpm=bpm)
                inImage.ivar[bpm.astype('bool')] = 0  # inImage.ivar[bpm.astype('bool')] / 1.2

        # If we did the crosstalk correction, we need to add the model
        # spectra back in and do a modified optimal extraction.
        if refine:
            _coefs, _psflets, _lam_psflets = coefs, psflets, lam_psflets
        else:
            _coefs, _psflets, _lam_psflets = None, None, None

        datacube = primitives.optext_spectra(inImage, loc, lam_midpts,
                                             instrument=instrument, delt_x=delt_x,
                                             coefs_in=_coefs, psflets=_psflets,
                                             lampsflets=_lam_psflets,
                                             sig=sig, header=inImage.header,
                                             lensletflat=lensletflat,
                                             smoothandmask=smoothandmask,
                                             maxcpus=maxcpus)

    if datacube is None:
        raise ValueError("Datacube extraction method " + method + " not implemented.")

    ################################################################
    # Add the original header for reference as the last HDU
    ################################################################
    extrahdr = fits.getheader(filename)
    datacube.extraheader = extrahdr
    ################################################################
    # Add WCS for the cube
    # for now assume the image is centered on the cube
    # in practice, we will have to register things with
    # the satellite spots
    ################################################################

    if instrument.instrument_name == 'CHARIS':
        ydim, xdim = datacube.data[0].shape
        rot_angle = 113  # empirically determined
        utr.addWCS(datacube.header, xpix=ydim // 2, ypix=xdim // 2,
                   xpixscale=-0.0164 / 3600., ypixscale=0.0164 / 3600.,
                   extrarot=rot_angle)

    if instrument.instrument_name == 'SPHERE' and resample:
        clip_info_file = os.path.join(
            calibration_path_instrument, 'hexagon_mapping_calibration.json')
        with open(clip_info_file) as json_data:
            clip_infos = json.load(json_data)
        datacube_resampled = copy.copy(datacube)
        datacube_resampled.data = resample_image_cube(
            datacube.data, clip_infos, hexagon_size=1 / np.sqrt(3))
        datacube_resampled.ivar = resample_image_cube(
            datacube.ivar, clip_infos, hexagon_size=1 / np.sqrt(3))

        datacube.write(
            re.sub(r'\.fits$','_cube' + file_ending + '.fits',
                   os.path.join(outdir, os.path.basename(filename))))
        datacube_resampled.write(
            re.sub(r'\.fits$','_cube_resampled' + file_ending + '.fits',
                   os.path.join(outdir, os.path.basename(filename))))
        return datacube, datacube_resampled

    else:
        datacube.write(
            re.sub(r'\.fits$','_cube.fits',
                   os.path.join(outdir, os.path.basename(filename))))
        return datacube
