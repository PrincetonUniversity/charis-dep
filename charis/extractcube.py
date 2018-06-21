#!/usr/bin/env python

#########################################################################
# A provisional routine for actually producing and returning data cubes.
#########################################################################

from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import range
import os
import configparser
import copy
import glob
import logging
import multiprocessing
import pickle
import re
import sys
import time

import numpy as np
from astropy.io import fits

from pdb import set_trace

try:
    import instruments
    import primitives
    import utr
    from image import Image
    from image.map_hexagon_to_rectilinear import resample_image_cube
except ImportError:
    from charis import instruments
    from charis import primitives
    from charis import utr
    from charis.image import Image
    from charis.image.map_hexagon_to_rectilinear import resample_image_cube
    import charis

log = logging.getLogger('main')


def getcube(filename, read_idx=[1, None], calibdir='calibrations/20160408/',
            bgsub=True, bgpath=None, mask=True, gain=2, noisefac=0, saveramp=False, R=30,
            method='lstsq', refine=True, suppressrn=True, fitshift=True,
            flatfield=True, smoothandmask=True,
            minpct=70, fitbkgnd=True, saveresid=False,
            maxcpus=multiprocessing.cpu_count(),
            instrument=None, resample=True, verbose=True):
    """Provisional routine getcube.  Construct and return a data cube
    from a set of reads.

    Inputs:
    1. filename: name of the file containing the up-the-ramp reads.
                 Should include the full path/to/file.
    Optional inputs:
    1. read_idx: list of two numbers, the first and last reads to use in
                 the up-the-ramp combination.  Default [2, None], i.e.,
                 discard the first read and use all of the rest.
    2. calibdir: name of the directory containing the calibration files.
                 Default calibrations/20160408/
    3. bgsub:    Subtract the file background.fits in calibdir?  Default
                 True.
    4. mask:     Apply the bad pixel mask mask.fits in the directory
                 calibdir?  Strongly recommended.  Default True.
    5. gain:     Detector gain, used to compute shot noise.  Default 2.
    6. noisefac: Extra factor of noise to account for imperfect lenslet
                 models:
                 variance = readnoise + shotnoise + noisefac*countrate
                 Default zero, values of around 0.05 should give
                 reduced chi squared values of around 1 in the fit.
    7. R:        integer, approximate resolution lam/delta(lam) of the
                 extracted data cube.  Resolutions higher than ~25 or 30
                 are comparable to or finer than the pixel sampling and
                 are not recommended--there is very strong covariance.
                 Default 30.
    8. method:   string, method used to extract data cube.  Should be
                 either 'lstsq' for a least-squares extraction or
                 'optext' for a quasi-optimal extraction.  Default
                 'lstsq'
    9. refine:   Fit the data cube twice to account for nearest neighbor
                 crosstalk?  Approximately doubles runtime.  This option
                 also enables read noise suppression (below).  Default
                 True
    10. suppress_rn: Remove correlated read noise between channels using
                 the residuals of the 50% least illuminated pixels?
                 Default True.
    11. fitshift: Fit a subpixel shift in the psflet locations across
                 the detector?  Recommended except for quicklook.  Cost
                 is modest compared to cube extraction

    Returns:
    1. datacube: an instance of the Image class containing the data cube.

    Steps performed:

    1. Up-the-ramp combination.  As of yet no special read noise
    suppression (just a channel-by-channel correction of the reference
    voltages).  Do a full nonlinear fit for high count pixels, and
    remove an exponential decay of the reference voltage in the first
    read.
    2. Subtraction of thermal background from file in calibration
    directory.
    3. Application of hot pixel mask (as zero inverse variances).
    4. Load calibration files from calibration directory for the data
    cube extraction, optionally fit for subpixel shifts across the
    detector.
    5. Extract the data cube.

    Notes for now: the quasi-optimal extraction isn't really an
    optimal extraction.  Every lenslet's spectrum natively samples a
    different set of wavelengths, so right now they are all simply
    interpolated onto the same wavelength array.  This is certainly
    not optimal, but I don't see any particularly good alternatives
    (other than maybe convolving to a lower, but uniform, resolution).
    The lstsq extraction can include all errors and covariances.
    Errors are included (the diagonal of the covariance matrix), but
    the full covariance matrix is currently discarded.

    """

    ################################################################
    # Initiate the header with critical data about the observation.
    # Then add basic information about the calibration data used to
    # extract a cube.
    ################################################################

    try:
        version = charis.__version__
    except:
        version = None

    header = fits.open(filename)[0].header
    instrument, _, _ = instruments.instrument_from_data(header, interactive=False)

    if instrument.instrument_name == 'CHARIS':
        header = utr.metadata(filename, version=version)

    try:
        calhead = fits.open(os.path.join(calibdir, 'cal_params.fits'))[0].header
        header.append(('comment', ''), end=True)
        header.append(('comment', '*' * 60), end=True)
        header.append(('comment', '*' * 21 + ' Calibration Data ' + '*' * 21), end=True)
        header.append(('comment', '*' * 60), end=True)
        header.append(('comment', ''), end=True)
        for key in calhead:
            header.append((key, calhead[key], calhead.comments[key]), end=True)
    except:
        log.warn('Unable to append calibration parameters to FITS header.')

    ################################################################
    # Read in file and return an instance of the Image class with the
    # up-the-ramp combination of reads.  Subtract the thermal
    # background and apply a bad pixel mask.
    ################################################################

    maskarr = None
    if mask is True:
        maskarr = fits.open(os.path.join(calibdir, 'mask.fits'))[0].data

    if instrument.instrument_name == 'CHARIS':
        inImage = utr.calcramp(filename=filename, mask=maskarr, read_idx=read_idx,
                               header=header, gain=gain, noisefac=noisefac,
                               maxcpus=maxcpus)

    elif instrument.instrument_name == 'SPHERE':
        data = fits.getdata(filename)
        if len(data.shape) == 3:
            data = np.mean(data.astype('float64'), axis=0) * maskarr
        inImage = Image(data=data, ivar=maskarr.astype('float64'),
                        instrument_name=instrument.instrument_name)

    if bgsub:
        try:
            if bgpath is None:
                hdulist = fits.open(os.path.join(calibdir, 'background.fits'))
            else:
                hdulist = fits.open(bgpath)

            bg = hdulist[0].data
            if bg is None:
                bg = hdulist[1].data
            if mask:
                bg *= maskarr
            inImage.data -= bg
        except:
            bgsub = False
            log.warn('No valid background image found in ' + calibdir)

    header['bgsub'] = (bgsub, 'Subtract background count rate from a dark?')
    if saveramp:
        inImage.write(re.sub('.*/', '', re.sub('.fits', '_ramp.fits', filename)))

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
        lensletflat = fits.open(
            os.path.join(calibdir, 'lensletflat.fits'))[0].data.astype('float64')
    else:
        lensletflat = None
    header['flatfld'] = (flatfield, 'Flatfield the detector and lenslet array?')

    datacube = None

    if method == 'lstsq' or suppressrn:
        try:
            keyfile = fits.open(os.path.join(calibdir, 'polychromekeyR%d.fits' % (R)))
            R2 = R
        except IOError:
            keyfilenames = glob.glob(calibdir + '*polychromekeyR*.fits')
            if len(keyfilenames) == 0:
                raise IOError("No key file found in " + calibdir)
            R2 = int(re.sub('.*keyR', '', re.sub('.fits', '', keyfilenames[0])))
            keyfile = fits.open(os.path.join(calibdir, 'polychromekeyR%d.fits' % (R2)))
            if verbose:
                print("Warning: calibration files not found at requested resolution of R = %d" % (R))
                print("Found files at R = %d, using these instead." % (R2))

        if fitshift:
            try:
                psflets = np.load(os.path.join(calibdir, 'polychromefullR%d.npy' % (R2)))
                offsets = np.arange(-5, 6)
                psflets = primitives.calc_offset(psflets, inImage, offsets, maxcpus=maxcpus)
            except:
                if verbose:
                    print('Fit shift failed. Continuing without fitting shift.')
                fitshift = False
        if not fitshift:
            psflets = fits.open(os.path.join(calibdir, 'polychromeR%d.fits' % (R2)))[0].data

        header.append(('fitshift', fitshift, 'Fit a subpixel shift in PSFlet locations?'), end=True)
        lam_midpts = keyfile[0].data
        x = keyfile[1].data
        y = keyfile[2].data
        good = keyfile[3].data

        if flatfield:
            psflets = psflets * fits.open(os.path.join(calibdir, 'pixelflat.fits'))[0].data

        ############################################################
        # Do an initial least-squares fit to remove correlated read
        # noise if method = optext and suppressrn = True
        ############################################################

        if method != 'lstsq':
            corrnoise = primitives.fit_spectra(
                inImage, psflets, lam_midpts, x, y, good,
                header=inImage.header, flat=lensletflat, refine=refine,
                suppressrdnse=suppressrn, smoothandmask=smoothandmask,
                minpct=minpct, fitbkgnd=fitbkgnd, maxcpus=maxcpus, return_corrnoise=True)
            inImage.data -= corrnoise
        else:
            result = primitives.fit_spectra(
                inImage, psflets, lam_midpts, x, y, good,
                header=inImage.header, flat=lensletflat, refine=refine,
                suppressrdnse=suppressrn, smoothandmask=smoothandmask,
                minpct=minpct, fitbkgnd=fitbkgnd, returnresid=saveresid,
                maxcpus=maxcpus)
            if saveresid:
                datacube, resid = result
                resid.write(re.sub('.*/', '', re.sub('.fits', '_resid.fits', filename)))
            else:
                datacube = result

    if method == 'optext' or method == 'apphot3' or method == 'apphot5':
        loc = primitives.PSFLets(load=True, infiledir=calibdir)
        try:
            lam_midpts = fits.open(os.path.join(calibdir, '/polychromekeyR%d.fits' % (R)))[0].data
            R2 = R
        except IOError:
            keyfilenames = glob.glob(calibdir + '*polychromekeyR*.fits')
            if len(keyfilenames) == 0:
                raise IOError("No key file found in " + calibdir)
            R2 = int(re.sub('.*keyR', '', re.sub('.fits', '', keyfilenames[0])))
            lam = fits.open(keyfilenames[0])[0].data
            lam1, lam2 = [lam[0], lam[-1]]
            n = int(np.log(lam2 * 1. / lam1) * R2 + 1.5)
            lam_midpts = np.exp(np.linspace(np.log(lam1), np.log(lam2), n))
            if verbose:
                print("Warning: calibration files not found at requested resolution of R = %d" % (R))
                print("Found files at R = %d, using these instead." % (R2))

        try:
            sig = fits.open(os.path.join(calibdir, 'PSFwidths.fits'))[0].data
        except IOError:
            sig = 0.7

        if method == 'apphot3' or method == 'apphot5':
            sig = 1e10
            if method == 'apphot3':
                delt_x = 3
            else:
                delt_x = 5
        else:
            delt_x = 7

        if flatfield:
            pixelflat = fits.open(os.path.join(calibdir, 'pixelflat.fits'))[0].data
            inImage.data /= pixelflat + 1e-20
            inImage.ivar *= pixelflat**2

        datacube = primitives.optext_spectra(inImage, loc, lam_midpts, delt_x=delt_x,
                                             sig=sig, header=inImage.header, flat=lensletflat, maxcpus=maxcpus)

    if datacube is None:
        raise ValueError("Datacube extraction method " + method + " not implemented.")

    ################################################################
    # Add the original header for reference as the last HDU
    ################################################################

    datacube.extrahead = fits.open(filename)[0].header

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

    if instrument.instrument_name == 'SPHERE' and resample == True:
        clip_info_file = os.path.join(os.path.split(instrument.calibration_path)
                                      [0], 'hexagon_mapping_calibration.pickle')
        clip_infos = pickle.load(open(clip_info_file, "rb"))
        datacube_resampled = copy.copy(datacube)
        datacube_resampled.data = resample_image_cube(datacube.data, clip_infos, hexagon_size=1 / np.sqrt(3))

        datacube.write(re.sub('.fits', '_cube.fits', re.sub('.*/', '', filename)))
        datacube_resampled.write(re.sub('.fits', '_cube_resampled.fits', re.sub('.*/', '', filename)))
        return datacube, datacube_resampled

    else:
        datacube.write(re.sub('.fits', '_cube.fits', re.sub('.*/', '', filename)))
        return datacube
