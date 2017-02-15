#!/usr/bin/env python

#########################################################################
# A provisional routine for actually producing and returning data cubes.
#########################################################################

import numpy as np
import time
import glob
import re
from astropy.io import fits
import primitives
import utr
from image import Image
import sys
import ConfigParser
import multiprocessing
import logging

log = logging.getLogger('main')

def getcube(filename, read_idx=[1, None], calibdir='calibrations/20160408/', 
            bgsub=True, mask=True, gain=2, noisefac=0, saveramp=False, R=30,
            method='lstsq', refine=True, suppressrn=True, fitshift=True, 
            flatfield=True, smoothandmask=True, saveresid=False,
            maxcpus=multiprocessing.cpu_count()):

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
    
    header = utr.metadata(filename)

    try:
        calhead = fits.open(calibdir + '/cal_params.fits')[0].header
        header.append(('comment', ''), end=True)
        header.append(('comment', '*'*60), end=True)
        header.append(('comment', '*'*21 + ' Calibration Data ' + '*'*21), end=True)
        header.append(('comment', '*'*60), end=True)    
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
    if mask == True:
        maskarr = fits.open(calibdir + '/mask.fits')[0].data  
    
    inImage = utr.calcramp(filename=filename, mask=maskarr,read_idx=read_idx, 
                           header=header, gain=gain, noisefac=noisefac, 
                           maxcpus=maxcpus)
        
    if bgsub:
        try:
            hdulist = fits.open(calibdir + '/background.fits')
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
    header.append(('comment', '*'*60), end=True)
    header.append(('comment', '*'*22 + ' Cube Extraction ' + '*'*21), end=True)
    header.append(('comment', '*'*60), end=True)    
    header.append(('comment', ''), end=True)

    if flatfield:
        lensletflat = fits.open(calibdir + '/lensletflat.fits')[0].data
    else:
        lensletflat = None
    header['flatfld'] = (flatfield, 'Flatfield the detector and lenslet array?')

    datacube = None

    if method == 'lstsq' or suppressrn:
        header.append(('fitshift', fitshift, 'Fit a subpixel shift in PSFlet locations?'), end=True)
        if fitshift:
            psflets = np.load(calibdir + '/polychromefullR%d.npy' % (R))
            offsets = np.arange(-5, 6)
            psflets = primitives.calc_offset(psflets, inImage, offsets, maxcpus=maxcpus)
        else:
            psflets = fits.open(calibdir + '/polychromeR%d.fits' % (R))[0].data
        keyfile = fits.open(calibdir + '/polychromekeyR%d.fits' % (R))
        lam_midpts = keyfile[0].data
        x = keyfile[1].data
        y = keyfile[2].data
        good = keyfile[3].data

        if flatfield:
            psflets = psflets*fits.open(calibdir + '/pixelflat.fits')[0].data

        ############################################################
        # Do an initial least-squares fit to remove correlated read
        # noise if method = optext and suppressrn = True
        ############################################################

        if method != 'lstsq':
            corrnoise = primitives.fit_spectra(inImage, psflets, lam_midpts, x, y, good, header=inImage.header, flat=lensletflat, refine=refine, suppressrdnse=suppressrn, smoothandmask=smoothandmask, maxcpus=maxcpus, return_corrnoise=True)
            inImage.data -= corrnoise
        else:
            result = primitives.fit_spectra(inImage, psflets, lam_midpts, x, y, good, header=inImage.header, flat=lensletflat, refine=refine, suppressrdnse=suppressrn, smoothandmask=smoothandmask, returnresid=saveresid, maxcpus=maxcpus)
            if saveresid:
                datacube, resid = result
                resid.write(re.sub('.*/', '', re.sub('.fits', '_resid.fits', filename)))
            else:
                datacube = result

    if method == 'optext':
        loc = primitives.PSFLets(load=True, infiledir=calibdir)
        lam_midpts = fits.open(calibdir + '/polychromekeyR%d.fits' % (R))[0].data
        datacube = primitives.optext_spectra(inImage, loc, lam_midpts, header=inImage.header, flat=lensletflat, maxcpus=maxcpus)

    if datacube is None:
        raise ValueError("Datacube extraction method " + method + " not implemented.")

    ################################################################
    # Add the original header for reference as the last HDU
    ################################################################

    datacube.extrahead = fits.open(filename)[0].header

    return datacube


if __name__ == "__main__":

    if len(sys.argv) < 3:
        errstring = "Must call extractcube.py with at least two arguments:\n"
        errstring += "1: string(s) parsed by glob matching files to be turned into data cubes\n"
        errstring += "2: a .ini configuration file processed by ConfigParser"
        try:
            print errstring
        except:
            print(errstring)
        exit()

    filenames = []
    for i in range(1, len(sys.argv) - 1):
        filenames += glob.glob(sys.argv[i])

    if len(filenames) == 0:
        raise ValueError("No matching CHARIS files found by extractcube.")

    Config = ConfigParser.ConfigParser()
    Config.read(sys.argv[len(sys.argv) - 1])

    read_0 = Config.getint('Ramp', 'read_0')
    try:
        read_1 = Config.getint('Ramp', 'read_f')
    except:
        read_1 = None
    read_idx = [read_0, read_1]
    try:
        gain = Config.getfloat('Ramp', 'gain')
    except:
        gain = 2
    try:
        noisefac = Config.getfloat('Ramp', 'noisefac')
    except:
        noisefac = 0

    saveramp = Config.getboolean('Ramp', 'saveramp')
    bgsub = Config.getboolean('Calib', 'bgsub')
    mask = Config.getboolean('Calib', 'mask')
    try:
        flatfield = Config.getboolean('Calib', 'flatfield')
    except:
        flatfield = True
    try:
        fitshift = Config.getboolean('Calib', 'fitshift')
    except:
        fitshift = True

    calibdir = Config.get('Calib', 'calibdir')
    R = Config.getint('Extract', 'R')
    method = Config.get('Extract', 'method')
    try:
        refine = Config.getboolean('Extract', 'refine')
    except:
        refine = True
    try:
        suppressrn = Config.getboolean('Extract', 'suppressrn')
    except:
        suppressrn = True
    try:
        saveresid = Config.getboolean('Extract', 'saveresid')
    except:
        saveresid = False

    ################################################################
    # Maximum threads must be between 1 and cpu_count, inclusive
    ################################################################

    try:
        maxcpus = Config.getint('Extract', 'maxcpus')
        if maxcpus <= 0:
            maxcpus = multiprocessing.cpu_count() + maxcpus
        maxcpus = min(maxcpus, multiprocessing.cpu_count())
        maxcpus = max(maxcpus, 1)
    except:
        maxcpus = multiprocessing.cpu_count()

    try:
        smoothandmask = Config.getboolean('Extract', 'smoothandmask')
    except:
        smoothandmask = True

    for filename in filenames:
        cube = getcube(filename=filename, read_idx=read_idx, bgsub=bgsub,
                       mask=mask, gain=gain, noisefac=noisefac, 
                       saveramp=saveramp, refine=refine, maxcpus=maxcpus, 
                       calibdir=calibdir, R=R, method=method, 
                       smoothandmask=smoothandmask, flatfield=flatfield,
                       fitshift=fitshift, suppressrn=suppressrn,
                       saveresid=saveresid)
        cube.write(re.sub('.fits', '_cube.fits', re.sub('.*/', '', filename)))
