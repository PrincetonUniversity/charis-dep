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
import os
import ConfigParser
import multiprocessing

def getcube(filename, read_idx=[2, None], biassub=None, phnoise=1.3, 
            calibdir='calibrations/20160408/', bgsub=True, mask=True,
            maxcpus=None, R=25, method='lstsq', refine=True,
            smoothandmask=True):

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
    3. R:        integer, approximate resolution lam/delta(lam) of the
                 extracted data cube.  Resolutions higher than ~25 or 30
                 are comparable to or finer than the pixel sampling and 
                 are not recommended--there is very strong covariance.
    4. method:   string, method used to extract data cube.  Should be 
                 either 'lstsq' for a least-squares extraction or 
                 'optext' for a quasi-optimal extraction.
    Returns:
    1. datacube: an instance of the Image class containing the data cube.

    Steps performed: 

    1. Up-the-ramp combination.  As of yet no special read noise
    suppression (just a channel-by-channel correction of the reference
    voltages), no nonlinearity correction.
    2. Subtraction of thermal background from file in calibration 
    directory.
    3. Application of hot pixel mask (as zero inverse variances).
    4. Load calibration files from calibration directory for the data
    cube extraction.
    5. Extract the data cube.

    Notes for now: the quasi-optimal extraction isn't really an
    optimal extraction.  Every lenslet's spectrum natively samples a
    different set of wavelengths, so right now they are all simply
    interpolated onto the same wavelength array.  This is certainly
    not optimal, but I don't see any particularly good alternatives
    (other than maybe convolving to a lower, but uniform, resolution).
    The lstsq extraction can include all errors and covariances, and
    soon will.

    """
    
    ################################################################
    # Read in file and return an instance of the Image class with the
    # up-the-ramp combination of reads.  Subtract the thermal
    # background and apply a bad pixel mask.
    ################################################################

    inImage = utr.utr(filename=filename, read_idx=read_idx, 
                      biassub=biassub, phnoise=phnoise)
    if bgsub:
        inImage.data -= fits.open(calibdir + '/background.fits')[0].data
    if mask:
        inImage.ivar *= fits.open(calibdir + '/mask.fits')[0].data

    ################################################################
    # Read in necessary calibration files and extract the data cube.
    ################################################################

    if method == 'lstsq':
        psflets = fits.open(calibdir + '/polychromeR%d.fits' % (R))[0].data
        keyfile = fits.open(calibdir + '/polychromekeyR%d.fits' % (R))
        lam_midpts = keyfile[0].data
        x = keyfile[1].data
        y = keyfile[2].data
        good = keyfile[3].data
        datacube = primitives.fit_spectra(inImage, psflets, lam_midpts, x, y, good, header=inImage.header, refine=refine, smoothandmask=smoothandmask, maxcpus=maxcpus)

    elif method == 'optext':
        loc = primitives.PSFLets(load=True, infiledir=calibdir)
        lam_midpts = fits.open(calibdir + '/polychromekeyR%d.fits' % (R))[0].data
        datacube = primitives.fitspec_intpix(inImage, loc, lam_midpts, header=inImage.header)
    
    else:
        raise ValueError("Datacube extraction method " + method + " not implemented.")
    return datacube

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Must call extractcube.py with two arguments:"
        print "1: a string parsed by glob matching files to be turned into data cubes"
        print "2: a .ini configuration file processed by ConfigParser"
        exit()

    filenames = []
    for i in range(1, len(sys.argv) - 1):
        filenames += glob.glob(sys.argv[i])
    Config = ConfigParser.ConfigParser()
    Config.read(sys.argv[len(sys.argv) - 1])

    read_0 = Config.getint('Ramp', 'read_0')
    try:
        read_1 = Config.getint('Ramp', 'read_f')
    except:
        read_1 = None
    read_idx = [read_0, read_1]
    biassub = Config.get('Ramp', 'biassub')
    if biassub == 'None':
        biassub = None
    try:
        phnoise = Config.getfloat('Ramp', 'phnoise')
        if phnoise < 0:
            phnoise = 0
    except:
        phnoise = 1.3  # approximate factor for photon  noise (!= 1 as
                      # the  asymptotic  ratio  of  up-the-ramp  noise
                      # weighting to CDS weighting)

    bgsub = Config.getboolean('Calib', 'bgsub')
    mask = Config.getboolean('Calib', 'mask')

    calibdir = Config.get('Calib', 'calibdir')
    R = Config.getint('Extract', 'R')
    method = Config.get('Extract', 'method')
    refine = Config.getboolean('Extract', 'refine')

    try:
        maxcpus = Config.getint('Extract', 'maxcpus')
        if maxcpus <= 0:
            maxcpus = multiprocessing.cpu_count() + maxcpus
        if maxcpus < 1:
            maxcpus = 1
    except:
        maxcpus = None

    try:
        smoothandmask = Config.getboolean('Extract', 'smoothandmask')
    except:
        smoothandmask = True

    for filename in filenames:
        cube = getcube(filename=filename, read_idx=read_idx, bgsub=bgsub,
                       mask=mask, biassub=biassub, phnoise=phnoise,
                       refine=refine, maxcpus=maxcpus, calibdir=calibdir,
                       R=R, method=method, smoothandmask=smoothandmask)
        cube.write(re.sub('.fits', '_cube.fits', re.sub('.*/', '', filename)))

