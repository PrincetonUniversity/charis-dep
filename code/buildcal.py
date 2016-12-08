#!/usr/bin/env python

import glob
import re
import os
import time
import numpy as np
from image import Image
import primitives
import tools
from astropy.io import fits
import multiprocessing
from parallel import Task, Consumer
import copy
import utr
import shutil
import sys

log = tools.getLogger('main')


def buildcalibrations(inImage, inLam, mask, indir, outdir="./",
                      order=3, lam1=1150, lam2=2400, R=25, trans=None,
                      upsample=True, ncpus=multiprocessing.cpu_count()):
    """
    Build the calibration files needed to extract data cubes from
    sequences of CHARIS reads.

    Inputs:
    1. inImage:  Image class, should include count rate and ivar for 
                 a narrow-band flatfield calibration image.
    2. inLam:    wavelength in nm of inImage
    3. mask:     bad pixel mask, =0 for bad pixels
    4. indir:    directory where master calibration files live

    Optional inputs:
    1. outdir:   directory in which to place
    1. order:    int, order of polynomial fit to position(lambda).
                 Default 3 (strongly recommended).
    2. lam1:     minimum wavelength (in nm) of the bandpass
                 Default 1150
    3. lam2:     maximum wavelength (in nm) of the bandpass
                 Default 2400
    4. R:        spectral resolution of the PSFlets templates  
                 Default 25
    5. trans:    ndarray, trans[:, 0] = wavelength in nm, trans[:, 1]
                 is fractional transmission through the filter and 
                 atmosphere.  Default None --> trans[:, 1] = 1
    6. ncpus:    number of threads for multithreading.  
                 Default multiprocessing.cpu_count()

    Returns None, writes calibration files to outdir.
    
    """

    tstart = time.time()

    #################################################################
    # Fit the PSFlet positions on the input image, compute the shift
    # in the mean position (coefficients 0 and 10) and in the linear
    # component of the fit (coefficients 1, 4, 11, and 14).  The
    # comparison point is the location solution for this wavelength in
    # the existing calibration files.
    #################################################################

    log.info("Loading wavelength solution from " + indir + "lamsol.dat")
    lam = np.loadtxt(indir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(indir + "lamsol.dat")[:, 1:]
    psftool = primitives.PSFLets()
    oldcoef = psftool.monochrome_coef(inLam, lam, allcoef, order=order)

    x, y, good, newcoef = primitives.locatePSFlets(inImage, polyorder=order, coef=oldcoef)
            
    psftool.geninterparray(lam, allcoef, order=order)
    dcoef = newcoef - oldcoef

    indx = np.asarray([0, 1, 4, 10, 11, 14])
    psftool.interp_arr[0][indx] += dcoef[indx]

    #################################################################
    # Load the high-resolution PSFlet images and associated
    # wavelengths.
    #################################################################

    hires_list = np.sort(glob.glob(indir + '/hires_psflets_lam????.fits'))
    hires_arrs = [fits.open(filename)[0].data for filename in hires_list]
    lam_hires = [int(re.sub('.*lam', '', re.sub('.fits', '', filename)))
                 for filename in hires_list]

    #################################################################
    # Wavelengths at which to return the PSFlet templates
    #################################################################

    Nspec = int(np.log(lam2*1./lam1)*R + 1.5)
    loglam_endpts = np.linspace(np.log(lam1), np.log(lam2), Nspec)
    loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1])/2
    lam_endpts = np.exp(loglam_endpts)
    lam_midpts = np.exp(loglam_midpts)

    #################################################################
    # Compute the PSFlets integrated over small ranges in wavelength,
    # accounting for atmospheric+filter transmission.  Do this
    # calculation in parallel.
    #################################################################

    xindx = np.arange(-100, 101)
    xindx, yindx = np.meshgrid(xindx, xindx)

    #################################################################
    # Oversampling in x in final calibration frame.  If >1, fitting a
    # subpixel shift is possible in cube extraction.
    #################################################################

    if upsample:
        upsamp = 5
    else:
        upsamp = 1
    psflet_res = 9 # Oversampling of high-resolution PSFlet images
    nlam = 10      # Number of monochromatic PSFlets per integrated PSFlet

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    consumers = [ Consumer(tasks, results)
                  for i in range(ncpus) ]
    for w in consumers:
        w.start()

    for i in range(upsamp*(Nspec - 1)):
        ilam = i//upsamp
        dx = (i%upsamp)*1./upsamp
        tool = copy.deepcopy(psftool)
        tool.interp_arr[0, 0] -= dx
        tasks.put(Task(i, primitives.make_polychrome,
                       (lam_endpts[ilam], lam_endpts[ilam + 1], hires_arrs,
                        lam_hires, tool, allcoef, xindx, yindx, 
                        psflet_res, nlam, trans)))
    for i in range(ncpus):
        tasks.put(None)
                      
    polyimage = np.empty((Nspec - 1, 2048, 2048*upsamp), np.float32)

    for i in range(upsamp*(Nspec - 1)):
        index, result = results.get()
        ilam = index//upsamp
        dx = (index%upsamp)
        polyimage[ilam, :, dx::upsamp] = result

    #################################################################
    # Save the positions of the PSFlet centers to cut out the
    # appropriate regions in the least-squares extraction
    #################################################################

    xpos = []
    ypos = []
    good = []
    for i in range(Nspec - 1):
        _x, _y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
        _good = (_x > 8)*(_x < 2040)*(_y > 8)*(_y < 2040)
        xpos += [_x]
        ypos += [_y]
        good += [_good]

    if upsamp > 1:
        np.save(outdir + 'polychromefullR%d.npy' % (R), polyimage)
        
    out = fits.HDUList(fits.PrimaryHDU(polyimage[:, :, ::upsamp].astype(np.float32)))
    out.writeto(outdir + 'polychromeR%d.fits' % (R), clobber=True)

    outkey = fits.HDUList(fits.PrimaryHDU(lam_midpts))
    outkey.append(fits.PrimaryHDU(np.asarray(xpos)))
    outkey.append(fits.PrimaryHDU(np.asarray(ypos)))
    outkey.append(fits.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(outdir + 'polychromekeyR%d.fits' % (R), clobber=True)
    
    print "Total time elapsed: %.0f" % (time.time() - tstart)
    return None
    


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print "Must call buildcal with at least three arguments:"
        print "  1: The path to the narrow-band flatfield image"
        print "  2: The wavelength, in nm, of the narrow-band image"
        print "  3: The band/filter: 'J', 'H', 'K', or 'lowres'"
        print "Example: buildcal CRSA00000000.fits 1550 lowres"
        print "Optional additional arguments: filenames of darks"
        print "  taken with the same observing setup."
        print "Example: buildcal CRSA00000000.fits 1550 lowres darks/CRSA*.fits"
        exit()

    infile = sys.argv[1]
    lam = int(sys.argv[2])
    band = sys.argv[3]
    bgfiles = []
    bgimages = []
    for i in range(4, len(sys.argv)):
        bgfiles += glob.glob(sys.argv[i])        

    print "\n" + "*"*60
    print "Oversample PSFlet templates to enable fitting a subpixel offset in cube"
    print "extraction?  Cost is a factor of ~2-4 in the time to build calibrations."
    print "*"*60
    while True:
        upsample = raw_input("     Oversample? [Y/n]: ")
        if upsample in ['', 'y', 'Y']:
            upsample = True
            break
        elif upsample in ['n', 'N']:
            upsample = False
            break
        else:
            print "Invalid input."

    print "\n" + "*"*60
    print "Building calibration files, placing results in current directory:"
    print os.path.abspath('.')
    print "\nSettings:\n"
    print "Narrow-band flatfield image: " + infile
    print "Wavelength:", lam, "nm"
    print "Observing mode: " + band
    print "Upsample PSFlet templates? ", upsample
    if len(bgfiles) > 0:
        print "Background count rates will be computed."
    else:
        print "No background will be computed."
    print "*"*60
    while True:
        do_calib = raw_input("     Continue with these settings? [Y/n]: ")
        if do_calib in ['', 'y', 'Y']:
            break
        elif do_calib in ['n', 'N']:
            exit()
        else:
            print "Invalid input."
   
    ###############################################################
    # Wavelength limits in nm
    ###############################################################

    if band == 'J':   
        lam1, lam2 = [1155, 1340]
    elif band == 'H':
        lam1, lam2 = [1470, 1800]
    elif band == 'K':
        lam1, lam2 = [2005, 2380]
    elif band == 'lowres':
        lam1, lam2 = [1140, 2410]
    else:
        raise ValueError('Band must be one of: J, H, K, lowres')

    if lam < lam1 or lam > lam2:
        raise ValueError("Error: wavelength " + str(lam) + " outside range (" + str(lam1) + ", " + str(lam2) + ") of mode " + band)


    prefix = os.path.dirname(os.path.realpath(__file__))

    ###############################################################
    # Spectral resolutions for the final calibration files
    ###############################################################

    if band in ['J', 'H', 'K']:
        indir = prefix + "/calibrations/highres_" + band + "/"
        R = 100
    else:
        indir = prefix + "/calibrations/lowres/"
        R = 30

    mask = fits.open(indir + 'mask.fits')[0].data

    ###############################################################
    # Mean background count rate, weighted by inverse variance
    ###############################################################

    num = 0
    denom = 1e-100
    for filename in bgfiles:
        bg = utr.calcramp(filename=filename, mask=mask)
        num = num + bg.data*bg.ivar
        denom = denom + bg.ivar
    if len(bgfiles) > 0:
        background = Image(data=num/denom, ivar=1./denom)
        background.write('background.fits')
        
    ###############################################################
    # Monochromatic flatfield image
    ###############################################################

    #inImage = utr.calcramp(filename=infile, mask=mask)
    inImage = Image(filename=infile)

    trans = np.loadtxt(indir + band + '_tottrans.dat')

    buildcalibrations(inImage, lam, mask, indir, lam1=lam1, lam2=lam2,
                      upsample=upsample, R=R, order=3, trans=trans)

    for filename in ['mask.fits', 'lensletflat.fits', 'pixelflat.fits']:
        shutil.copy(indir + filename, './' + filename)
