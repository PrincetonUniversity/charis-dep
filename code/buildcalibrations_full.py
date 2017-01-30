#!/usr/bin/env python

import glob
import re
import os
import time
import numpy as np
from image import Image
import primitives
import tools
import unittest
from astropy.io import fits
from scipy import ndimage, stats
from parallel import Task, Consumer
import multiprocessing
import sys
import copy

log = tools.getLogger('main')


def buildcalibrations(filelist, lamlist, 
                      calibdir="calibrations/", outdir="./",
                      order=3, lam1=1150, lam2=2400, R=25,
                      genwavelengthsol=True, makehiresPSFlets=True,
                      savehiresimages=True):
    """
    """

    try: 
        os.makedirs(outdir)
    except OSError:
        if not os.path.isdir(outdir):
            raise

    log.info("Building calibration files, placing results in " + outdir)

    tstart = time.time()
    coef = None
    allcoef = []
    imlist = []
    for i, ifile in enumerate(filelist):
        im = Image(filename=ifile)
        imlist += [im]
        if genwavelengthsol:
            x, y, good, coef = primitives.locatePSFlets(im, polyorder=order, coef=coef, scale=15.28, phi=np.arctan2(1.973, -1))
            allcoef += [[lamlist[i]] + list(coef)]
    
    if genwavelengthsol:
        log.info("Saving wavelength solution to " + outdir + "lamsol.dat")
        allcoef = np.asarray(allcoef)
        np.savetxt(outdir + "lamsol.dat", allcoef)
        lam = allcoef[:, 0]
        allcoef = allcoef[:, 1:]
    else:
        log.info("Loading wavelength solution from " + outdir + "lamsol.dat")
        lam = np.loadtxt(outdir + "lamsol.dat")[:, 0]
        allcoef = np.loadtxt(outdir + "lamsol.dat")[:, 1:]

    log.info("Computing wavelength values at pixel centers")
    psftool = primitives.PSFLets()
    psftool.genpixsol(lam, allcoef, lam1=lam1/1.04, lam2=lam2*1.03)
    psftool.savepixsol(outdir=outdir)

    xindx = np.arange(-100, 101)
    xindx, yindx = np.meshgrid(xindx, xindx)
    n = allcoef.shape[0]//2
    xpos, ypos = psftool.return_locations_short(allcoef[n], xindx, yindx)
    xpos = np.reshape(xpos, -1)
    ypos = np.reshape(ypos, -1)
    xindx = np.reshape(xindx, -1)
    yindx = np.reshape(yindx, -1)
    good = (ypos > 4)*(ypos < 2044)*(xpos > 4)*(xpos < 2044)


    hires_arrs = []
    allxpos = []
    allypos = []

    for i in range(len(lamlist)):

        xpos, ypos = psftool.return_locations(lamlist[i], allcoef, xindx, yindx)

        xpos = np.reshape(xpos, -1)
        ypos = np.reshape(ypos, -1)
        allxpos += [xpos]
        allypos += [ypos]

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    ncpus = multiprocessing.cpu_count()
    consumers = [ Consumer(tasks, results)
                  for i in range(ncpus) ]
    for w in consumers:
        w.start()
        
    for i in range(len(lamlist)):
        tasks.put(Task(i, primitives.gethires, (allxpos[i], allypos[i],
                                                    imlist[i], 9)))
        
    for i in range(ncpus):
        tasks.put(None)
    for i in range(len(lamlist)):
        index, hiresarr = results.get()
        
        if savehiresimages:
            di, dj = hiresarr.shape[0], hiresarr.shape[2]
            outim = np.zeros((di*dj, di*dj))
            for ii in range(di):
                for jj in range(di):
                    outim[ii*dj:(ii + 1)*dj, jj*dj:(jj + 1)*dj] = hiresarr[ii, jj]
            out = fits.HDUList(fits.PrimaryHDU(hiresarr.astype(np.float32)))
            out.writeto(outdir + 'hires_psflets_lam%d.fits' % (lamlist[index]), clobber=True)

    Nspec = int(np.log(lam2*1./lam1)*R + 1.5)
    loglam_endpts = np.linspace(np.log(lam1), np.log(lam2), Nspec)
    loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1])/2
    lam_endpts = np.exp(loglam_endpts)
    lam_midpts = np.exp(loglam_midpts)
    polyimage = np.zeros((Nspec - 1, 2048, 2048))
    xpos = []
    ypos = []
    good = []

    for i in range(Nspec - 1):
        
        polyimage[i] = primitives.make_polychrome(lam_endpts[i], lam_endpts[i + 1],
                                                  hires_arrs, lam, psftool, 
                                                  allcoef, xindx, yindx)
        _x, _y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
        _good = (_x > 10)*(_x < 2038)*(_y > 10)*(_y < 2038)
        xpos += [_x]
        ypos += [_y]
        good += [_good]

    out = fits.HDUList(fits.PrimaryHDU(polyimage.astype(np.float32)))
    out.writeto(outdir + 'polychromeR%d.fits' % (R), clobber=True)

    outkey = fits.HDUList(fits.PrimaryHDU(lam_midpts))
    outkey.append(fits.PrimaryHDU(np.asarray(xpos)))
    outkey.append(fits.PrimaryHDU(np.asarray(ypos)))
    outkey.append(fits.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(outdir + 'polychromekeyR%d.fits' % (R), clobber=True)
    
    print "Total time elapsed: %.0f" % (time.time() - tstart)

if __name__ == "__main__":

    datadir = './'

    filelist = np.sort(glob.glob(datadir + "monochrome_*.fits"))
    lamlist = []
    for ifile in filelist:
        lam = int(re.sub(".*monochrome_", "", re.sub(".fits", "", ifile)))
        lamlist += [lam]

    #print filelist
    buildcalibrations(filelist, lamlist, calibdir="calibrations/", 
                      outdir="./", genwavelengthsol=True,
                      makehiresPSFlets=True, savehiresimages=True,
                      lam1=1150, lam2=2400, R=28, order=3)

