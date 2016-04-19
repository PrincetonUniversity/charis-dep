#!/usr/bin/env python 

from astropy.io import fits
import numpy as np
import utr
import primitives
from image import Image
import cProfile
import time

# lam_min = 1150
def fit_lam(lam, datadir='/scratch/tbrandt/CHARIS_reads/wavelength_stability/', initcoef=None):
    
    #lam = 1150
    order = 3
    
    if False: #initcoef is None:
        initcoef = np.loadtxt('test_coef3_' + str(lam) + '_O3.dat')

        #initcoef = np.loadtxt('spotcoef/coef_1910.dat')
    #except:
    #    initcoef = None

#fn = 'spotimages/monochrom_'+str(lam)+'_cds.fits'
    fn = datadir + 'monochrom_' + str(lam) + '_bgsub.fits'
    
    monochrom = Image(filename=fn)
    monochrom.ivar = np.ones(monochrom.data.shape)
    
    _x, _y, good, coef = primitives.locatePSFlets(monochrom, polyorder=order, coef=initcoef)
    _x = np.reshape(_x, -1)
    _y = np.reshape(_y, -1)
    good = np.reshape(good, -1)
    
    outarr = np.zeros((_x.shape[0], 3))
    outarr[:, 0] = _x
    outarr[:, 1] = _y
    outarr[:, 2] = good
    
    coef = np.asarray(coef)

    np.savetxt('test_coef4_'+str(lam)+'_O'+str(order)+'.dat', coef)
    np.savetxt('test_distortion4_'+str(lam)+'_O'+str(order)+'.dat', outarr, fmt="%.7g")
    
    return coef


if __name__ == "__main__":
    #cProfile.run('fit_lam()', 'locate_psflets_stats')
    t0 = time.time()
    coef = None

    #loc = primitives.PSFLets(lam1=1150/1.04, lam2=2400*1.03)
    loc = primitives.PSFLets(load=True)
    #loc = primitives.PSFLets(load=False)

    #loc.genpixsol(lam1=1150/1.04, lam2=1850*1.04)
    #loc.savepixsol()

    #lam = np.arange(1150, 2405, 50)
    #for i in range(100):
    #    print i
    #    for j in range(100):
    #        xloc, yloc = loc.lamtopix(lam, 50, 50)
    #loc.pixinterp(1150/1.04, 2400*1.03, x, y)

    #x, y, lam = loc.genpixsol()
    #if True:
    #    loc.genpixsol(lam1=1150/1.04, lam2=2400*1.03)
    #    loc.savepixsol()
        #out = fits.HDUList(fits.PrimaryHDU(loc.xindx))
        #out.append(fits.PrimaryHDU(loc.yindx))
        #out.append(fits.PrimaryHDU(loc.lam_indx))
        #out.append(fits.PrimaryHDU(loc.nlam))
        #out.writeto('foo2.fits', clobber=True)


    
    t = time.time()

    #inImage = Image('/scratch/tbrandt/CHARIS_reads/CRSA00007386_2D.fits')
    inImage = Image('/scratch/tbrandt/CHARIS_reads/wavelength_stability/CRSA00008191_2D_bgsub.fits')
    #bkgnd = Image('/scratch/tbrandt/CHARIS_reads/CRSA00007387_2D.fits')
    #inImage.data -= bkgnd.data

    #loc.loadpixsol()

    datacube = primitives.fitspec_intpix(inImage, loc)
    #datacube = primitives.fitspec_intpix(inImage, x, y, lam)
    print '%.2f' % (time.time() - t)

    datacube.write('testcube4.fits')

    
    #arr = np.zeros(x.shape)
    #for i in range(arr.shape[0]):
    #    #print i
    #    for j in range(arr.shape[1]):
    #        x, y, lam = loc.getarr(i, j)
    #        arr[i, j] = x.shape[0]

    #out = fits.HDUList(fits.PrimaryHDU(arr))
    #out.writeto('foo.fits', clobber=True)

    #x, y, lam = loc.getarr(100, 100)
    #for i in range(x.shape[0]):
    #    print x[i], y[i], lam[i]
           
    #for lam in range(1150, 2405, 5): 
    #    xloc, yloc = loc.lamtopix(lam, x, y)
    #    print lam #, xloc, yloc
        #coefs = loc.getallpix(lam, x, y)
        #print coefs[0]
        #print lam, xloc[5, 5], yloc[5, 5], xloc[5, -5], yloc[5, -5], 
        #print xloc[-5, -5], yloc[-5, -5], xloc[-5, 5], yloc[-5, 5]

        

        #xloc = np.reshape(x, -1)
        #yloc = np.reshape(y, -1)
        
    #coef = np.loadtxt('test_coef4_1840_O3.dat')
    #for i in range(1150, 1855, 30):
    #    coef = fit_lam(i, initcoef=coef)
    #coef = fit_lam(1850, initcoef=coef)
    #    t1 = time.time()
    #    print '%.2f' % (t1 - t0)
    #    t0 = t1
