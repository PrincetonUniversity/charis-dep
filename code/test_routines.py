#!/usr/bin/env python

#########################################################################
# Note: this file contains a bunch of routines for real-time testing
# during development. 
# It is not designed to be clean or useful for anything else.
#########################################################################

import numpy as np
import primitives
import testimage
from image import Image

def _test_locate_psflets(norm=1000, rms=10, mode='bkgnd', order=2, 
                         prefix = '/home/tbrandt/CHARIS/Strehl80/'):
    
    """
    A test function this routine with read noise and Poisson noise
    added to a uniform background to try out the recovery of the 
    PSFlet locations.
    """
    
    lam = 1.150
    if mode == 'bkgnd':
        inImage = Image(prefix + 'bkgnd_%.3fum.fits' % (lam))
    elif mode == 'image':
        inImage = Image(prefix + 'image_%.3fum.fits' % (lam))
    else:
        raise ValueError("mode keyword must be 'bkgnd' or 'image'")
    inImage.data *= norm/np.amax(inImage.data)
    inImage.data = testimage.addnoise(inImage.data, darkrms=rms)
        
    _x, _y, good, coef = primitives.locatePSFlets(inImage, polyorder=order)

    _x = np.reshape(_x, -1)
    _y = np.reshape(_y, -1)
    good = np.reshape(good, -1)

    outarr = np.zeros((_x.shape[0], 3))
    outarr[:, 0] = _x
    outarr[:, 1] = _y
    outarr[:, 2] = good

    np.savetxt('test_norm%d_rms%d.dat' % (norm, rms), outarr, fmt="%.5g")


def test_specfit(prefix = '/home/tbrandt/CHARIS/Strehl80/', R=35, norm=1, order=2):
    
    inBkgnd = Image('testimage/summed_bkgnd_Strehl80.fits')
    inImage = Image('testimage/summed_image_Strehl80.fits')
    inImage.data += inBkgnd.data

    loglam = np.arange(np.log(1.15) + 1./(2*R), np.log(2.39), 1./R)
    lam = np.exp(loglam)

    psflets = [Image(prefix + 'bkgnd_%.3fum.fits' % (l)) for l in lam]
    x = []
    y = []
    good = []
    coef = None
    for psflet in psflets:
        psflet.data *= norm/np.amax(psflet.data)
        _x, _y, _good, coef = primitives.locatePSFlets(psflet, polyorder=order,
                                                       coef=coef)
        x += [_x]
        y += [_y]
        good += [_good]

    outarr = np.zeros((len(x)*2, np.prod(x[0].shape)))
    for i in range(len(x)):
        outarr[2*i] = np.reshape(x[i], -1)
        outarr[2*i + 1] = np.reshape(y[i], -1)

    np.savetxt('test.dat', outarr.T, fmt="%.5g")
    datacube = primitives.fit_spectra(inImage, psflets, x, y, good)

    outImage = Image(data=datacube)
    outImage.write('testcube.fits')

if __name__ == "__main__":
    #_test_locate_psflets(norm=5, rms=10)
    test_specfit(R=25, order=2)
