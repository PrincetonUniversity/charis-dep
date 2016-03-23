#!/usr/bin/env python 

from astropy.io import fits
import numpy as np
import utr
import primitives
from image import Image

# lam_min = 1150

lam = 1170
order = 3

initcoef = np.loadtxt('spotcoef/coef_'+str(lam-20)+'.dat')

fn = 'spotimages/monochrom_'+str(lam)+'_cds.fits'

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

np.savetxt('test_coef_'+str(lam)+'_O'+str(order)+'.dat', coef)
np.savetxt('test_distortion_'+str(lam)+'_O'+str(order)+'.dat', outarr, fmt="%.7g")
