#!/usr/bin/env python 

from astropy.io import fits
import numpy as np
import utr
import primitives

datadir = '../../../../data/charis/lab/'
monochrom_fn = 'CRSA00007408.fits'
laser_off_fn = 'CRSA00007387.fits'

monochrom = utr.utr(datadir=datadir, filename=monochrom_fn, biassub='top')
laser_off = utr.utr(datadir=datadir, filename=laser_off_fn, read_idx=[1,21], biassub='top')
monochrom.data = monochrom.data - laser_off.data
monochrom.filename = 'test_monochrom_1510.fits'
monochrom.write(monochrom.filename)

order = 2
_x, _y, good, coef = primitives.locatePSFlets(monochrom, polyorder=order)
_x = np.reshape(_x, -1)
_y = np.reshape(_y, -1)
good = np.reshape(good, -1)

outarr = np.zeros((_x.shape[0], 3))
outarr[:, 0] = _x
outarr[:, 1] = _y
outarr[:, 2] = good

coef = np.asarray(coef)

np.savetxt('test_coef.dat', coef)
np.savetxt('test_distortion.dat', outarr, fmt="%.5g")
