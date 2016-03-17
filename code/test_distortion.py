#!/usr/bin/env python 

from astropy.io import fits
import numpy as np
import utr
import primitives
from image import Image

make_image = False
fn = 'monochrom_1510.fits'

if make_image:
    datadir = '../../../../data/charis/lab/'
    monochrom_fn = 'CRSA00007408.fits'
    laser_off_fn = 'CRSA00007387.fits'
    monochrom = utr.utr(datadir=datadir, filename=monochrom_fn, biassub='top')
    laser_off = utr.utr(datadir=datadir, filename=laser_off_fn, read_idx=[1,21], biassub='top')
    monochrom.data = monochrom.data - laser_off.data
    monochrom.filename = fn
    monochrom.write(monochrom.filename)
else:
    monochrom = Image(filename=fn)

monochrom.ivar = np.ones(monochrom.data.shape)

ydim, xdim = monochrom.data.shape
x = np.arange(-(ydim//20), ydim//20+ 1)
x, y = np.meshgrid(x, x)

order = 1
_x, _y, good, coef = primitives.locatePSFlets(monochrom, polyorder=order)
_x = np.reshape(_x, -1)
_y = np.reshape(_y, -1)
good = np.reshape(good, -1)

outarr = np.zeros((_x.shape[0], 5))
outarr[:, 0] = _x
outarr[:, 1] = _y
outarr[:, 2] = good
outarr[:, 3] = x.flatten() 
outarr[:, 4] = y.flatten()

coef = np.asarray(coef)

np.savetxt('test_coef_O'+str(order)+'.dat', coef)
np.savetxt('test_distortion_O'+str(order)+'.dat', outarr, fmt="%.7g")
