#!/usr/bin/env python 

import numpy as np
from utr import getreads, utr
from astropy.io import fits

#datadir = '/Users/protostar/Dropbox/data/charis/lab/'
datadir = '../../../../data/charis/lab/'
signal_fn = 'CRSA00007219.fits'
bckgrd_fn = 'CRSA00007220.fits'

read_idx = [2,50]
signal_reads = getreads(datadir, signal_fn, read_idx, biassub='top')
bckgrd_reads = getreads(datadir, bckgrd_fn, read_idx, biassub='top')

# generate cds image 
#hdulist = fits.open(datadir+signal_fn)
#signal_cds = hdulist[-1].data - hdulist[1].data
#hdulist = fits.open(datadir+bckgrd_fn)
#bckgrd_cds = hdulist[-1].data - hdulist[1].data
#cds = signal_cds - bckgrd_cds
#fits.writeto('test_cds.fits', cds, clobber=True)

signal = utr(signal_reads)
signal.write('test_utr_signal.fits')
bckgrd = utr(bckgrd_reads)
bckgrd.write('test_utr_bckgrd.fits')
signal.data = signal.data - bckgrd.data

var = 1.0/signal.ivar + 1.0/bckgrd.ivar
signal.ivar = 1.0/var
signal.write('test_utr.fits')
