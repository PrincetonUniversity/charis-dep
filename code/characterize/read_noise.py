#!/usr/bin/env python 

import numpy as np
from astropy.io import fits

datadir = '/Users/protostar/Dropbox/data/charis/lab/2016-02-19/'

fn = 'CRSA00007122.fits'
hdulist = fits.open(datadir+fn)

refchan = False

ramps = hdulist[2:]
reads = []
reads_mean_sub = []

for r in ramps:
    start_col = 64 if refchan else 0
    refpix = np.concatenate([r.data[:4,start_col:], r.data[-4:,start_col:]])
    reads.append(refpix)
    for chan in range(32):
        refpix[:, 64*chan:64*(1 + chan)] -= refpix[:, 64*chan:64*(1 + chan)].mean()
    reads_mean_sub.append(refpix)
reads = np.array(reads)
reads_mean_sub = np.array(reads_mean_sub)
print np.mean(np.std(reads_mean_sub, axis=0, ddof=1))

reads = reads - np.mean(reads, axis=0)
print np.mean(np.std(reads, axis=0, ddof=1))
