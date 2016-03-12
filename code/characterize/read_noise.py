#!/usr/bin/env python 

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import norm

#######################
# script parameters
#######################
sig_clip = 8 
ramp_idx = [2, None]
fn = 'CRSA00007220.fits'
refchan = False
datadir = '/Users/protostar/Dropbox/data/charis/lab/'
#######################

hdulist = fits.open(datadir+fn)
ramps = hdulist[ramp_idx[0]:ramp_idx[1]]

reads = []
for r in ramps:
    start_col = 64 if refchan else 0
    refpix = np.concatenate([r.data[4:-4,start_col:start_col+4].flatten(), r.data[4:-4,start_col+2044:].flatten(),\
                             r.data[:4,start_col:].flatten(), r.data[-4:,start_col:].flatten()])
    reads.append(refpix)
reads = np.array(reads)
reads = reads - np.mean(reads, axis=0)
mean_sig = np.mean(np.std(reads, axis=0, ddof=1))
reads = reads[np.abs(reads) < sig_clip*mean_sig]

f, a = plt.subplots(1,1)
a.hist(reads.flatten(), bins=100, histtype='step', normed=True)
mu, sigma = norm.fit(reads.flatten())
x = np.linspace(-sigma*sig_clip, sigma*sig_clip, 100)
pdf_fitted = norm.pdf(x, loc=mu, scale=sigma)
a.plot(x, pdf_fitted,'r--', lw=2.)
a.text(0.2, 0.7, r'$\sigma = '+str(round(sigma,2))+'$', transform=a.transAxes, fontsize=18, color='r')
a.minorticks_on()
a.set_xlabel(r'$(C_\mathit{i}\,-\, \mu_\mathit{i})_{refpix}\ [ADU]$')

import RaiseWindow
plt.show()
