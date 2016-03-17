#!/usr/bin/env python

try:
    from astropy.io import fits
except:
    import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt

def getreads(filename, meansub=False):
    hdulist = fits.open(filename)
    reads = np.array([h.data[4:-4,64+4:-4] for h in hdulist[1:30]])
    return reads

def calcvar(reads):
    diff = np.zeros((reads.shape[0]-1, reads.shape[1], reads.shape[2]))
    for i in range(reads.shape[0]-1):
        diff[i,:,:] = reads[i+1,:,:] - reads[i,:,:]
    var = np.var(diff, axis=0)
    fits.writeto('test_var.fits', var, clobber=True)
    return var

def calcgain(filename):
    reads = getreads(filename)
    count = (reads[-1] - reads[0])/(reads.shape[0]-1)
    fits.writeto('test_count.fits', count, clobber=True)
    count = count.flatten()
    var = calcvar(reads).flatten()
    #clip = (count>50) & (count<200) & (var<40000)
    #count = count[clip]
    #var = var[clip]
    x = np.array([np.ones(count.size), count]).T
    beta = np.dot(np.linalg.inv(np.dot(x.T,x)), np.dot(x.T, var))
    print beta[0], beta[1]
    testplots(var, count)

def testplots(var, count):
    f1, a1 = plt.subplots(1,1)
    a1.scatter(count, var, marker='o', alpha=0.1)
    import RaiseWindow
    plt.show()

if __name__=='__main__':
    fn = 'CRSA00007220.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    calcgain(datadir+fn)
