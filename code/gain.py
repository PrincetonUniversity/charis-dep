#!/usr/bin/env python

try:
    from astropy.io import fits
except:
    import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt

def getreads(filename):
    hdulist = fits.open(filename)
    reads = np.array([h.data[4:-4,64+4:-4] for h in hdulist[1:6]])
    return reads

def calcvar(reads):
    diff = np.zeros((reads.shape[0]-1, reads.shape[1], reads.shape[2]))
    for i in range(reads.shape[0]-1):
        diff[i,:,:] = reads[i+1,:,:] - reads[i,:,:]
    var = np.var(diff, axis=0)
    return var

def calcgain(filename):
    reads = getreads(filename)

    pixmask = (reads[0] < 17000.).flatten()

    diff = (reads[-1] - reads[0]).flatten()[pixmask]/(reads.shape[0]-1)
    var = calcvar(reads).flatten()[pixmask]
    clip = (var<5000) & (diff>200) & (diff<1500)
    diff = diff[clip]
    var = var[clip]
    x = np.array([np.ones(diff.size), diff]).T
    beta = np.dot(np.linalg.inv(np.dot(x.T,x)), np.dot(x.T, var))
    print np.sqrt(beta[0]/2), beta[1]
    testplots(var, diff)

def testplots(var, diff):
    f1, a1 = plt.subplots(1,1)
    a1.scatter(diff, var, marker='x')
    f2, a2 = plt.subplots(1,1)
    a2.hist(diff, bins=50)
    f3, a3 = plt.subplots(1,1)
    a3.hist(var, bins=50)
    import RaiseWindow
    plt.show()

if __name__=='__main__':
    fn = 'CRSA00006343.fits'
    #fn = 'CRSA00006842.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    calcgain(datadir+fn)
