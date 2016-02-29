#/usr/bin/env python 

try:
    from astropy.io import fits
except:
    import pyfits as fits
import numpy as np
from image import Image
import tools

def _getreads(filename):
    """
    Get 
    """
    hdulist = fits.open(filename)
    reads = np.array([h.data[4:-4,64+4:-4] for h in hdulist[1:]])
    return reads

def buildimage(filename, rn_limit=True):
    """
    Build an Image class object from a series of reads. 

    Inputs: 
    1. filename:    string, directory+name of fits file containing reads 

    Optional inputs:
    rn_limit:    bool, if True, assume read noise limited
                 (always True for now)

    Returns:
    1. im:    Image class object    
    """
    sig_r = 20.0
    reads = getreads(filename)
    nreads = reads.shape[0]
    cov = calccov(count, sig_r, nreads)
    invcov = np.linalg.inv(cov)
    return slope, intercept

if __name__=='__main__':
    fn = 'CRSA00006343.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    slope, intercept = buildimage(datadir+fn)
    out1 = fits.HDUList(fits.PrimaryHDU(intercept))
    out1.writeto('intercept.fits', clobber=True)
    out2 = fits.HDUList(fits.PrimaryHDU(slope))
    out2.writeto('slope.fits', clobber=True)
