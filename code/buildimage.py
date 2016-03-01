#/usr/bin/env python 

import numpy as np
from image import Image
import tools

def buildimage(reads, sig_rn=15.0, rn_limit=True):
    """
    Build an Image class object from a series of reads. 

    Inputs: 
    1. reads:    3D ndarray, the reads to be read up the ramp. Currenty
                 the shape should be (nreads, 2040, 2040), i.e. the reference
                 pixels have been removed.

    Optional inputs:
    rn_limit:     bool, if True, assume read noise limited
                  (always True for now)

    Returns:
    1. im:    Image class object    
    """
    c_arr, chisq, ivar = tools.utr(reads, sig_rn, rn_limit)
    im = Image(data=c_arr, ivar=ivar)
    return im

if __name__=='__main__':
    try:
        from astropy.io import fits
    except:
        import pyfits as fits
    fn = 'CRSA00006343.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    hdulist = fits.open(datadir+fn)
    reads = np.array([h.data[4:-4,64+4:-4] for h in hdulist[1:]])
    im = buildimage(reads)
