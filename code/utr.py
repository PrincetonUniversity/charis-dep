import numpy as p
from image import Image

def utr_rn(reads, sig_rn=15.0, calc_chisq=False):
    """
    Sample reads up-the-ramp in the read noise limit. We assume the counts 
    in each pixel obey the linear relation y_i = a + i*b*dt = a + i*c, 
    where i is an integer from 1 to nreads, and c = b*dt is the count. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currenty
                   the shape should be (nreads, 2040, 2040), i.e. the 
                   reference pixels have been removed.

    Optional inputs:
    1. sig_rn:     float, the std of the read noise. 
    2. calc_chisq: bool, if True, calculate chi-squared for every 
                   pixel. This will only be necessary if we want 
                   to generate flags in this function. 

    Returns:
    1. im:       Image class object
    """
    
    assert reads.shape[1:] == (2040, 2040), 'reads is not the correct shape'
    nreads = reads.shape[0]

    ###################################################################
    # If we are read noise limited, then the count (c = b*dt) is given
    # by 12/(N^3 - N)*sum((i - (N+1)/2)*y_i). We simply sum the reads, 
    # weighting each by (i - (N+1)/2). 
    ###################################################################

    factor = 12.0/(nreads**3 - nreads)
    weights = np.arange(1,nreads+1) - (nreads+1)/2.0
    c_arr = factor*np.tensordot(weights, reads, axes=(0,0))
    var_c = (factor*sig_rn)**2*np.sum(weights**2)
    ivar = (1.0/var_c)*np.ones(c_arr.shape)

    if calc_chisq:
        ymean = np.mean(reads, axis=0)
        imean = 0.5*(nreads+1)
        a_arr = ymean - c_arr*imean
        i_arr = np.arange(1, nreads+1)*np.ones((reads.shape[2], reads.shape[1], nreads))
        chisq = np.sum((reads - a_arr - c_arr*i_arr.T)**2/sig_rn**2, axis=0)
    
    ###################################################################
    # Build an Image class object. Will generate bitmask here. For the
    # read noise limited case, ivar is the same for all pixels. Do we 
    # want to store this? 
    ###################################################################

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
    im = utr_rn(reads)
    im.write('test_utr_rn.fits')
