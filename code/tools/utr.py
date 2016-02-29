import numpy as np

def utr(reads, sig_rn=15.0, rn_limit=True, return_a=False):
    """
    Sample reads up-the-ramp. Here, we assume the counts in each pixel 
    obey the linear relation y_i = a + i*b*dt = a + i*c, where i is an 
    integer from 1 to nreads, and c = b*dt is the count. 

    Inputs:
    1. reads:    3D ndarray, the reads to be read up the ramp. Currenty
                 the shape should be (nreads, 2040, 2040), i.e. the reference
                 pixels have been removed.

    Optional inputs:
    1. sig_rn:   float, the std of the read noise. 
    2. rn_limit  bool, if True, assume read noise limited 
    3. return_a: bool, if True, return a (the best-fit intercepts)

    Returns:
    1. c_arr:    2D ndarray, the best-fit counts (c = b*dt) in each pixel
    2. chisq:    2D ndarray, chi-squared of each pixel
    3. ivar:     2D ndarray or float, the inverse variance of each pixel
                 if read noise limited, a single number is given
    4. a_arr:    2D ndarray, the best-fit intercepts (a) in each pixel.
                 only returned if return_a = True.                 
    """
    
    assert reads.shape[1:] == (2040, 2040), 'reads is not the correct shape'
    nreads = reads.shape[0]

    if rn_limit:
        
        ###################################################################
        # If we are read noise limited, then the count (c = b*dt) is given
        # by 12/(N^3 - N)*sum((i - (N+1)/2)*y_i). We simply sum the measured
        # count and weight each read by (i - (N+1)/2). 
        ###################################################################

        factor = 12.0/(nreads**3 - nreads)
        weights = np.arange(1,nreads+1) - (nreads+1)/2.0
        c_arr = factor*np.tensordot(weights, reads, axes=(0,0))
        var_c = (factor*sig_rn)**2*np.sum(weights**2)

        ymean = np.mean(reads, axis=0)
        imean = 0.5*(nreads+1)
        a_arr = ymean - c_arr*imean

        i_arr = np.arange(1, nreads+1)*np.ones((reads.shape[2], reads.shape[1], nreads))
        chisq = np.sum((reads - a_arr - c_arr*i_arr.T)**2/sig_rn**2, axis=0)

        return (c_arr, chisq, 1/var_c, a_arr) if return_a else (c_arr, chisq, 1/var_c)
    else:
        print 'This option has not been implemented yet!'
        exit(1)

if __name__=='__main__':
    from astropy.io import fits
    fn = 'CRSA00006343.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    hdulist = fits.open(datadir+fn)
    reads = np.array([h.data[4:-4,64+4:-4] for h in hdulist[1:]])
    count, chisq, ivar = utr(reads)
    out1 = fits.HDUList(fits.PrimaryHDU(count))
    out1.writeto('../count.fits', clobber=True)
    out2 = fits.HDUList(fits.PrimaryHDU(chisq))
    out2.writeto('../chisq.fits', clobber=True)
