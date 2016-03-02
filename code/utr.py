import numpy as np
from image import Image

def _interp_coeff(nreads, sig_rn, cmin, cmax, kind='linear'):
    """
    Build interpolation functions for the up-the-ramp coefficients, 
    which are the weights that give the count (c) and intercept (a)
    in the equation y_i = a + i*c, where i is an integer from 1 to 
    nreads. This function returns interpolation objects for the 
    coefficients needed to calculate a and c. 

    Inputs:
    1. nreads:   int, the number of reads in the ramp
    2. sig_rn:   float, the std of the read noise
    3. cmin:     float, the minimum possible count rate
    4. cmax:     float, the maximum possible count rate

    Optional inputs:
    1. kind      string, interpolation method. Possible methods 
                 are the same as for scipy's interp1d.

    Returns: 
    1. ia_coeff: Interpolation object for the coefficients needed to 
                 calculate a: ia_coeff(count) = [w1, w2, w3...w_nreads]
    1. ic_coeff: Interpolation object for the coefficients needed to 
                 calculate c: ic_coeff(count) = [w1, w2, w3...w_nreads]

    Note:
    The weights are given by the generalized least squares solution 
    for a linear model. It turns out that, if we know the read noise
    and the number of reads, the weights are only a function of the 
    count. The purpose of this function is to precalculate the
    weights to speed up the up-the-ramp calculation. 
    """
    from scipy import interpolate

    i_arr = np.arange(1, nreads+1)
    x = np.array([np.ones(i_arr.size), i_arr]).T

    ###################################################################
    # Calculate the components of the covariance matrix: cov_shot = 
    # shot noise component (divided by the count), cov_read = the read
    # noise component. Thus, cov = cov_shot*count + cov_read. 
    ###################################################################

    uptri = np.triu(np.repeat(np.arange(1, nreads+1)[:,None], nreads, axis=1))
    cov_shot = uptri + uptri.T - np.tril(uptri)
    cov_read = np.diag(np.ones(nreads))*sig_rn**2

    ###################################################################
    # Calculate the up-the-ramp coefficients for many count values, 
    # store them, and interpolate using scipy's interp1d function.
    # note: a = intercept and c = count
    ###################################################################

    c_coeff = []
    a_coeff = []
    cvals = np.arange(cmin, cmax+1)
    for c in cvals:
        cov = cov_shot*c + cov_read
        invcov = np.linalg.inv(cov)
        coeff = np.linalg.inv(np.dot(x.T, invcov).dot(x))
        coeff = np.dot(coeff, np.dot(x.T, invcov))
        a_coeff.append(coeff[0])
        c_coeff.append(coeff[1])
    a_coeff = np.array(a_coeff)
    c_coeff = np.array(c_coeff)
    ia_coeff = interpolate.interp1d(cvals, a_coeff, axis=0, kind=kind)
    ic_coeff = interpolate.interp1d(cvals, c_coeff, axis=0, kind=kind)

    return ia_coeff, ic_coeff


def utr_rn(reads, sig_rn=15.0, return_im=False):
    """
    Sample reads up-the-ramp in the read noise limit. We assume the counts 
    in each pixel obey the linear relation y_i = a + i*b*dt = a + i*c, 
    where i is an integer from 1 to nreads, and c = b*dt is the count. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (2040, 2040, nreads), i.e. the 
                   reference pixels have been removed.

    Optional inputs:
    1. sig_rn:     float, the std of the read noise. 
    2. return_im:  bool, if True, return an Image class object. 

    Returns:
    1. c_arr       2D ndarry of Image class object, best-fit 
                   count in each pixel
    """
    
    assert reads.shape[:2] == (2040, 2040), 'reads is not the correct shape'
    nreads = reads.shape[2]

    ###################################################################
    # If we are read noise limited, then the count (c = b*dt) is given
    # by 12/(N^3 - N)*sum((i - (N+1)/2)*y_i). We simply sum the reads, 
    # weighting each by (i - (N+1)/2). 
    ###################################################################

    factor = 12.0/(nreads**3 - nreads)
    weights = (np.arange(1,nreads+1) - (nreads+1)/2.0)*np.ones(reads.shape)
    c_arr = factor*np.sum(weights*reads, axis=2)
    if return_im:
        ivar = (factor*sig_rn)**2*np.sum(weights**2, axis=2)
        ivar = 1/ivar 
        return Image(data=c_arr, ivar=ivar) 
    else:
        return c_arr

def utr(reads, sig_rn=15.0, interp_meth='linear'):
    """
    Sample reads up-the-ramp taking both shot noise and read noise 
    into account. We assume the counts in each pixel obey the linear 
    relation y_i = a + i*b*dt = a + i*c, where i is an integer from 
    1 to nreads, and c = b*dt is the count. 

    Inputs:
    1. reads:         3D ndarray, the reads to be read up the ramp. 
                      Currently the shape should be (2040, 2040, nreads), 
                      i.e. the reference pixels have been removed.

    Optional inputs:
    1. sig_rn:        float, the std of the read noise. 
    2. interp_meth:   string, the interpolation method to use when 
                      interpolating over the up-the-ramp coefficients.
                      Possible methods are the same as for scipy's interp1d.

    Returns:
    1. im             Image class object containing the count, ivar (not yet),
                      and flags (not yet) for every pixel in the image. 
    """

    assert reads.shape[:2] == (2040, 2040), 'reads is not the correct shape'
    nreads = reads.shape[2]

    ###################################################################
    # Sample up-the-ramp in the read noise limit to get an estimate of 
    # the count for the covariance matrix calculation 
    ###################################################################

    c_rn_arr = utr_rn(reads)

    ###################################################################
    # Generate interpolation objects for the up-the-ramp coefficients,
    # calculate the count (c) and intercept (a) for every pixel
    ###################################################################

    icoeff = _interp_coeff(nreads, sig_rn, c_rn_arr.min(), c_rn_arr.max())

    c_coeff = icoeff[1](c_rn_arr)
    c_arr = np.sum(c_coeff*reads, axis=2)
    del c_coeff

    a_coeff = icoeff[0](c_rn_arr)
    a_arr = np.sum(a_coeff*reads, axis=2)
    del a_coeff

    ###################################################################
    # Calculate chi-squared for every pixel. The flags will be generated
    # here. Also need to add calculation of ivar, but I'm not sure how
    # to optimize this.
    ###################################################################

    i_arr = np.arange(1, nreads+1)*np.ones(reads.shape)
    chisq = np.zeros(reads.shape[:2])
    for i in range(nreads):
        chisq += (reads[:,:,i] - a_arr - c_arr*i_arr[:,:,i])**2/sig_rn**2

    im = Image(data=c_arr)

    return im

if __name__=='__main__':
    try:
        from astropy.io import fits
    except:
        import pyfits as fits
    fn = 'CRSA00006343.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    hdulist = fits.open(datadir+fn)
    reads = np.zeros((2040,2040,len(hdulist[1:])))
    for i, read in enumerate(hdulist[1:]):
        reads[:,:,i] = read.data[4:-4,64+4:-4]
    im = utr(reads)
    im.write('test_utr.fits')
    im = utr_rn(reads, return_im=True)
    im.write('test_utr_rn.fits')
