import numpy as np
from image import Image

def _getreads(datadir, filename, refchan=True):
    """
    Get reads from fits file and put them in the correct 
    format for the up-the-ramp functions. 

    Inputs:
    1. datadir:   string, the data directory 
    2. filename:  string, name of the fits file

    Optional inputs:
    1. refchan:   bool, if True, the reference channel
                  is included in the readouts

    Returns:
    1. reads:     3D ndarray, shape = (2040, 2040, nreads)
    """
    try:
        from astropy.io import fits
    except:
        import pyfits as fits
    hdulist = fits.open(datadir+filename)
    reads = np.zeros((2040,2040,len(hdulist[1:])))
    shift = 64 if refchan else 0
    for i, read in enumerate(hdulist[1:]):
        reads[:,:,i] = read.data[4:-4,4+shift:-4]
    return reads

def _interp_coef(nreads, sig_rn, cmin, cmax, interp_meth='linear'):
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
    1. interp_meth: string, interpolation method. Possible methods 
                    are the same as for scipy's interp1d.

    Returns: 
    1. ia_coef: Interpolation object for the coefficients needed to 
                 calculate a: ia_coef(count) = [w1, w2, w3...w_nreads]
    1. ic_coef: Interpolation object for the coefficients needed to 
                 calculate c: ic_coef(count) = [w1, w2, w3...w_nreads]

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

    a_coef = []
    c_coef = []
    cvals = np.arange(np.floor(cmin), np.ceil(cmax)+1)
    for c in cvals:
        cov = cov_shot*c + cov_read
        invcov = np.linalg.inv(cov)
        coef = np.linalg.inv(np.dot(x.T, invcov).dot(x))
        coef = np.dot(coef, np.dot(x.T, invcov))
        a_coef.append(coef[0])
        c_coef.append(coef[1])
    a_coef = np.array(a_coef)
    c_coef = np.array(c_coef)
    ia_coef = interpolate.interp1d(cvals, a_coef, axis=0, kind=interp_meth)
    ic_coef = interpolate.interp1d(cvals, c_coef, axis=0, kind=interp_meth)

    return ia_coef, ic_coef


def utr_rn(reads=None, datadir=None, filename=None, sig_rn=15.0,\
           return_im=False, refchan=True):
    """
    Sample reads up-the-ramp in the read noise limit. We assume the counts 
    in each pixel obey the linear relation y_i = a + i*b*dt = a + i*c, 
    where i is an integer from 1 to nreads, and c = b*dt is the count. The 
    user can either pass the reads directly to this function or give the 
    data directory and file name containing the reads. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (2040, 2040, nreads), i.e. the 
                   reference pixels have been removed. If None, directory
                   and file name of fits file must be given. 
    2. datadir:    string, the data directory. Only needed when the reads 
                   are not given. 
    3. filename:   string, fits file name. Only needed when the reads 
                   are not given. 

    Optional inputs:
    1. sig_rn:     float, the std of the read noise. 
    2. return_im:  bool, if True, return an Image class object. 
    3. refchan:    bool, if True, the reference channel is included 
                   in each read. Not necessary if the reads are passed 
                   directly to this function. 

    Returns:
    1. c_arr       2D ndarry of Image class object, best-fit 
                   count in each pixel
    """

    if reads is not None:
        assert reads.shape[:2]==(2040, 2040), 'reads is not the correct shape'
    else:
        reads = _getreads(datadir, filename)

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

def utr(reads=None, datadir=None, filename=None, sig_rn=15.0,\
        interp_meth='linear', refchan=True):
    """
    Sample reads up-the-ramp taking both shot noise and read noise 
    into account. We assume the counts in each pixel obey the linear 
    relation y_i = a + i*b*dt = a + i*c, where i is an integer from 
    1 to nreads, and c = b*dt is the count. The user can either pass 
    the reads directly to this function or give the data directory 
    and file name containing the reads. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (2040, 2040, nreads), i.e. the 
                   reference pixels have been removed. If None, directory
                   and file name of fits file must be given. 
    2. datadir:    string, the data directory. Only needed when the reads 
                   are not given. 
    3. filename:   string, fits file name. Only needed when the reads 
                   are not given. 

    Optional inputs:
    1. sig_rn:        float, the std of the read noise. 
    2. interp_meth:   string, the interpolation method to use when 
                      interpolating over the up-the-ramp coefficients.
                      Possible methods are the same as for scipy's interp1d.
    3. refchan:       bool, if True, the reference channel is included 
                      in each read. Not necessary if the reads are passed 
                      directly to this function. 

    Returns:
    1. im             Image class object containing the count, ivar (not yet),
                      and flags (not yet) for every pixel in the image. 
    """

    if reads is not None:
        assert reads.shape[:2]==(2040, 2040), 'reads is not the correct shape'
    else:
        reads = _getreads(datadir, filename, refchan)

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

    icoef = _interp_coef(nreads, sig_rn, c_rn_arr.min(),\
                           c_rn_arr.max(), interp_meth=interp_meth)

    c_coef = icoef[1](c_rn_arr)
    c_arr = np.sum(c_coef*reads, axis=2)
    del c_coef

    a_coef = icoef[0](c_rn_arr)
    a_arr = np.sum(a_coef*reads, axis=2)
    del a_coef

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
    fn = 'CRSA00006343.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    reads = _getreads(datadir, fn)
    im = utr(reads)
    im.write('test_utr.fits')
    im = utr_rn(reads, return_im=True)
    im.write('test_utr_rn.fits')
