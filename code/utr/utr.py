import numpy as np
from image import Image
import tools
log = tools.getLogger('main')

def getreads(datadir, filename, read_idx=[1,None], biassub=None):
    """
    Get reads from fits file and put them in the correct 
    format for the up-the-ramp functions. 

    Inputs:
    1. datadir:   string, the data directory 
    2. filename:  string, name of the fits file

    Optional inputs:
    1. read_idx:  list, [first index, last index] to extract from
                  the hdulist: index = 1 is the first read, use 
                  last index = None to use all reads after the 
                  first index
    2. biassub:   string, perform bias subtraction using the
                  'top', 'bottom', or 'all' (top and bottom)
                  the reference pixels. If None, do not perform 
                  the bias subtraction

    Returns:
    1. reads:     3D ndarray, shape = (2048, 2048, nreads)
                  or (2048, 2112, nreads) 

    Note: The current return shape feels a bit awkward, but it 
    works better for broadcasting; we may decide to change this 
    in the future. 
    """

    try:
        from astropy.io import fits
    except:
        import pyfits as fits

    log.info("Getting reads from "+datadir+filename)
    if biassub is not None:
        log.info("Subtracting mean from "+biassub+" reference pixels")

    hdulist = fits.open(datadir+filename)
    shape = hdulist[1].data.shape
    reads = np.zeros((shape[0], shape[1], len(hdulist[read_idx[0]:read_idx[1]])))

    for i, r in enumerate(hdulist[read_idx[0]:read_idx[1]]):
        reads[:,:,i] = r.data
        if biassub is not None:
            numchan = shape[1]/64
            for j in xrange(numchan):
                if biassub=='top':
                    refpix = reads[-4:, j*64:(j+1)*64, i]
                elif biassub=='bottom':
                    refpix = reads[:4, j*64:(j+1)*64, i]
                elif biassub=='all':
                    top = reads[-4:, j*64:(j+1)*64, i]
                    bottom = reads[:4, j*64:(j+1)*64, i]
                    refpix = np.concatenate([top, bottom])
                reads[:, j*64:(j+1)*64, i] -= refpix.mean()
    return reads

def _interp_coef(nreads, sig_rn, cmin, cmax, cpad=500, interp_meth='linear'):
    """
    Build interpolation functions for the up-the-ramp coefficients, 
    which are the weights that give the count (c) and intercept (a)
    in the equation y_i = a + i*c, where i is an integer from 1 to 
    nreads. This function returns interpolation objects for the 
    coefficients needed to calculate a and c. 

    Inputs:
    1. nreads:      int, the number of reads in the ramp
    2. sig_rn:      float, the std of the read noise in electrons
    3. cmin:        float, the minimum possible count rate in electrons
    4. cmax:        float, the maximum possible count rate in electrons

    Optional inputs:
    1. cpad:        int, pad for count rate, which may be necessary to
                    ensure that the interpolation range is large enough
    2. interp_meth: string, interpolation method. Possible methods 
                    are the same as for scipy's interp1d.

    Returns: 
    1. ia_coef:     Interpolation object for the coefficients needed to 
                    calculate a: ia_coef(count) = [w1, w2, w3...w_nreads]
    2. ic_coef:     Interpolation object for the coefficients needed to 
                    calculate c: ic_coef(count) = [w1, w2, w3...w_nreads]
    3. ic_ivar:     Interpolation object for count rate inverse variance

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
    # note: a = intercept and c = count (all in units of electrons)
    ###################################################################

    a_coef = []
    c_coef = []
    c_ivar = []
    cvals = np.arange(np.floor(cmin)-cpad, np.ceil(cmax)+1)
    for c in cvals:
        cov = cov_shot*c + cov_read
        invcov = np.linalg.inv(cov)
        var = np.linalg.inv(np.dot(x.T, invcov).dot(x))
        coef = np.dot(var, np.dot(x.T, invcov))
        a_coef.append(coef[0])
        c_coef.append(coef[1])
        c_ivar.append(1.0/var[1,1])
    a_coef = np.array(a_coef)
    c_coef = np.array(c_coef)
    c_ivar = np.array(c_ivar)
    ia_coef = interpolate.interp1d(cvals, a_coef, axis=0, kind=interp_meth)
    ic_coef = interpolate.interp1d(cvals, c_coef, axis=0, kind=interp_meth)
    ic_ivar = interpolate.interp1d(cvals, c_ivar, axis=0, kind=interp_meth)

    return ia_coef, ic_coef, ic_ivar


def utr_rn(reads=None, datadir=None, filename=None, sig_rn=20.0, return_im=False, **kwargs):
    """
    Sample reads up-the-ramp in the read noise limit. We assume the counts 
    in each pixel obey the linear relation y_i = a + i*b*dt = a + i*c, 
    where i is an integer from 1 to nreads, and c = b*dt is the count. The 
    user can either pass the reads directly to this function or give the 
    data directory and file name containing the reads. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (2048, 2112, nreads) or
                   (2048, 2048, nreads), i.e. with or without the reference 
                   channel
    2. datadir:    string, the data directory. Only needed when the reads 
                   are not given
    3. filename:   string, fits file name. Only needed when the reads 
                   are not given

    Optional inputs:
    1. sig_rn:     float, the std of the read noise in ADU 
    2. return_im:  bool, if True, return an Image class object

    Returns:
    1. c_arr:      2D ndarry of Image class object, best-fit 
                   count in each pixel
    """

    if reads is None:
        reads = getreads(datadir, filename, **kwargs)

    nreads = reads.shape[2]

    ###################################################################
    # If we are read noise limited, then the count (c = b*dt) is given
    # by 12/(N^3 - N)*sum((i - (N+1)/2)*y_i). We simply sum the reads, 
    # weighting each by (i - (N+1)/2). 
    ###################################################################

    factor = 12.0/(nreads**3 - nreads)
    weights = np.arange(1,nreads+1) - (nreads+1)/2.0
    c_arr = factor*np.sum(weights*reads, axis=2)
    if return_im:
        ivar = (factor*sig_rn)**2*np.sum(weights**2)
        ivar = (1.0/ivar)*np.ones(c_arr.shape)
        return Image(data=c_arr, ivar=ivar) 
    else:
        return c_arr

def utr(reads=None, datadir=None, filename=None, sig_rn=20.0, gain=4.0,\
        interp_meth='linear', calc_chisq=False, **kwargs):
    """
    Sample reads up-the-ramp taking both shot noise and read noise 
    into account. We assume the counts in each pixel obey the linear 
    relation y_i = a + i*b*dt = a + i*c, where i is an integer from 
    1 to nreads, and c = b*dt is the count. The user can either pass 
    the reads directly to this function or give the data directory 
    and file name containing the reads. 

    Inputs:
    1. reads:      3D ndarray, the reads to be read up the ramp. Currently
                   the shape should be (2048, 2112, nreads) or
                   (2048, 2048, nreads), i.e. with or without the reference 
                   channel
    2. datadir:    string, the data directory. Only needed when the reads 
                   are not given. 
    3. filename:   string, fits file name. Only needed when the reads 
                   are not given. 

    Optional inputs:
    1. sig_rn:        float, the std of the read noise in ADU 
    2. gain:          float, the detector gain (electrons/ADU) 
    3. interp_meth:   string, the interpolation method to use when 
                      interpolating over the up-the-ramp coefficients.
                      Possible methods are the same as for scipy's interp1d.

    Returns:
    1. im             Image class object containing the count in electrons, 
                      ivar, and flags (not yet) for every pixel in the image. 
    """

    if reads is None:
        reads = getreads(datadir, filename, **kwargs)

    nreads = reads.shape[2]

    ###################################################################
    # Sample up-the-ramp in the read noise limit to get an estimate of 
    # the count for the covariance matrix calculation. Convert everthing
    # from ADU to electrons
    ###################################################################

    c_rn_arr = utr_rn(reads, sig_rn=sig_rn) # count rate (ADU) in RN limit
    c_rn_arr *= gain
    sig_rn *= gain
    reads *= gain

    ###################################################################
    # Generate interpolation objects for the up-the-ramp coefficients,
    # calculate the count (c) and intercept (a) for every pixel. The
    # calculation is done per row to conserve memory 
    ###################################################################

    interp_objs = _interp_coef(nreads, sig_rn, c_rn_arr.min(),\
                               c_rn_arr.max(), interp_meth=interp_meth)

    c_arr = np.zeros(c_rn_arr.shape)
    a_arr = np.zeros(c_rn_arr.shape)
    ivar_arr = np.zeros(c_rn_arr.shape)
    for row in xrange(c_rn_arr.shape[0]):
        a_coef = interp_objs[0](c_rn_arr[row,:])
        a_arr[row,:] = np.sum(a_coef*reads[row,:,:], axis=1)
        c_coef = interp_objs[1](c_rn_arr[row,:])
        c_arr[row,:] = np.sum(c_coef*reads[row,:,:], axis=1)
        ivar_arr[row,:] = interp_objs[2](c_rn_arr[row,:])
    del a_coef
    del c_coef

    ###################################################################
    # Calculate chi-squared for every pixel. The flags will be generated
    # here. Currently, we are calculating this in the read noise limit.
    # We are still thinking about how to optimize the general case. 
    ###################################################################

    if calc_chisq:
        chisq = np.zeros(c_arr.shape)
        for i in range(nreads):
            chisq += (reads[:,:,i] - a_arr - c_arr*(i+1))**2
        chisq /= sig_rn**2 
        im = Image(data=c_arr, ivar=ivar_arr, chisq=chisq)
    else:
        im = Image(data=c_arr, ivar=ivar_arr)

    return im
