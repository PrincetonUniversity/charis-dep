
try:
    from astropy.io import fits
except:
    import pyfits as fits

import numpy as np
import scipy.signal

def gen_bad_pix_mask(image, filsize=5, threshold=5.0, return_smoothed_image=False):
    """
    """
    image_sm = scipy.signal.medfilt(image, filsize)
    res = image - image_sm
    sigma = np.std(res)
    goodpix = np.abs(res)/sigma < threshold
    return (goodpix, image_sm) if return_smoothed_image else goodpix

if __name__=='__main__':
    fn = 'CRSA00006343.fits'
    datadir = '/Users/protostar/Dropbox/data/charis/lab/'
    hdulist = fits.open(datadir+fn)
    reads = np.array([h.data[4:-4,64+4:-4] for h in hdulist[1:]])
    diff = reads[5] - reads[0]
    goodpix, image_sm = gen_bad_pix_mask(diff, return_smoothed_image=True)
