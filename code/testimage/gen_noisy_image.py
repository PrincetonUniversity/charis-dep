#!/usr/bin/env python

import numpy as np
import glob
import os
from astropy.io import fits

def addnoise(image, darkrms=5, darkdir='~/HiCIAO_Darks'):
    """
    Function addnoise

    Inputs: 
    1. image: 2D ndarray to which to add noise.
    
    Optional Inputs:
    2. darkrms: float. Read noise rms after correlated noise suppression.
                       Default 5 ADU.
    3. darkdir: string. Directory where HiCIAO dark frames live.  
                       Default '~/HiCIAO_Darks'

    Returns an image of the same shape as the input image with photon
    noise and read noise included.

    """
    
    try:
        assert len(image.shape) == 2, "Input image to addnoise must be a 2D array."
    except:
        raise AttributeError("Input image to addnoise must be a 2D array.")

    shotnoise = np.random.poisson(lam=image)

    #######################################################################
    # Pick, load, and scale a random dark from darkdir
    #######################################################################

    darklist = glob.glob(os.path.expanduser(darkdir) + '/HICA*.fits')
    i = int(np.random.random()*len(darklist))

    try:
        dark = fits.open(darklist[i])[0].data
    except:
        raise OSError("Unable to load HiCIAO dark frames from " + darkdir + " in addnoise.")

    if not (dark.shape[0] >= image.shape[0] and dark.shape[1] >= image.shape[1]):
        raise ValueError("Dark frame must be larger than input image to addnoise.")

    dark *= darkrms/np.std(dark)
    dark = dark[:shotnoise.shape[0], :shotnoise.shape[1]]

    return shotnoise + dark

if __name__ == "__main__":

    #######################################################################
    # Test case with a local file.  May need to adjust location of HiCIAO darks.
    # The dark frames are assumed to live on a local filesystem in ~/HiCIAO_Darks
    #######################################################################

    im = fits.open('summed_bkgnd_Strehl80.fits')[0].data
    im *= 1e5
    noisy = addnoise(im, darkrms=10)
    
    fits.writeto('test_noisy.fits', noisy.astype(np.float32), clobber=True)
    
