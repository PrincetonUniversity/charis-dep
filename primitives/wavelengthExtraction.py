import tools
import os
import re
import pyfits as pf
import numpy as np
import scipy as sp
import copy
import warnings

log = tools.getLogger('main.prims',lvl=100,addFH=False)

def pcaTest(flux, ncomp=5):
    """
    """
    
    log.debug("Now inside pcaTest primitive.  Will try to perform PCA.")
    np.seterr(all='ignore')              
    nframes = flux.shape[0]
    oldshape = flux.shape
    log.debug("input array had shape: "+repr(oldshape))
    
    # convert to a 2D array [nframes,flattened 2d original data array to 1D]
    fluxFlattened = np.reshape(flux, (nframes, -1))
    log.debug("reshaped array is now: "+repr(flux))
    
    # First step: subtract the mean
    meanflux = np.mean(fluxFlattened, axis=0)
    for i in range(nframes):
        flux[i] -= meanflux
    
    # super basic version of SVD.  ACORNS uses a much more advances version with multiprocessing we will implement later
    log.debug("about to try np.linalg.svd")
    u, s, V = np.linalg.svd(fluxFlattened.T,full_matrices=False)
    uNew = U.T[:ncomp]  # These are the top components from the PCA/SVD decomposition, but need to be reshaped to de-flatten
    log.debug("output uNew has shape: "+repr(uNew.shape))
    
    # de-flatten U array to make PCA component array
    pcaAry = np.reshape(uNew, (ncomp, oldshape[1], oldshape[2]))
    log.debug("pcaAry has shape: "+repr(pcaAry.shape))
    pcaAryFlattend = uNew
    
    # fit the PCA components to the frames and store their coefficients
    coefAry = []
    for i in range(nframes):
        coeffs = np.linalg.lstsq(pcaArryFlattened.T,fluxFlattened[i])[0]
        coefAry.append(coeffs)
    
    # subtract fitted components from a frame
    fluxFlattendSubbed = fluxFlattend
    for i in range(nframes):
        fluxFlattendSubbed[i] -= sp.dot(coefAry[i], pcaAryFlattened)
    fluxSubbed = np.reshape(fluxFlattendSubbed,(nframes, oldshape[1], oldshape[2]))
    
    # find the residuals for each fitted component subtracted 
    residuals = np.zeros((nframes,ncomp,oldshape[1], oldshape[2]))
    for i in range(nframes):
        fluxCur = flux[i]
        for j in range(ncomp):
            fluxCur -= coefAry[i][j]*pcaAry[j]
            residuals[i][j] = fluxCur
    
    ##### NOW INPUT SOME GDPS DATA FOR A GOOD RUN AND FIRST SEE WHAT THE PCA 
    ##### COMPS LOOK LIKE COMPARED TO THE ORIG FRAMES, THEN TRY OUT FITTING 
    ##### AND VISUALLY COMPARE RESULTS, THEN MAKE THE PLOT OF RESIDUALS LIKE 
    ##### TIM ASKED FOR (y=var(residual),x=number of components subtracted).
    
    