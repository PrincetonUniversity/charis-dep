import tools
import os
import re
import pyfits as pf
import numpy as np
import copy
import warnings
from scipy.signal import medfilt2d

log = tools.getLogger('main.prims',lvl=100,addFH=False)

def testCalPrim():
    """
    Just a test prim during very early development.
    """
    #log = tools.getLogger('main.prims',lvl=100,addFH=False)
    print("this is an empty test calibration primitive")
    log.info('testInfoMsgInsidePrim')
    
    tools.testToolFunc()
    
def maskHotPixels(inSciNDR, BPM):
    """
    This primitive will mask the bad pixels in each provided science frame.
    """
    #log = tools.getLogger('main.prims',lvl=100,addFH=False)
    log.info("Performing BPM masking on input with input BPM ")
    ## Make the type that is passed into the primitives standardized!!!!
    ## Load the provided sci data into a ndarray.
    inData = tools.loadDataAry(inSciNDR)
    bpmData = tools.loadDataAry(BPM)
    
    ## Got the data into a ndarray, now apply the BPM array.
    outData = False
    try:
        outData = inData*bpmData
    except:
        log.error("Something when wrong while performing maskHotPixels")
        
    return outData

def fitNDRs(ndrs):
    """
    A function to take a list of NDR frames, fit the slope and produce a single
    frame of photons/second.
    
    In latex, the slope equation used is:
    
    a \delta t = (12/(N^3-N))\sum_{i=1}^N (i-((N+1)/2))y_i
    
    OR
    
    a  = \sum_{i=1}^N [(12/((N^3-N)*\delta t))(i-((N+1)/2))] [y_i]
    
    
    Where a is the slope, \delta t is the duration of each exposure, N is the
    total number of exposures, i is the index of the exposure and y_i is the 
    number of counts for a pixel in exposure index i.
    
    The output PyFits object will take its header from the first frame in the
    list of NDRs passed into fitNDRs.
    
    Note: basics of the equation used here are in NDR_coefficientDerivation4.pdf
    
    Args:
        ndrs (list of PyFits objects or strings for their complete filenames):
            The files to have their slope fit.  The items in this list will be 
            loaded into pyfits objects if strings are provided.
    Returns:
        outHDU (A single PyFits object):  The resulting single frame
    """
    summaryLog = tools.getLogger('main.summary')
    ndrsLoaded = []
    exptimeFirst = 0
    #objectFirst = ''
    for ndr in ndrs:
        try:
            ndrOut = tools.loadHDU(ndr)
            exptime = ndrOut[0].header['P_TFRAME']
            #object = ndrOut[0].header['OBJECT']
            if exptimeFirst==0:
                exptimeFirst=exptime
                ndrsLoaded.append(ndrOut)
                objectFirst = object
            else:
                if exptime!=exptimeFirst:
                    log.error("The first frame's P_TFRAME was "+str(exptimeFirst)+
                              ", while this frame's is "+str(exptime))
                    log.error("Mismatching P_TFRAME frames cannot have their slope fit!")
                    break
#                 elif objectFirst!=object:
#                     log.error("The first frame's OBJECT was "+objectFirst+
#                               ", while this frame's is "+object)
#                     log.error("Mismatching OBJECT frames cannot have their slope fit!")
#                     break
                else:
                    ndrsLoaded.append(ndrOut)
            #ndrsLoaded.append(ndrOut)
        except:
            log.error("An exception occurred while loading one of the NDRs.")
                
    if len(ndrsLoaded)==len(ndrs):
        N = len(ndrsLoaded)
        summaryLog.info("The set of NDRs has an P_TFRAME of "+str(exptime)+" for each of its "+str(N)+" frames.")    
        deltaT = exptime
        constA = 12.0/((N**3.0-N)*deltaT)
        constB = (N+1.0)/2.0
        first = ndrsLoaded[0]
        outArray = np.zeros(first[-1].data.shape)
        outHDU = copy.deepcopy(ndrsLoaded[0])
        for i in range(1,N+1):
            left = constA*(i-constB)
            log.debug("The left value for frame "+str(i)+" = "+str(left))
            ndr = ndrsLoaded[i-1]
            outArray+=left*ndr[-1].data
            
        log.info("** MAYBE ADD MORE MSGS TO SUMMARYLOG DURING fitNDRs?? **")
        log.critical("** ADD COADD KEY TO HEADERS AND SUMMARYLOG **")
        log.info("COADD header added, but not sure if correct!!!!!")
        outHDU[0].header.update('COADD', str(len(ndrs)), "Number of NDRs fit together")
        outHDU[-1].data = outArray
        return outHDU
        
    else:
        log.error("The ndrs list passed into fitNDRs failed to have its slope fit!")
        return False
    
def destripe(frame, flat, hotpix, write_files, output_dir, bias_only,
             clean=True, storeall=True, r_ex=0, extraclean=True):

    """
    This function will destripe the input science/flux frame.  By that we mean 
    it will take the bias and reference pixels properly into account and remove
    the readnoise as best as possible following the ideas in Mosely 2010 and 
    described more fully in Brandt 2013 (ACORNS paper). 
    
    Function destripe takes two arguments:
    1.  A (usually) 2048 x 2048 array of flux values NOTE: Must be a str or HDUList
    2.  A (usually) 2048 x 2048 flatfield image NOTE: must be an ndarray
    3.  The coordinates of the hot pixels to be masked
    4.  Write destriped data to files? 
    5.  Directory to write data (ignored if write_files=False)
    6.  Use only reference pixels?

    Optional arguments:
    7.  Interpolate over hot pixels?  default True
    8.  Store all frames in memory?  default True
    9.  Radial exclusion region for vertical destriping, default 0.
        Ignored if using only reference pixels.
    10. Mask deviant pixels by smoothing with a large median filter
        and looking for discrepancies?  default True

    This function returns the destriped data.  It uses verticalmed,
    verticalref, horizontal, and interpbadpix from destripe_utils.
    
    NOTE: This code is a modified version of that found in the destripe 
    directory of the ACORNS-ADI pipeline along with the tools it calls.
    """
    #load hotpix and flat arrays (although currently just the BPM and fake flat)!!!!!
    hotpix = tools.loadDataAry(hotpix)
    #hotpix = np.where(hotpix>10000)#$$$$ making a sorta fax hotpix array
    flat = tools.loadDataAry(flat)
    #print 'type(hotpix) = '+repr(type(hotpix))#$$$$$$$$$$$
    #print 'type(flat) = '+repr(type(flat))#$$$$$$$$$$$
    
    np.seterr(all='ignore')
    if not (storeall or write_files):
        log.error("Error: attempting to run destripe without saving files to either disk or memory")

    ncoadd = 1
    try:
        if isinstance(frame, pf.hdu.hdulist.HDUList):
            fluxfits = frame
        elif isinstance(frame,str):
            fluxfits = pyf.open(frame, "readonly")
        else:
            log.error("input type was not a string or HDUList!!!")     
        header = fluxfits[0].header
        #print 'ln157'#$$$$$$$$$$$
        try:
            ncoadd = header['COADD']
        except:
            ncoadd = 1
        #print 'ln179'#$$$$$$$$$$$
        flux = fluxfits[-1].data.astype(np.float32)
        #print 'ln181'#$$$$$$$$$$$
        dimy, dimx = flux.shape
        #print 'flux.shape = '+repr(flux.shape)+', hotpix.shape = '+repr(hotpix.shape)
        try:
            #print 'ln190'#$$$$$$$$$$
            if hotpix is not None:
                flux[hotpix] = np.nan
            #print 'ln193'#$$$$$$$$$$
        except:
            #print 'ln195'#$$$$$$$$$$
            log.info("Original hotpix replacement method did not work, so using new np.where approach.")
            flux = np.where(hotpix>0,flux,np.nan)
            #print 'ln198'#$$$$$$$$$$
    except:
        log.error("Error reading file " + repr(frame))
        exit()

    ##############################################################
    # reference voltage scaled by a number less than one provides
    # the best estimate of the vertical pattern, 0.87 in my tests.
    ##############################################################

    if bias_only:
        sub_coef = 0.87
    else:
        sub_coef = 1

    try:
        for stripe in range(32):      
            tools.horizontal(flux, stripe)
    except:
        log.error("Horizontal destriping failed on frame " + frame)
        exit()
            
    ##############################################################
    # Calculate and subtract the vertical pattern.
    ##############################################################

    try:
        if bias_only:
            oddstripe = tools.verticalref(flux, 1)
            evenstripe = oddstripe[::-1, :]
        else:
            oddstripe, evenstripe = tools.verticalmed(flux, flat, r_ex=r_ex)
    except:
        log.error("Vertical destriping failed on frame " + frame)
        exit()
        
    for i in range(1, 33, 2):
        flux[64 * i:64 * i + 64] -= oddstripe * sub_coef
        flux[64 * i - 64:64 * i] -= evenstripe * sub_coef

    ##############################################################
    # Four rows on each edge are reference pixels--don't
    # flatfield them
    ##############################################################

    flux[4:-4, 4:-4] /= flat[4:-4, 4:-4]

    np.putmask(flux, flux < -1000, 0)
    np.putmask(flux, flux > 5e4 * ncoadd, np.nan)

    try:
        if clean:
            if extraclean:
                
                #############################################################
                # We'll be taking a median, so make half the bad pixels
                # inf and the other half ninf
                #############################################################
                
                np.putmask(flux[::2, :], np.isnan(flux[::2, :]), np.NINF)
                np.putmask(flux[1::2, :], np.isnan(flux[1::2, :]), np.inf)
                resid = medfilt2d(flux, 11)
                
                fluxresid = np.abs(flux - resid)
                sigval = medfilt2d(fluxresid, 9)
                
                #############################################################
                # Mask everything deviant by at least 3.5 'sigma'.  Since
                # sigval is a median, for Gaussian errors, this is
                # 3.5 * sqrt(2*ln(2)) ~= 4.1 sigma.
                #############################################################
                
                mask = fluxresid > 3.5 * sigval
                mask[:10] = 0
                mask[-10:] = 0
                mask[:, :10] = 0
                mask[:, -10:] = 0

                np.putmask(flux, mask, np.nan)
                np.putmask(flux, np.isinf(flux), np.nan)
                
            tools.interpbadpix(flux)
    except:
        log.error("Cleaning bad pixels failed on frame " + frame)
        sys.exit(1)
        
    ##############################################################
    # We don't want any NaNs or infs in the returned data
    ##############################################################

    np.putmask(flux, np.logical_not(np.isfinite(flux)), 0)

    if write_files:
        try:
            fluxout = pyf.HDUList()
            flux_hdu = pyf.PrimaryHDU(flux, header)
            fluxout.append(flux_hdu)
    
            outname = re.sub(".fits", "_ds.fits", frame)
            outname = re.sub(".*HICA", output_dir + "/" + "HICA", outname)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fluxout.writeto(outname, clobber=True)
                fluxout.close()
        except IOError, err:
            log.error(err)
            sys.exit(1)

    if storeall:
        return flux
    else:
        return 
    
    
    
    