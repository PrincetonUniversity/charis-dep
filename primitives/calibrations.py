import tools
import os
import pyfits as pf
import numpy as np
import copy

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
        outHDU[-1].data = outArray
        return outHDU
        
    else:
        log.error("The ndrs list passed into fitNDRs failed to have its slope fit!")
        return False
    
    
        
    
        
        
        
        
        
        
        
        
    
    
    