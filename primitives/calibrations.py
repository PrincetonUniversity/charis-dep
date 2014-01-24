import tools
import os
import pyfits as pf
import numpy as np

def testCalPrim():
    """
    Just a test prim during very early development.
    """
    log = tools.setUpLogger('testLog.prims')
    print("this is an empty test calibration primitive")
    log.info('testInfoMsgInsidePrim')
    
    tools.testToolFunc()
    
def maskHotPixels(inSciNDR, BPM):
    """
    This primitive will mask the 
    """
    log = tools.setUpLogger('testLog.prims')
    print("this primitive will mask the BPM/hot pixels; still in test mode!!")
    
    ## Make the type that is passed into the primitives standardized!!!!
    ## Load the provided sci data into a ndarray.
    inData = loadDataAry(inSciNDR)
    bpmData = loadDataAry(BPM)
    
    ## Got the data into a ndarray, now apply the BPM array.
    try:
        outData = outData*bpmData
    except:
        log.error("Something when wrong while performing maskHotPixels")
        
    return outData
    