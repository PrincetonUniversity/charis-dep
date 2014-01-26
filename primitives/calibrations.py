import tools
import os
import pyfits as pf
import numpy as np

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
    This primitive will mask the 
    """
    #log = tools.getLogger('main.prims',lvl=100,addFH=False)
    print("this primitive will mask the BPM/hot pixels; still in test mode!!")
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
    