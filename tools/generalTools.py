import tools
import numpy as np
import pyfits as pf

def testToolFunc():
    """
    Just a test tool func.  Will be deleted once more real tools are written.
    """
    log = tools.setUpLogger('testLog.tools')
    print 'Inside testToolFunc'
    log.info('InfoMsgInsideTools')
    
def loadDataAry(input):
    """
    """
    if isinstance(input, np.ndarray):
        log.debug("Input inSciNDR was of type np.ndarray.")
        outData = input
    elif isinstance(input, pf.hdu.hdulist.HDUlist) or isinstance(input,str):
        if isinstance(input,str):
            log.debug('input was a string, so checking it exists and loading it.')
            if os.path.exists(input):
                inFits = pf.open(input,'readonly')
            else:
                self.critical = log.critical("The provided file name, "+input+", does not exist!!")    
        elif isinstance(input, pf.hdu.hdulist.HDUlist):
            log.debug("Input input was of type pf.hdu.hdulist.HDUlist")
            inFits = input
            outFits = copy.deepcopy(inFits)
        if len(outFits)==1:
            outData = outFits[0].data
        if len(inFits)==2:
            outData = outFits[1].data
        else:
            log.critical("The length of the input data array was greater than 2, which isn't handled right now!!")
    else:
        log.critical("The type for input, "+repr(type(input))+", is not supported by this primitive!")
    
    return outData