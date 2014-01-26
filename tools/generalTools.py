import tools
import numpy as np
import pyfits as pf
import copy
import os

log = tools.getLogger('main.tools',lvl=100,addFH=False)
    
def testToolFunc():
    """
    Just a test tool func.  Will be deleted once more real tools are written.
    """
    #log = tools.getLogger('main.tools',lvl=100,addFH=False)
    print 'Inside testToolFunc'
    log.info('InfoMsgInsideTools')
    
def loadDataAry(input):
    """
    """
    #log = tools.getLogger('main.tools',lvl=100,addFH=False)
    log.debug("Trying to load input of type: "+repr(type(input)))
    if isinstance(input, np.ndarray):
        log.debug("Input was of type np.ndarray.")
        outData = input
    elif isinstance(input, pf.hdu.hdulist.HDUList) or isinstance(input,str):
        if isinstance(input,str):
            log.debug('input was a string, so checking it exists and loading it.')
            if os.path.exists(input):
                inHDU = pf.open(input,'readonly')
            else:
                log.critical("The provided file name, "+input+", does not exist!!")   
        elif isinstance(input, pf.hdu.hdulist.HDUList):
            log.debug("Input input was of type pf.hdu.hdulist.HDUlist")
            inHDU = input
        if len(inHDU)<=2:
            outData =  inHDU[-1].data.copy()        
        else:
            log.critical("The length of the input data array was greater than 2, which isn't handled right now!!")
        inHDU.close()
    else:
        log.critical("The type for input, "+repr(type(input))+", is not supported by this primitive!")
    log.info("data from input file correctly loaded and being returned as an ndarray")
    return outData

def loadHDU(input):
    """
    To load an input HDUlist or string into a copied HDUlist.
    
    Args:
        input (str, or hdu.hdulist.HDUList object): Object to either deepcopy 
                                                    if it is already an HDUlist,
                                                    or open it and then deepcopy 
                                                    it and close the original 
                                                    file if input is a string.
    """
    #log = tools.getLogger('main.tools',lvl=100,addFH=False)
    if isinstance(input, pf.hdu.hdulist.HDUList) or isinstance(input,str):
        if isinstance(input,str):
            log.debug('input was a string, so checking it exists and loading it.')
            if os.path.exists(input):
                inHDU = pf.open(input,'readonly')
            else:
                log.critical("The provided file name, "+input+", does not exist!!")   
        elif isinstance(input, pf.hdu.hdulist.HDUList):
            log.debug("Input input was of type pf.hdu.hdulist.HDUlist")
            inHDU = input
        else:
            log.critical("The length of the input data array was greater than 2, which isn't handled right now!!")
    elif isinstance(input, np.ndarray):
        log.critical("The function loadHDU can not return an HDU list from the input ndarray!")
    else:
        log.critical("The type for input, "+repr(type(input))+", is not supported by this primitive!")
    # Got inHDU so deepcopy it and its data to the outHDU
    outHDU = copy.deepcopy(inHDU)
    outHDU[-1].data = inHDU[-1].data.copy()
    inHDU.close()
    
    return outHDU

def updateOutputFitsHeader(hdu, logFileName='main.summary'):
    """
    Takes the lines in the log file and copies them to the header of an HDUlist
    header.  It is assumed that the header to be updated is that of the PHU,
    ie. hdu[0].header .
    
    Args:
        hdu (PyFits HDU list): HDUlist that will have its PHU updated with the
                                lines of the summary log file.
        
        logFileName (str): Complete path file name of the summary file to copy
                            its lines into the provided HDU's PHU.        
    """
    #log = tools.getLogger('main.tools',lvl=100,addFH=False)
    log.info("Loading up HDUlist with lines from log file: "+logFileName)
    if os.path.exists(logFileName):
        lf = open(logFileName,"r")
        lines = lf.readlines()
        log.info(str(len(lines))+" lines are being converted to 70 characters per line and written to the PHU")
        print '$$$$'+str(len(lines))+" lines are being converted to 70 characters per line and written to the PHU"
        lineNum = 1
        for line in lines:
            log.debug("Converting and loading line # "+str(lineNum)+" to the PHU")
            log.debug("This line of length "+str(len(line))+" will be converted into "+str(len(line)//70+1)+" History keys")
            lineNum+=1
            i= 0
            last = 0
            while i<=(len(line)//70):
                if i<(len(line)//70):
                    hdu[0].header.add_history(line[last:last+70])
                    last+=70
                    i+=1
                else:
                    hdu[0].header.add_history(line[last:-1])
                    i+=1
    else:
        log.critical("Log file provided does not exists! So, can't update header!")
        
    
    
    