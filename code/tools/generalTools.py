import tools
import numpy as np
import pyfits as pf
import copy
import os
import sys
import re
import warnings

log = tools.getLogger('main.tools',lvl=0,addFH=False)
    
def testToolFunc():
    """
    Just a test tool func.  Will be deleted once more real tools are written.
    """
    #log = tools.getLogger('main.tools',lvl=100,addFH=False)
    print 'Inside testToolFunc'
    log.info('InfoMsgInsideTools')
 
def arrayRepr(ary):
    """
    Return a well represented string for any input 2D array.  
    Start said output array on a new line so all the array elements line up. 
    """
    if not isinstance(ary, np.ndarray):
        ary = np.array(ary)
    
    s = ""
    if len(ary.shape)==2:
        for i in range(0,ary.shape[0]):
            for j in range(0,ary.shape[1]): 
                if j>0:
                    s +=("  ")
                s2 = "%.5f"%(ary[i][j])
                s+=s2
            s+="\n"
    elif len(ary.shape)==1:
        for j in range(0,ary.shape[0]): 
            if j>0:
                s +=("  ")
            s2 = "%.5f"%(ary[j])
            s+=s2
    return s 
 
def timeString(duration):
    """
    takes a time duration in seconds and returns it in a nice string with info on number of 
    hours, minutes and/or seconds depending on duration.
    
    :param str duration: The duration in seconds.
    :return: The duration reformatted into a nice string.
    :rtype: str
    """
    if duration<60:
        totalTimeString = str(int(duration))+' seconds'
    elif duration>3600:
        duration = duration/3600.0
        totalTimeString = str(int(duration))+' hours and '+str(int(60*((duration)-int(duration))))+' minutes'
    else:
        duration = duration/60.0
        totalTimeString = str(int(duration))+' minutes and '+str(int(60*((duration)-int(duration))))+' seconds'
        
    return totalTimeString
   
def writeIntermediate(hdus, outputDir="", postStr = "", closeAfter=False):
    """
    A function to write a hdu, or list of them, to disk with an post-pended string 
    if desired.
    """    
    if isinstance(hdus, pf.hdu.hdulist.HDUList):
        hdus = [hdus]
        
    for hdu in hdus:
        try:
            frameName = hduToFileName(hdu)
            ps = postStr+".fits"
            log.debug("replacement to .fits will be '"+ps+"' ")
            outname = re.sub(".fits", ps, frameName)
            outname = os.path.join(outputDir,os.path.basename(outname))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                hdu.writeto(outname, clobber=True)
                log.info("Intermediate flux file written to: "+outname)
                if closeAfter:
                    hdu.close()
        except IOError, err:
            log.error(err)
            sys.exit(1)

def hduToFileName(hdu):
    """
    A function to take a input HDUList object and either get its filename or 
    make one from the OBJECT and DATE header keys if it doesn't exist.
    """
    frameName = ""
    log.debug("\ninput type into hduToFileName = "+repr(type(hdu))+'\n')
    try:
        frameName = hdu.filename()
        if not isinstance(frameName,str):
            log.critical("\nhdu.filename() did not return a string!!!\n")
    except:
        log.debug('trying to make frameName as the HDU had no filename')
        obj = ""
        try:
            obj = hdu[0].header['OBJECT']
            log.debug("obj found to be = "+obj)
        except:
            #for key in fluxFits[0].header:
            #    print key+" = "+str(fluxFits[0].header[key])
            log.error("Input HDUList did not have the OBJECT key or filename, so using 'UNKNOWNOBJ'."+\
                      "  NOTE: this happens when a new empty fits file is created without any original PHU loaded in.")
            obj = "UNKNOWNOBJ"
        date = str(hdu[0].header["DATE"])
        log.debug("date found to be = "+date)
        date = re.sub(" ","",date)
        frameName = obj+"_"+date+'.fits'
        log.debug('frame name created = '+frameName)
    log.debug("returning the fileName = '"+frameName+"'")
    
    return frameName

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
def loadListOfHDUs(input):
    """
    Load a list of strings to a list of HDULists, or just append the HDUList objects into a list if already opened.
    """
    HDUs = []
    if isinstance(inSci,list) and (not isinstance(inSci, pf.hdu.hdulist.HDUList)):
        for input in inSci:
            if isinstance(inSci, pf.hdu.hdulist.HDUList):
                fluxFits = frame
                frameName = tools.hduToFileName(fluxFits)
                HDUs.append(fluxFits)
            elif isinstance(inSci,str):
                fluxFits = tools.loadHDU(frame)
                frameName = frame
                HDUs.append(fluxFits)
    elif isinstance(inSci, pf.hdu.hdulist.HDUList):
        fluxFits = frame
        frameName = tools.hduToFileName(fluxFits)
        HDUs.append(fluxFits)
    elif isinstance(inSci,str):
        fluxFits = tools.loadHDU(frame)
        frameName = frame
        HDUs.append(fluxFits)       
    else:
        log.error("input type was not a list, a string or HDUList!!!")
        
    return HDUs

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
    #log.setStreamLevel(lvl=0)
    if os.path.exists(logFileName):
        lf = open(logFileName,"r")
        lines = lf.readlines()
        log.info(str(len(lines))+" lines are being converted to 70 characters per line and written to the PHU")
        #print '$$$$ '+str(len(lines))+" line(s) are being converted to 70 characters per line and written to the PHU"
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
                    #print line[last:last+70]#$$$$$$$$$$$$$$$$$$$$
                    last+=70
                    i+=1
                else:
                    hdu[0].header.add_history(line[last:])
                    #print line[last:] #$$$$$$$$$$$$$$$$$$$
                    i+=1
        if True:
            # a test of get_history
            print "\n\n generalTools:ln120: $$ testing get_history func of pf $$"
            history = hdu[0].header.get_history()
            print repr(history)
            print '\n\n generalTools:ln123: $$ Done get_history test $$'
    else:
        log.critical("Log file provided does not exists! So, can't update header!")
        
def rebin(a, shape):
    """
    (stolen from https://gist.github.com/zonca/1348792)
    Resizes a 2d array by averaging or repeating elements, 
    new dimensions must be integral factors of original dimensions
 
    Parameters
    ----------
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array
 
    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array, the data are averaged, 
        if the new shape is bigger array elements are repeated
 
    See Also
    --------
    resize : Return a new array with the specified shape.
 
    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
 
    >>> c = rebin(b, (2, 3)) #downsize
    >>> c
    array([[ 0. ,  0.5,  1. ],
           [ 2. ,  2.5,  3. ]])
 
    """
    M, N = a.shape
    m, n = shape
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    #print "sh = "+repr(sh)
    if m<M:
        #return a.reshape(sh).mean(-1).mean(1)
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)
    
    