import tools
import os
import re
import pyfits as pf
import numpy as np
from scipy import ndimage
import copy
import warnings
import pylab
plt = pylab.matplotlib.pyplot

log = tools.getLogger('main.prims',lvl=0,addFH=False)#('main.prims',lvl=100,addFH=False)

def pcaTest(flux, ncomp=5, outputDirRoot='.', writeFiles=True):
    """
    """
    
    log.debug("Now inside pcaTest primitive.  Will try to perform PCA.")
    np.seterr(all='ignore')
    # Check inputs are ndarrays, if not, extract data from the fits HDUlists
    print repr(len(flux))     
    inShape = tools.loadDataAry(flux[0]).shape
    fluxUse = np.zeros((len(flux),inShape[0],inShape[1]))
    for i in range(len(flux)):
        if type(flux[i])!=np.ndarray:        
            fluxUse[i]=tools.loadDataAry(flux[i]) 
        else:
            fluxUse[i]=flux[i]
    nframes = fluxUse.shape[0]
    oldshape = fluxUse.shape
    log.debug("input array had shape: "+repr(oldshape))
    
    # convert to a 2D array [nframes,flattened 2d original data array to 1D]
    fluxFlattened = np.reshape(fluxUse, (nframes, -1))
    log.debug("reshaped array is now: "+repr(fluxFlattened.shape))
    
    # First step: subtract the mean
    meanflux = np.mean(fluxFlattened, axis=0)
    for i in range(nframes):
        fluxFlattened[i] -= meanflux
    
    # super basic version of SVD.  ACORNS uses a much more advanced version with multiprocessing we will implement later $$$$$$$$$$$$$$$
    log.debug("about to try np.linalg.svd")
    u, s, V = np.linalg.svd(fluxFlattened.T,full_matrices=False)
    uNew = u.T[:ncomp]  # These are the top components from the PCA/SVD decomposition, but need to be reshaped to de-flatten
    log.debug("output uNew has shape: "+repr(uNew.shape))
    
    # de-flatten U array to make PCA component array
    pcaAry = np.reshape(uNew, (ncomp, oldshape[1], oldshape[2]))
    log.debug("pcaAry has shape: "+repr(pcaAry.shape))
    pcaAryFlattened = uNew
    
    # save PCA components as individual fits files
    outPCAdir= os.path.join(outputDirRoot,"pcaOutputs")
    if writeFiles:
        os.mkdir(outPCAdir)
        for j in range(ncomp):
            hdu = pf.PrimaryHDU(pcaAry[j])
            hduList = pf.HDUList([hdu])
            outFilename = os.path.join(outputDirRoot+"/pcaOutputs/","pcaComponent_"+str(j+1)+".fits")
            hduList.writeto(outFilename)
            hduList.close()
    
    # fit the PCA components to the frames and store their coefficients
    coefAry = []
    for i in range(nframes):
        coeffs = np.linalg.lstsq(pcaAryFlattened.T,fluxFlattened[i])[0]
        coefAry.append(coeffs)
    
    # subtract fitted components from a frame
    fluxFlattenedSubbed = np.zeros((nframes,ncomp+1,fluxFlattened.shape[1]))
    for i in range(nframes): #$$$$$$$$$$$$$$$$$$$$
        fluxFlattenedSubbed[i][0] = fluxFlattened[i]
        fluxTemp = fluxFlattened[i]
        for j in range(ncomp):
            fluxTemp -= coefAry[i][j]*pcaAryFlattened[j]
            fluxFlattenedSubbed[i][j+1] = fluxTemp
        print 'fluxFlattenedSubbed var for frame '+str(i)+" = "+repr(np.var(fluxFlattenedSubbed[i][-1]))
    fluxSubbed = np.reshape(fluxFlattenedSubbed,(nframes,ncomp+1, oldshape[1], oldshape[2]))
    
    ##### NOW INPUT SOME GDPS DATA FOR A GOOD RUN AND FIRST SEE WHAT THE PCA 
    ##### COMPS LOOK LIKE COMPARED TO THE ORIG FRAMES, THEN TRY OUT FITTING 
    ##### AND VISUALLY COMPARE RESULTS, THEN MAKE THE PLOT OF RESIDUALS LIKE 
    ##### TIM ASKED FOR (y=var(residual),x=number of components subtracted).
    if writeFiles:
#         # find the residuals for each fitted component subtracted 
#         residuals = np.zeros((1,ncomp,oldshape[1], oldshape[2]))#(nframes,ncomp,oldshape[1], oldshape[2]))
#         for i in range(1):#range(nframes):#$$$$$$$$$$$$$$$$$$$$$$
#             fluxCur = fluxUse[i]
#             for j in range(ncomp):
#                 fluxCur -= coefAry[i][j]*pcaAry[j]
#                 residuals[i][j] = fluxCur
            
        # make x and y arrays to plot
        x = np.arange(ncomp+1)
        y = []
        y.append(np.var(fluxSubbed[0][0]))
        for j in range(ncomp):
            y.append(np.var(fluxSubbed[0][j+1]))
        #print '$$$$$$$$$ '+repr(len(y))
        #print repr(y)
        #y = np.ndarray(y)
        # sorta normalize y values
        y = y/y[0]*100.0
        #print repr(y)
        
        # set up fig and plot x,y data
        fig = plt.figure(1, figsize=(20,15) ,dpi=300) 
        subPlot = fig.add_subplot(111)
        subPlot.plot(x,y)
        plt.savefig(os.path.join(outPCAdir,"pcaSubVarPlot.png"))
        

def findPSFcentersTest(inMonochrom, ncomp = 5,outputDir='',writeFiles=True):
    """
    Intial test version of the tool/prim to find the centers of the 
    PSFs in a monochromatic image.  Maybe in here we will also test the
    creation of the PCA components for those PSFs.  Later we will need to
    upgrade this to work on a series of monochromatic images for all 
    the wavelengths of an entire band or all the bands together.
    In the end, this will be part (or all) of process to develop
    a wavelength solution to be used to extract the 3D data cube from 
    a single frame of science data.
    """
    debug = True
    
    inMono = tools.loadDataAry(inMonochrom)
    
    startX = 8.5
    startY = 30.0
    
    yMax = inMono.shape[0]
    xMax = inMono.shape[1]
    xAry = np.arange(xMax)
    yAry = np.arange(yMax)
    xAry, yAry = np.meshgrid(xAry,yAry)
    
    centers = []
    
    yTop = y = startY
    xTop = x = startX
    ############################################
    # Stage 1: go up the left side of the array
    ############################################
    if debug:
        log.info("\n"+"*"*10+"   STARTING STAGE 1   "+"*"*10+"\n")
        log.debug("starting with 'top' = ["+str(yTop)+" , "+str(xTop)+"]")
    while yTop<(yMax-13.0):
        log.debug("\n\nNew 'top' = ["+str(y)+" , "+str(x)+"]")   
        ## 'run down the diagonal' till you hit the bottom
        while y>5.0:
            ## Do stuff to this PSF from its rough center 
            x = np.sum(xAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6]) 
            y = np.sum(yAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6])
            log.debug("Re-centered value = ["+str(y)+" , "+str(x)+"]\n")
            centers.append([y,x])
            ## Move to next one in this line
            if y>20:
                y -= 14.0
                x += 7.0  
                #print "proposed next center down the diagonal, ["+str(y)+" , "+str(x)+"]"
            else:
                break
        if yTop<(yMax-35.0):
            ## Hit the bottom, so back to the top and jump to next top 
            y = yTop = yTop + 34.73
            x = xTop
            #print "latest predicted 'top' = ["+str(y)+" , "+str(x)+"]"  
        else:
            #print "failed 'top' = ["+str(yTop)+" , "+str(xTop)+"]"
            break 
           
    ############################################    
    # Stage 2: go along the bottom of the array
    ############################################
    if debug:
        log.info("\n"+"*"*5+"   STARTING STAGE 2   "+"*"*5+"\n")
        log.debug("making first jump from the last 'top' found in stage 1 = ["+str(yTop)+" , "+str(xTop)+"]")
    ## Update initial rough center for next line top
    (yTop,xTop) = updatedStage2PSFtopJump(yTop,xTop,yMax,xMax,debug)
    y = yTop
    x = xTop
    if debug:
        log.debug("First new 'top' of stage 2 is = ["+str(yTop)+" , "+str(xTop)+"]")
    while xTop<(xMax-5):
        log.debug("\n\nNew 'top' = ["+str(y)+" , "+str(x)+"]")
        while (x<(xMax-5))and(y>5):
            ## Do stuff to this PSF from its rough center 
            x = np.sum(xAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6]) 
            y = np.sum(yAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6])
            log.debug("re-centered value = ["+str(y)+" , "+str(x)+"]")
            centers.append([y,x])
            if x<(xMax-9):
                ## Move to next one in this line
                y -= 14.0
                x += 7.0
                #print "proposed next center down the diagonal, ["+str(y)+" , "+str(x)+"]"
            else:
                break
        ##  Hit the right wall, so back to the top and jump to next top 
        (yTop,xTop) = updatedStage2PSFtopJump(yTop,xTop,yMax,xMax,False)
        y = yTop
        x = xTop
        #print "proposed next top, ["+str(y)+" , "+str(x)+"]"
            
    if True:
        f = open(os.path.join(outputDir,'originalPSFcenters.txt'), 'w')
        for i in range(0,len(centers)):
            s = "PSF # "+str(i+1)+" = "+repr(centers[i])
            f.write(s+"\n")
            #print s
        f.close()
    
    ########################################################################
    # Re-center PSFs in an iterative loop
    ########################################################################
    centersUpdated = centers
    if True:
        centersUpdated = refinePSFcentersTest(inMono,xAry,yAry,centers)
    
    if False:
        for i in range(0,50):#len(centersUpdated)+1):
            log.debug("PSF # "+str(i+1)+" = "+repr(centersUpdated[i]))
      
    #########################################################################
    # Extract centered and cropped 13x13 PSFs, stack and perform PCA on them.
    #########################################################################
    log.info("*"*10+"   Starting to extract 13x13pix PSFs, stack and perform PCA on them   "+"*"*10)
    psfStack = []
    numAdded = 0
    numNotAdded = 0
    for i in range(0,len(centersUpdated)):
        y = centersUpdated[i][0]
        x = centersUpdated[i][1]
        if ((x>(xMax-6))or(y>(yMax-6)))or((x<6)or(y<6)):
            numNotAdded += 1
            log.debug("This PSF has insufficient surrounding pixels to be cropped to 13x13, center = ["+str(y)+" , "+str(x)+"]")
        else:
            numAdded += 1
            psfStack.append(ndimage.map_coordinates(inMono,[yAry[y-6:y+7,x-6:x+7],xAry[y-6:y+7,x-6:x+7]],order=3))
    psfStack = np.array(psfStack)
    log.debug( "numAdded = "+str(numAdded)+", numNotAdded = "+str(numNotAdded))
    log.debug( "Shape of psfStack cropping and stacking = "+repr(psfStack.shape))    
    
    nPSFs = psfStack.shape[0]
    oldshape = psfStack.shape
    log.debug("input array had shape: "+repr(oldshape))
    
    # convert to a 2D array [nframes,flattened 2d original data array to 1D]
    stackFlattened = np.reshape(psfStack, (nPSFs, -1))
    log.debug("reshaped array is now: "+repr(stackFlattened.shape))
    
    # First step: subtract the mean
    stackMean = np.mean(stackFlattened, axis=0)
    for i in range(nPSFs):
        stackFlattened[i] -= stackMean
    
    # super basic version of SVD.  ACORNS uses a much more advances version with multiprocessing we will implement later
    log.debug("about to try np.linalg.svd")
    u, s, V = np.linalg.svd(stackFlattened.T,full_matrices=False)
    uNew = u.T[:ncomp]  # These are the top components from the PCA/SVD decomposition, but need to be reshaped to de-flatten
    log.debug("output uNew has shape: "+repr(uNew.shape))
    
    # de-flatten U array to make PCA component array
    pcaAry = np.reshape(uNew, (ncomp, oldshape[1], oldshape[2]))
    log.debug("pcaAry has shape: "+repr(pcaAry.shape))
    pcaAryFlattened = uNew
    
    # save PCA components as individual fits files
    outPCAdir= os.path.join(outputDirRoot,"pcaOutputs")
    if writeFiles:
        os.mkdir(outPCAdir)
        for j in range(ncomp):
            hdu = pf.PrimaryHDU(pcaAry[j])
            hduList = pf.HDUList([hdu])
            outFilename = os.path.join(outputDirRoot+"/pcaOutputs/","pcaComponent_"+str(j+1)+".fits")
            hduList.writeto(outFilename)
            hduList.close()    
    
    #return centersUpdated
    
    
def updatedStage2PSFtopJump(yTop,xTop,yMax,xMax,debug=False):
    if yTop<=(yMax-16.0):#(xTop<(xMax-15))and(yTop>16.0):
        #if debug:
        #    print "yTop>16 so subtracting 7, value was = "+str(yTop)
        yTop += 7.0
        xTop +=14.0
    elif yTop>(yMax-16.0):#(xTop<(xMax-22))and(yTop<9.0):
        #if debug:
        #    print "yTop<16 so adding 7, value was = "+str(yTop)
        yTop -= 7.0
        xTop += 20.7
    else:
        log.warning( "Not in either of the 'top' ranges for stage 2, yTop = "+str(yTop)+\
                     ". yTop<=(yMax-16.0) = "+repr(yTop<=(yMax-16.0))+", yTop>=16.0 = "+repr(yTop>(yMax-16.0)))
    return (yTop,xTop)
    
def refinePSFcentersTest(inMono,xAry,yAry, centers):
    """
    A function to loop over all the centers found in the findPSFcentersTest func in a loop to refine 
    them further using the centerOfLight func.  This will be done iteratively until convergence.
    NOTE: Still not sure yet how to merge these two functions, or if that is even a good idea...
    """
    meanDiff = 10.0
    debug = True
    centersLast = centers
    iteration = 0
    log.debug("*"*5+"  Performing iterative PSF centering loop  "+"*"*5)
    while meanDiff>0.01:
        iteration+=1
        if debug:
            log.debug("\n Starting iteration = "+str(iteration))
        centersUpdated = []
        for i in range(0,len(centersLast)):
            y = centersLast[i][0]
            x = centersLast[i][1]
            try:
                 ## Do stuff to this PSF from its rough center 
                if ((x<(inMono.shape[1]-6))or(y<(inMono.shape[0]-6)))or((x>6)or(y>6)):
                    x = np.sum(xAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6]) 
                    y = np.sum(yAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6])
                elif ((x>(inMono.shape[1]-6))or(y>(inMono.shape[0]-6)))or((x<6)or(y<6)):
                     ## PSF is too close to the edge, so shrink the box
                    x = np.sum(xAry[y-2:y+3,x-2:x+3]*inMono[y-2:y+3,x-2:x+3])/np.sum(inMono[y-2:y+3,x-2:x+3]) 
                    y = np.sum(yAry[y-2:y+3,x-2:x+3]*inMono[y-2:y+3,x-2:x+3])/np.sum(inMono[y-2:y+3,x-2:x+3])
                else: 
                    log.warning("Center coords did not match either box size.  Its [y,x] were = ["+str(y)+" , "+str(x)+"]")
                centersUpdated.append([y,x])
            except:
                log.error("an error occurred while trying to re-center a PSF.  Its [y,x] were = ["+str(y)+" , "+str(x)+"]")
                break
        meanDiff = abs(np.mean(centersLast)-np.mean(centersUpdated))
        #if debug:
        #    print "meanDiff = "+str(meanDiff)
        centersLast = centersUpdated
    log.info("Finished PSF re-centering loop in "+str(iteration)+" iterations, resulting in a mean difference of "+str(meanDiff)+"\n")
    
    return centersUpdated
