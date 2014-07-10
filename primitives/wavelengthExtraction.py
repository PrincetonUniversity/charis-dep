import tools
import os
import re
import pyfits as pf
import numpy as np
import scipy as sp
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
    
    # super basic version of SVD.  ACORNS uses a much more advances version with multiprocessing we will implement later
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
        print '$$$$$$$$$ '+repr(len(y))
        print repr(y)
        #y = np.ndarray(y)
        # sorta normalize y values
        y = y/y[0]*100.0
        print repr(y)
        
        # set up fig and plot x,y data
        fig = plt.figure(1, figsize=(20,15) ,dpi=300) 
        subPlot = fig.add_subplot(111)
        subPlot.plot(x,y)
        plt.savefig(os.path.join(outPCAdir,"pcaSubVarPlot.png"))
        

def findPSFcentersTest(inMonochrom, outputDir='',writeFiles=True):
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
    
    startX = 9.0
    startY = 784.0
    
    xMax = inMono.shape[0]
    yMax = inMono.shape[1]
    
    centers = []
    
    yTop = y = startY
    xTop = x = startX
    odd = False
    # Stage 1: go up the left side of the array
    if debug:
        print "starting stage 1"
        print "starting with 'top' = ["+str(xTop)+" , "+str(yTop)+"]"
    while yTop>34.0:
        if (y==startY) and (x==startX):
            # first PSF, so centering box is bigger
            (expectationX,expectationY) = centerOfLight(inMono[x-5:x+6,y-5:y+6], x, y,11)
            centers.append([expectationX+x,expectationY+y])
            y = yTop = expectationY+y
            #x = expectationX+x
            if debug:
                print "re-centered 'top' = ["+str(xTop)+" , "+str(yTop)+"]\n"
        elif (y==yTop)and(x==xTop):
            # Re-center this 'top'
            (expectationX,expectationY) = centerOfLight(inMono[x-2:x+3,y-2:y+3], x, y)
            centers.append([expectationX+x,expectationY+y])
            y = yTop = expectationY+y
            #x = xTop = expectationX+x
            if debug:
                print "re-centered 'top' = ["+str(xTop)+" , "+str(yTop)+"]\n"
        while y<(yMax-14.0):
            # Do stuff to this PSF from its rough center 
            (expectationX,expectationY) = centerOfLight(inMono[x-2:x+3,y-2:y+3], x, y)
            centers.append([expectationX+x,expectationY+y])
            y = expectationY+y
            x = expectationX+x
            # Move to next one in this line
            y = y + 14.0
            x = x + 7.0
        # Update rough center for next line top
        yTop = yTop - 35.0
        y = yTop
        x = xTop
        if debug:
            print "New 'top' = ["+str(x)+" , "+str(y)+"]"
        
    # Stage 2: go along the bottom of the array
    if debug:
        print "\nstarting stage 2"
        print "making first jump from the last 'top' found in stage 1"
    # Update initial rough center for next line top
    (xTop,yTop) = updatedStage2PSFtopJump(xTop,yTop,debug)
    x = xTop
    y = yTop
    if debug:
        print "First new 'top' of stage 2 is = ["+str(xTop)+" , "+str(yTop)+"]"
    while xTop<(xMax-8):
        if ((y==yTop)and(x==xTop)):
            # Re-center this 'top'
            (expectationX,expectationY) = centerOfLight(inMono[x-2:x+3,y-2:y+3], x, y)
            centers.append([expectationX+x,expectationY+y])
            y = yTop = expectationY+y
            x = xTop = expectationX+x
            if debug:
                print "re-centered 'top' = ["+str(xTop)+" , "+str(yTop)+"]"
        while x<(xMax-15):
            # Do stuff to this PSF from its center 
            (expectationX,expectationY) = centerOfLight(inMono[x-2:x+3,y-2:y+3], x, y)
            centers.append([expectationX+x,expectationY+y])
            # Update rough center for this line
            x = x + 14.0 + expectationX
            y = y + 7.0 + expectationY
        # Update rough center for next line top
        (xTop,yTop) = updatedStage2PSFtopJump(xTop,yTop)
        x = xTop
        y = yTop
        if debug:
            print "\nNew 'top' = ["+str(x)+" , "+str(y)+"]"
            
            
    if True:
        f = open(os.path.join(outputDir,'originalPSFcenters.txt'), 'w')
        for i in range(0,len(centers)):
            s = "PSF # "+str(i+1)+" = "+repr(centers[i])
            f.write(s+"\n")
            #print s
        f.close()
    
    centersUpdated = centers
    if False:
        centersUpdated = refinePSFcentersTest(inMono,centers)
    
    if False:
        for i in range(0,50):#len(centersUpdated)+1):
            print "PSF # "+str(i+1)+" = "+repr(centersUpdated[i])
            
    return centersUpdated
    
def updatedStage2PSFtopJump(xTop,yTop,debug=False):
    if yTop>=16.0:#(xTop<(xMax-15))and(yTop>16.0):
        if debug:
            print "yTop>16 so subtracting 7, value was = "+str(yTop)
        yTop = yTop - 7.0
        xTop = xTop + 14.0
    elif yTop<16.0:#(xTop<(xMax-22))and(yTop<9.0):
        if debug:
            print "yTop<16 so adding 7, value was = "+str(yTop)
        yTop = yTop + 7.0
        xTop = xTop + 21.0
    else:
        print "Not in either of the 'top' ranges for stage 2, yTop = "+str(yTop)+". yTop<16.0 = "+repr(yTop<16.0)+", yTop>=16.0 = "+repr(yTop>=16.0)
    
    return (xTop,yTop)
    
def refinePSFcentersTest(inMono, centers):
    """
    A function to loop over all the centers found in the findPSFcentersTest func in a loop to refine 
    them further using the centerOfLight func.  This will be done iteratively until convergence.
    NOTE: Still not sure yet how to merge these two functions, or if that is even a good idea...
    """
    meanDiff = 10.0
    debug = True
    centersLast = centers
    iteration = 1
    while meanDiff>0.05:
        if debug:
            print "iteration = "+str(iteration)
        centersUpdated = []
        for i in range(0,len(centersLast)):
            x = centersLast[i][0]
            y = centersLast[i][1]
            try:
                (expectationX,expectationY) = centerOfLight(inMono[x-2:x+3,y-2:y+3], x, y)
                centersUpdated.append([expectationX+x,expectationY+y])
            except:
                print "an error occurred while trying to re-center a PSF.  Its [x,y] were = ["+str(x)+" , "+str(y)+"]"
                break
        meanDiff = abs(np.mean(centersLast)-np.mean(centersUpdated))
        if debug:
            print "meanDiff = "+str(meanDiff)
        iteration+=1
        centersLast = centersUpdated
    
    return centersUpdated

def centerOfLight(subArray, xCent, yCent, width=5):
    """
    Find the center of light in a 5x5 box around the current approximated center.
    
    NOTE: currently only the choices of 5 and 11 are available for the width.
    """
    # Make a sub array of 5x5
    Is = subArray
    # Make Xs and Ys arrays
    if width==5.0:
        Xs = [-2.0,-1.0,0.0,1.0,2.0]
        Xs = [Xs,Xs,Xs,Xs,Xs]
    if width==11.0:
        Xs = [-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0]
        Xs = [Xs,Xs,Xs,Xs,Xs,Xs,Xs,Xs,Xs,Xs,Xs]
    Ys = np.array(Xs)
    Ys = Ys.T
    
    # calculate center of light expectation value
    expectationX = np.sum(Xs*Is)/np.sum(Is) # where Xs are the X locations within a box around a PSF
    expectationY = np.sum(Ys*Is)/np.sum(Is)
    
    return (expectationX,expectationY)