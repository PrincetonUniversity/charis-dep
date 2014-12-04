import tools
import os
import re
import shutil
import timeit
import pyfits as pf
import numpy as np
from scipy import ndimage
from scipy import optimize
from scipy import linalg
from progressbar import ProgressBar
import copy
import warnings
import pylab
plt = pylab.matplotlib.pyplot

log = tools.getLogger('main.prims',lvl=100,addFH=False)#('main.prims',lvl=100,addFH=False)

def pcaTest(flux, ncomp=5, outputDirRoot='.', writeFiles=True):
    """
    This function/prim will take any set of equally sized sequence of images with at least
    5 frames of a point source and extract its top (5) principle components using PCA.
    
    NOTE: This code is basically a test of concept and shall be used primarily as sample
    code for further applications during the wavelength extraction and wavelength solution 
    primitives.
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
        

def findPSFcentersTest(inMonochrom, ncomp = 20,outputDir='',writeFiles=True):
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
    iterationsCOL = []
    yTop = y = startY
    xTop = x = startX
    ############################################
    # Stage 1: go up the left side of the array
    ############################################
    if debug:
        log.info("\n"+"*"*10+"   STARTING STAGE 1   "+"*"*10+"\n")
        log.debug("starting with 'top' = ["+str(yTop)+" , "+str(xTop)+"]")
    while yTop<(yMax-13.0):
        #log.debug("\n\nNew 'top' = ["+str(y)+" , "+str(x)+"]")   
        ## 'run down the diagonal' till you hit the bottom
        while y>5.0:
            ## Do stuff to this PSF from its rough center 
            xDiff = yDiff = 10.0
            yPrev = y
            xPrev = x
            iterationCOL = 1
            while (xDiff>0.05)and(yDiff>0.05):
                ## Do stuff to this PSF from its rough center 
                x = np.sum(xAry[yPrev-5:yPrev+6,xPrev-5:xPrev+6]*inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6])/np.sum(inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6]) 
                y = np.sum(yAry[yPrev-5:yPrev+6,xPrev-5:xPrev+6]*inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6])/np.sum(inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6])
                xDiff = abs(x-xPrev)
                yDiff = abs(y-yPrev)
                yPrev = y
                xPrev = x
                iterationCOL+=1
                if iterationCOL>5:
                    print "!!!!!!! iterationCOL>5 !!!!!!!"
            #log.debug("Re-centered value = ["+str(y)+" , "+str(x)+"]\n")
            centers.append([y,x])
            iterationsCOL.append(iterationCOL)
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
        #log.debug("\n\nNew 'top' = ["+str(y)+" , "+str(x)+"]")
        while (x<(xMax-5))and(y>5):
            xDiff = yDiff = 10.0
            yPrev = y
            xPrev = x
            iterationCOL = 1
            while (xDiff>0.05)and(yDiff>0.05):
                ## Do stuff to this PSF from its rough center 
                x = np.sum(xAry[yPrev-5:yPrev+6,xPrev-5:xPrev+6]*inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6])/np.sum(inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6]) 
                y = np.sum(yAry[yPrev-5:yPrev+6,xPrev-5:xPrev+6]*inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6])/np.sum(inMono[yPrev-5:yPrev+6,xPrev-5:xPrev+6])
                xDiff = abs(x-xPrev)
                yDiff = abs(y-yPrev)
                yPrev = y
                xPrev = x
                iterationCOL+=1
                if iterationCOL>5:
                    print "!!!!!!! iterationCOL>5 !!!!!!!"
            #log.debug("re-centered value = ["+str(y)+" , "+str(x)+"]")
            centers.append([y,x])
            iterationsCOL.append(iterationCOL)
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

    #print 'iterationsCOL = '+repr(iterationsCOL)#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    ########################################################################
    # Re-center PSFs in an iterative loop
    # First set-up initial, or constant, variables/values, then enter the loop.
    ########################################################################
    centersUpdated = centers
    if False:
        centersUpdated = refinePSFcentersTest(inMono,xAry,yAry,centers)
    
    meanDiff = np.Inf
    nHiRes = 9.0
    centersLast = centersUpdated
    chi2BestsTotAry = []
    inMonoCorrected = inMono
    np.putmask(inMonoCorrected,np.isnan(inMonoCorrected),0.0)
    yAryHiRes = np.linspace(-6.0,+6.0,(12*nHiRes+1))
    xAryHiRes = yAryHiRes
    iteration = 0
    log.info("*"*10+"   Starting to extract PCA comps and use them to re-center in an iterative loop   "+"*"*10)
    while (meanDiff>0.00003)and(iteration<7):
        #########################################################################
        # Extract centered and cropped 13x13 PSFs, stack and perform PCA on them.
        #########################################################################
        # record the time PCA extraction started
        tic1 = timeit.default_timer()
        iteration+=1
        if debug:
            print "\n\n\n"+"#"*75
        log.debug(" Starting iteration = "+str(iteration))
        if debug:
            print "#"*75+"\n\n\n"
        log.info("*"*10+"   Starting to extract 13x13pix PSFs, stack and perform PCA on them   "+"*"*10)
        psfStack = []
        numAdded = 0
        numNotAdded = 0
        for i in range(0,len(centersLast)):
            currPSFary = np.zeros((13,13))
            yCentInit = centersLast[i][0]
            xCentInit = centersLast[i][1]
            if ((xCentInit>(xMax-6))or(yCentInit>(yMax-6)))or((xCentInit<6)or(yCentInit<6)):
                numNotAdded += 1
                log.debug("This PSF has insufficient surrounding pixels to be cropped to 13x13, center = ["+str(y)+" , "+str(x)+"]")
            else:
                numAdded += 1
                yAry2 = yAryHiRes+yCentInit
                xAry2 = xAryHiRes+xCentInit
                xAry2,yAry2 = np.meshgrid(xAry2,yAry2)
                currPSFary = ndimage.map_coordinates(inMonoCorrected,[yAry2,xAry2],order=3)
                psfStack.append(currPSFary)
            if np.isnan(np.sum(currPSFary)):
                log.error("currPSFary's sum was NaN!")
                log.error("\n\ncurrPSFary = "+repr(currPSFary))
        psfStack = np.array(psfStack)
        log.debug("numAdded = "+str(numAdded)+", numNotAdded = "+str(numNotAdded))
        log.debug("Shape of psfStack cropping and stacking = "+repr(psfStack.shape))    
        
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
        
        log.debug("about to create PCA comps")
        # super basic version of SVD.  ACORNS uses a much more advances version with multiprocessing we will implement later
        u, s, V = linalg.svd(stackFlattened.T,full_matrices=False)
        uNew = u.T[:ncomp]  # These are the top components from the PCA/SVD decomposition, but need to be reshaped to de-flatten
        log.debug("output uNew has shape: "+repr(uNew.shape))
        
        # de-flatten U array to make PCA component array
        pcaAry = np.reshape(uNew, (ncomp, oldshape[1], oldshape[2]))
        # de-flatten mean
        stackMeanUnflat = np.reshape(stackMean, (oldshape[1], oldshape[2]))
        log.debug("pcaAry has shape: "+repr(pcaAry.shape))
        pcaAryFlattened = uNew
        
        # save PCA components as individual fits files
        outPCAdir= os.path.join(outputDir,"pcaOutputs")
        if writeFiles:
            if os.path.exists(outPCAdir):
                log.info("outPCAdir already exists, so just adding these files with a different 'iterations' str post-pended.")
            #    shutil.rmtree(outPCAdir)
            else:
                os.mkdir(outPCAdir)
            # store mean first
            hdu = pf.PrimaryHDU(stackMeanUnflat)
            hduList = pf.HDUList([hdu])
            outFilename = os.path.join(outPCAdir,"pcaComponent_mean"+str(iteration)+".fits")
            hduList.writeto(outFilename)
            hduList.close()
            for j in range(ncomp):
                hdu = pf.PrimaryHDU(pcaAry[j])
                hduList = pf.HDUList([hdu])
                outFilename = os.path.join(outPCAdir,"pcaComponent_"+str(j+1)+"-"+str(iteration)+".fits")
                hduList.writeto(outFilename)
                hduList.close()    
        # Push mean in as zeroth PCA comp
        pcaAry2 = []
        pcaAry2.append(stackMeanUnflat)
        for j in range(ncomp):
            pcaAry2.append(pcaAry[j])
        pcaAry2 = np.array(pcaAry2)
        
        # write total elapsed time to screen and log.
        toc=timeit.default_timer()
        totalTimeString = tools.timeString(toc - tic1)
        log.debug('\n\nPCA comp extraction took '+totalTimeString+' to complete.\n')
        ###################################################################################################################
        # PCA was done using 13x13 arrays.  First the central 9x9 of the low/standard resolution PSF will be cropped out.
        # Then the high resolution PCA components will have their central 9x9pix cropped out, binned down to 1pix resolution
        # and fit to the PSF.  It will be shifted around the center using a 5x5 box at 0.1pix resolution until the best new 
        # center is found based on the chi squared of the fit.  These fit values are integer based, so they are then fit 
        # again using a secondary stage of least square fitting to find the sub-pixel resolution center.
        ###################################################################################################################
        nref = 4
        nPixSteps = 1.0
        # create a progress bar object for updates to user of progress
        p = ProgressBar('red',width=30,block='=',empty='-',lastblock='>')
        ## Create the stepping array for moving around the center
        x = np.arange(int(-9.0*nPixSteps),int(9.0*nPixSteps+1))#$$$$$$$$$$$$$$$$$ make this size a free parameter and make sure to understand it fully
        stepBoxWidth = x.shape[0]
        xStepAry, yStepAry = np.meshgrid(x, x) 
        centersUpdated2 = []
        centersUpdated3 = []
        offsetsBest = []
        chi2Bests = []
        chi2Best2s = []
        recentSuccess = []
        # record the time PCA extraction started
        tic2=timeit.default_timer()
        log.info("*"*10+"   Starting to perform PCA based re-centering   "+"*"*10)
        ## Loop over each PSF center in the updated centers array produced by the center of light approach
        for center in range(0,len(centersLast)):
            #print '\ncenter #'+str(center)
            chi2Best = np.inf
            iBest = jBest = -1
            chi2 = np.zeros((stepBoxWidth, stepBoxWidth))
            ybest, xbest = [0, 0]
            y1Orig = y1 = centersLast[center][0]
            x1Orig = x1 = centersLast[center][1]
            y1FracShift = y1%1
            x1FracShift = x1%1
            y1 = int(y1)
            x1 = int(x1)
            # check if fraction is over 0.5, if so shift 1 whole pix toward bottom left corner
            if y1FracShift>=0.5:
                y1 +=1
                y1FracShift -= 1.0
            if x1FracShift>=0.5:
                x1 += 1
                x1FracShift -= 1.0
            # crop 9x9 PSF and flatten.  This Ary is in 1pix resolution
            currPSFflat = np.reshape(inMonoCorrected[y1-4:y1+5,x1-4:x1+5],-1)
            #yBestStr = xBestStr = ''
            # create a progress bar object for updates to user of progress
            #p2 = ProgressBar('green',width=30,block='=',empty='-',lastblock='>')
            ## Loop over each shifted center point to find best new center based on lowest chi squared value
            for i in range(stepBoxWidth):
                for j in range(stepBoxWidth):              
                    ## My version, Not sure if correct as don't know why multiplication by std was used in Tim's version
                    pcaCompsUSE = np.zeros((nref,currPSFflat.shape[0]))
                    for k in range(nref):
                        # crop 9x9 high resolution PCA components, rebin and flatten
                        ##NOTE: shifting stepping array to left/down, which will shift the PCA comps to the left/down
                        yShift = yStepAry[i, j]-int(y1FracShift*9.0)
                        xShift = xStepAry[i, j]-int(x1FracShift*9.0)
                        pcaCropped = pcaAry2[k][14+yShift:-14+yShift,14+xShift:-14+xShift]
                        pcaBinned = tools.rebin(pcaCropped,(9,9))
                        pcaCompsUSE[k] = np.reshape(pcaBinned,-1)
                    A = pcaCompsUSE
                    b = currPSFflat #1pix resolution 9x9 array centered on integer (1pix res) center from COL
                    coef = linalg.lstsq(A.T, b)[0]
             
                    # Compute residuals, sum to get chi2
                    resid = currPSFflat - coef[0] * pcaCompsUSE[0]
                    for k in range(1, nref):
                        resid -= coef[k] * pcaCompsUSE[k]
                    chi2[i, j] = np.sum(resid**2)
                    
                    if chi2[i, j] < chi2Best:
                        chi2Best = chi2[i, j]
                        iBest, jBest = [i, j]
                #p2.render((i+1) * 100 // stepBoxWidth, ' of i vals complete so far.')
                
            chi2Bests.append(chi2Best)
            offsetsBest.append([yStepAry[iBest,jBest],xStepAry[iBest,jBest]])
            ##NOTE: PCA shifted and fit outputs are shifted to left/down by FracShift, thus shift back to the right/up
            centersUpdated2.append([y1+(1.0/9.0)*int(y1FracShift*9.0)+(yStepAry[iBest,jBest]/9.0),x1+(1.0/9.0)*int(x1FracShift*9.0)+(xStepAry[iBest,jBest]/9.0)])
            
            ################################################################################################################################################ 
            # Use a least squares fit to a parabola of (chi2,x,y) to find true x,y shift values instead of integer based ones above.  Only using central 3x3
            ################################################################################################################################################
            success = True
            initGuess = []
            preParaSummaryStr =  "\n"+"-"*20+" Iteration #"+str(iteration)+", PSF #"+str(center)+"   "+"-"*20+\
                              "\n[iBest,jBest] = "+"["+str(iBest)+", "+str(jBest)+"]"+\
                              " -> ("+str(yStepAry[iBest,jBest])+", "+str(xStepAry[iBest,jBest])+")"+\
                              "\nSize of inMono [yMax,xMax] = ["+str(yMax)+", "+str(xMax)+"]"+\
                              "\nInitial COL center = "+repr(centersUpdated[center])+\
                              "\nPrevious center = "+repr(centersLast[center])+\
                              "\nLatest 1/9 resolution PCA based center = "+repr(centersUpdated2[center])+\
                              "\ny1 = "+str(y1)+", x1 = "+str(x1)+"\n"+\
                              "\nyFracShift = "+str(y1FracShift)+", xFracShift = "+str(x1FracShift)+"\n"
            paraSummaryStr = ""
            try:
                yPara = np.reshape(yStepAry[iBest-1:iBest+2,jBest-1:jBest+2],-1)
                xPara = np.reshape(xStepAry[iBest-1:iBest+2,jBest-1:jBest+2],-1)
                initGuess = [2.0, 2.0, 2.0, chi2Best, 0.0, 0.0]
                chi2Para = chi2[iBest-1:iBest+2,jBest-1:jBest+2]
                chi2Para = np.reshape(chi2Para,-1)
                paraSummaryStr =  "\nyPara.shape[0] = "+repr(yPara.shape[0])+\
                                "\nyStepAry = "+repr(yPara)+\
                                "\nxStepAry = "+repr(xPara)+\
                                "\ninitGuess = "+repr(initGuess)+"\n"
                
                if (yPara.shape[0]<=3) or (xPara.shape[0]<=3):
                    success = False
                    sideStr = ""
                    if centersUpdated2[center][1]>(xMax-3):
                        sideStr = "\ncenter is too close to right side!!\n"
                    if centersUpdated2[center][1]<3:
                        sideStr = sideStr+"\ncenter is too close to left side!!\n"
                    if centersUpdated2[center][0]>(yMax-3):
                        sideStr = sideStr+"\ncenter is too close to top!!\n"
                    if centersUpdated2[center][0]<3:
                        sideStr = sideStr+"\ncenter is too close to bottom!\n"
                    if sideStr=="":
                        sideStr = "Center of this PSF was not near a side or the array.\n"
                    log.error("\n** Stepping array size under 3x3 **"+preParaSummaryStr+paraSummaryStr+sideStr+"-"*50+"\n")                         
                else:
                    #print "about to  call optimize"
                    #print "xPara.shape = "+str(xPara.shape)+", yPara.shape = "+str(yPara.shape)+", chi2Para.shape = "+str(chi2Para.shape)
                    bestFitVals,s = optimize.leastsq(residual, initGuess[:], args=(xPara,yPara,chi2Para))
                    chi2Best2s.append(bestFitVals[3])
                    yBestOut = bestFitVals[4]
                    xBestOut = bestFitVals[5]
            except:
                log.error("\nAn error occurred while trying to refine best center from PCA re-centering"+preParaSummaryStr+paraSummaryStr+"-"*50+"\n")

            recentSuccess.append(success)
            centersUpdated3.append([centersUpdated2[center][0]+(yBestOut/9.0)-(yStepAry[iBest,jBest]/9.0),centersUpdated2[center][1]+(xBestOut/9.0)-(xStepAry[iBest,jBest]/9.0)])
            
            p.render((center+1) * 100 // len(centersLast), 'Centers complete so far.')
            if center==50: #$$$$$$$$$$$$$$
                #monoAry501 = inMonoCorrected[centersUpdated[center][0]-4:centersUpdated[center][0]+5,centersUpdated[center][1]-4:centersUpdated[center][1]+5]
                chi2Ary502 = chi2 #$$$$$$$$$$$$$$
                chi2Best502 = chi2[iBest,jBest]
                yBest502 = xStepAry[iBest,jBest]
                xBest502 = yStepAry[iBest,jBest]
                yPara50 = yPara
                xPara50 = xPara
                chi2Para50 = chi2Para
                yBestPara50 = yBestOut
                xBestPara50 = xBestOut
                chi2BestPara50 = bestFitVals[3]
                #monoAry502 = inMonoCorrected[centersUpdated2[center][0]-4:centersUpdated2[center][0]+5,centersUpdated2[center][1]-4:centersUpdated2[center][1]+5]
                #monoAry503 = inMonoCorrected[centersUpdated3[center][0]-4:centersUpdated3[center][0]+5,centersUpdated3[center][1]-4:centersUpdated3[center][1]+5]
            elif center==1000:#$$$$$$$$$$$$$$
                chi2Ary10002 = chi2#$$$$$$$$$$$$$$
                chi2Best10002 = chi2[iBest,jBest]
                yBest10002 = xStepAry[iBest,jBest]
                xBest10002 = yStepAry[iBest,jBest]
                yPara1000 = yPara
                xPara1000 = xPara
                chi2Para1000 = chi2Para
                yBestPara1000 = yBestOut
                xBestPara1000 = xBestOut
                chi2BestPara1000 = bestFitVals[3]
        # write total elapsed time to screen and log for both.
        toc=timeit.default_timer()
        totalTimeString = tools.timeString(toc - tic2)
        log.debug('\n\nPCA-based re-centering took '+totalTimeString+' to complete.\n')
        totalTimeString = tools.timeString(toc - tic1)
        log.info("\nIteration "+str(iteration)+" of PCA extraction and re-centering took "+totalTimeString+' to complete.\n')
        
        ########################################################################################
        # Calculate mean difference to determine if loop can exit, and update 'centersLast'
        # with most recently found centers (ie. centersUpdated3).
        # NOTE: re-centering iterative loop not set up yet!!!!
        ########################################################################################
        meanDiff1 = abs(np.mean(centersLast)-np.mean(centersUpdated3))
        diffAryY = []
        diffAryX = []
        for c in range(0,len(centersLast)):
            diffY = centersLast[c][0]-centersUpdated3[c][0]
            diffX = centersLast[c][1]-centersUpdated3[c][1]
            diffAryY.append(diffY)
            diffAryX.append(diffX)
        meanDiffY = np.mean(diffAryY)
        meanDiffX = np.mean(diffAryX)
        meanDiff = abs(np.mean([meanDiffY,meanDiffX]))
        
        chi2BestsTotAry.append(chi2Best2s)
        
        log.debug("PSF #50: Original center = "+repr(centersUpdated[50])+", newest ones are = "+repr(centersUpdated3[50]))
        print "-"*75+"\n"+"\nchi2Ary502:\n"+tools.arrayRepr(chi2Ary502)+"\nBest pre-Para: y = "+str(yBest502)+", x = "+str(xBest502)+", chi2 = "+str(chi2Best502)+"\n"+\
                "\nchi2Para50:\n"+tools.arrayRepr(chi2Para50)+"\nyPara50:\n"+tools.arrayRepr(yPara50)+"\nxPara50:\n"+tools.arrayRepr(xPara50)+"\n"+\
            "\nyBestPara50 = "+str(yBestPara50)+"\nxBestPara50 = "+str(xBestPara50)+"\nchi2BestPara50 = "+str(chi2BestPara50)+"\n"+"-"*75
        
        log.debug("PSF #1000: Original center = "+repr(centersUpdated[1000])+", newest ones are = "+repr(centersUpdated3[1000]))
        print "-"*75+"\n"+"\nchi2Ary10002\n"+tools.arrayRepr(chi2Ary10002)+"\nBest pre-Para: y = "+str(yBest10002)+", x = "+str(xBest10002)+", chi2 = "+str(chi2Best10002)+"\n"+\
            "\nchi2Para1000:\n"+tools.arrayRepr(chi2Para1000)+"\nyPara1000:\n"+tools.arrayRepr(yPara1000)+"\nxPara10000:\n"+tools.arrayRepr(xPara1000)+"\n"+\
            "\nyBestPara1000 = "+str(yBestPara1000)+"\nxBestPara1000 = "+str(xBestPara1000)+"\nchi2BestPara1000 = "+str(chi2BestPara1000)+"\n"+"-"*75
        centersLast = centersUpdated3 
        log.info("Finished iteration "+str(iteration)+" of PCA-based re-centering resulting in a total mean difference of "+str(meanDiff)+\
                 "\n meanDiff1 = "+str(meanDiff1)+", meanDiffY = "+str(meanDiffY)+", meanDiffX = "+str(meanDiffX)+"\n")
    log.info("Finished PCA-based re-centering loop in "+str(iteration)+" iterations\n")
    
    if False:
        ## write centers from each stage
        f = open(os.path.join(outputDir,'multiStageCentOuts.txt'), 'w')
        f.write("PCA#  Ycl       Xcl     Ypca1   Xpca1   chi2pca1    Ypca2   Xpca2   chi2pca2\n")
        for i in range(0,len(centersUpdated3)):
            s = "%.1f   %.3f   %.3f   %.3f   %.3f  %.5f     %.3f   %.3f   %.5f "%(i,centersUpdated[i][0],centersUpdated[i][1],centersUpdated2[i][0],centersUpdated2[i][1],chi2Bests[i],centersUpdated3[i][0],centersUpdated3[i][1],chi2Best2s[i])
            f.write(s+"\n")
            #print s
        f.close()
    if True:
        ## write chi2s to file
        f = open(os.path.join(outputDir,'iterativePSFcenterChi2s.txt'),'w')
        for i in range(0,len(chi2BestsTotAry)):
            for j in range(0,len(chi2BestsTotAry[0])):
                f.write(str(chi2BestsTotAry[i][j])+"\n")
            f.write("\n")
        f.close()
        
def residual(p,x,y,chi2):
    #print "inside residual"
    #print "a,b,c,d,yc,xc = "+repr(p)
    a,b,c,d,xc,yc = p 
    val = chi2 - (a*(x-xc)**2.0+b*(y-yc)**2.0+c*(x-xc)*(y-yc)+d)
    #print repr(val)
    return val
    
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
    log.debug("*"*5+"  Performing iterative PSF centering with C.O.L. loop  "+"*"*5)
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
