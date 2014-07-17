import tools
import os
import re
import pyfits as pf
import numpy as np
from scipy import ndimage
from scipy import optimize
import copy
import warnings
import pylab
plt = pylab.matplotlib.pyplot

log = tools.getLogger('main.prims',lvl=0,addFH=False)#('main.prims',lvl=100,addFH=False)

def pcaTest(flux, ncomp=5, outputDirRoot='.', writeFiles=True):
    """
    This function/prim will take any set of equally sized sequence of images with at least
    5 frames of a point source and extract its top 5 principle components using PCA.
    
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
            x = np.sum(xAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6]) 
            y = np.sum(yAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6])
            #log.debug("Re-centered value = ["+str(y)+" , "+str(x)+"]\n")
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
        #log.debug("\n\nNew 'top' = ["+str(y)+" , "+str(x)+"]")
        while (x<(xMax-5))and(y>5):
            ## Do stuff to this PSF from its rough center 
            x = np.sum(xAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6]) 
            y = np.sum(yAry[y-5:y+6,x-5:x+6]*inMono[y-5:y+6,x-5:x+6])/np.sum(inMono[y-5:y+6,x-5:x+6])
            #log.debug("re-centered value = ["+str(y)+" , "+str(x)+"]")
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
    nHiRes = 9.0
    log.info("*"*10+"   Starting to extract 13x13pix PSFs, stack and perform PCA on them   "+"*"*10)
    psfStack = []
    numAdded = 0
    numNotAdded = 0
    inMonoCorrected = inMono
    np.putmask(inMonoCorrected,np.isnan(inMonoCorrected),0.0)
    yAryHiRes = np.linspace(-6.0,+6.0,(12*nHiRes+1))
    print 'yAryHiRes.shape = '+repr(yAryHiRes.shape)
    xAryHiRes = yAryHiRes
    for i in range(0,len(centersUpdated)):
        currAry = np.zeros((13,13))
        y = centersUpdated[i][0]
        x = centersUpdated[i][1]
        if ((x>(xMax-6))or(y>(yMax-6)))or((x<6)or(y<6)):
            numNotAdded += 1
            log.debug("This PSF has insufficient surrounding pixels to be cropped to 13x13, center = ["+str(y)+" , "+str(x)+"]")
        else:
            numAdded += 1
            yAry2 = yAryHiRes+y
            xAry2 = xAryHiRes+x
            xAry2,yAry2 = np.meshgrid(xAry2,yAry2)
            currAry = ndimage.map_coordinates(inMonoCorrected,[yAry2,xAry2],order=3)
            psfStack.append(currAry)
        if np.isnan(np.sum(currAry)):
            print "\n\n"+repr(currAry)
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
    from scipy import linalg  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ HACK!!
    u, s, V = linalg.svd(stackFlattened.T,full_matrices=False)
    uNew = u.T[:ncomp]  # These are the top components from the PCA/SVD decomposition, but need to be reshaped to de-flatten
    log.debug("output uNew has shape: "+repr(uNew.shape))
    
    # de-flatten U array to make PCA component array
    pcaAry = np.reshape(uNew, (ncomp, oldshape[1], oldshape[2]))
    # de-flatten mean
    print "stackMean shape = "+repr(stackMean.shape)
    print "oldshape[1] = "+str(oldshape[1])+", oldshape[2] = "+str(oldshape[2])
    stackMeanUnflat = np.reshape(stackMean, (oldshape[1], oldshape[2]))
    log.debug("pcaAry has shape: "+repr(pcaAry.shape))
    pcaAryFlattened = uNew
    
    # save PCA components as individual fits files
    outPCAdir= os.path.join(outputDir,"pcaOutputs")
    if writeFiles:
        os.mkdir(outPCAdir)
        # store mean first
        hdu = pf.PrimaryHDU(stackMeanUnflat)
        hduList = pf.HDUList([hdu])
        outFilename = os.path.join(outPCAdir,"pcaComponent_mean.fits")
        hduList.writeto(outFilename)
        hduList.close()
        for j in range(ncomp):
            # Push mean in as zeroth PCA comp
            hdu = pf.PrimaryHDU(pcaAry[j])
            hduList = pf.HDUList([hdu])
            outFilename = os.path.join(outPCAdir,"pcaComponent_"+str(j+1)+".fits")
            hduList.writeto(outFilename)
            hduList.close()    
    # Push mean in as zeroth PCA comp
    pcaAry2 = []
    pcaAry2.append(stackMeanUnflat)
    for j in range(ncomp):
        pcaAry2.append(pcaAry[j])
    pcaAry2 = np.array(pcaAry2)
    
    
# take 5-7 PCA comps including mean frame as zeroth,
# bin down to 11x11 and subtract the PCA components from the PSFs, calculating chi squared 
# Then use the below commented out code from ACORNS to shift the centers and recalculate the chi squared, 
# this will find the center with the lowest chi squared which will be the updated best center.
# Once this is complete.  redo the iterative re-centering using this PCA approach.
 
    meanDiff = 100
    centersLast = centersUpdated
    iteration = 0
    log.info("*"*5+"  Performing iterative PSF centering with PCA comps loop  "+"*"*5)
    nref = 7
    ## Create the stepping array for moving around the center
    x = np.arange(-4,5)
    stepBoxWidth = x.shape[0]
    x, y = np.meshgrid(x, x) 
    while meanDiff>0.1:
        iteration+=1
        if debug:
            log.debug("\n Starting iteration = "+str(iteration))
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # FIRST: think about how to fit a quadratic to the 2D outputs y,x using the chi2 array to find the fractional best fit center, not just integer based center.
        # if stuck, Tim used something similar in his ACORNS centering routines.
        
        
        # No PCA component re-centering in a loop.  Instead, with the new centers from a single round of PCA re-centering
        # use those new centers to reproduce the PCA comps, and do another round of PCA based re-centering.  Put this 
        # two stage process in a loop until convergence.  AND remove the iterative re-centering with COF, just once if fine.
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        ###################################################################################################################
        # PCA was done using 13x13 arrays.  First the central 9x9 of the low/standard resolution PSF will be cropped out.
        # Then the high resolution PCA components will have their central 11x11pix cropped out, binned down to 1pix resolution
        # and fit to the PSF.  It will be shifted around the center using a 9x9 box at 0.1pix resolution until the best new 
        # center is found based on the chi squared of the fit.
        ###################################################################################################################
        #log.info("*"*10+"   Starting to shift and compare central 11x11 of PSF centers to PCA comps to find better centers  "+"*"*10)
        centersUpdated2 = []
        centersUpdated3 = []
        offsetsBest = []
        recentSuccess = []
        ## Loop over each PSF center in the updated centers array produced by the center of light approach
        for center in range(1):#len(centersLast)):
            #print '\ncenter #'+str(center)
            chi2Best = np.inf
            chi2 = np.zeros((stepBoxWidth, stepBoxWidth))
            ybest, xbest = [0, 0]
            y1Orig = y1 = centersLast[center][0]
            x1Orig = x1 = centersLast[center][1]
            y1FracShift = y1%1
            x1FracShift = x1%1
            y1 = int(y1)
            x1 = int(x1)
            # check if fraction is over 0.5, shift if so 1 whole pix to right
            if y1FracShift>=0.5:
                y1 +=1
                y1FracShift -= 1.0
            if x1FracShift>=0.5:
                x1 += 1
                x1FracShift -= 1.0
            # crop 11x11 PSF and flatten
            currPSFflat = np.reshape(inMonoCorrected[y1-4:y1+5,x1-4:x1+5],-1)
            print 'y1Orig, x1Orig = %2f, %2f'%(y1Orig,x1Orig)
            print 'y1,x1 = (%.2f + %.2f, %.2f + %.2f)' % (y1,y1FracShift, x1, x1FracShift)
            iBest = jBest = 0
            yBest = xBest = 0
            yBestStr = xBestStr = ''
            ## Loop over each shifted center point to find best new center based on lowest chi squared value
            for i in range(stepBoxWidth):
                for j in range(stepBoxWidth):
                    y2 = y[i, j] 
                    x2 = x[i, j]                
                    ## My version, Not sure if correct as don't know why multiplication by std was used in Tim's version
                    pcaCompsUSE = np.zeros((nref,currPSFflat.shape[0]))
                    for k in range(nref):
                        # crop 11x11 high resolution PCA components, rebin and flatten
                        #print 'pca orig size'+repr(pcaAry2[k].shape)
                        yShift = y2-int(y1FracShift*9.0)
                        xShift = x2-int(x1FracShift*9.0)
                        #print "yshift = %.2f + %2f"%(y2,y1FracShift*9.0)
                        #print "xshift = %.2f + %2f"%(x2,x1FracShift*9.0)
                        #print "yShift,xShift = (%.2f,%.2f) "%(yShift,xShift)
                        pcaCropped = pcaAry2[k][14+yShift:-14+yShift,14+xShift:-14+xShift]
                        #print "pcaCropped.shape  = "+repr(pcaCropped.shape)
                        pcaBinned = tools.rebin(pcaCropped,(9,9))
                        #print "pcaBinned.shape = "+repr(pcaBinned.shape)
                        pcaCompsUSE[k] = np.reshape(pcaBinned,-1)
                    A = pcaCompsUSE
                    b = currPSFflat
                    coef = linalg.lstsq(A.T, b)[0]
             
                    # Compute residuals, sum to get chi2
                    resid = currPSFflat - coef[0] * pcaCompsUSE[0]
                    #print "zeroth residual = "+str(resid)
                    for k in range(1, nref):
                        resid -= coef[k] * pcaCompsUSE[k]
                    #print "resid afterwards = "+str(resid)
                    #print "raw chi2  = "+str(np.sum(currPSFflat**2))
                    chi2[i, j] = np.sum(resid**2)
                    #print "chi2[i,j] = "+repr(chi2[i,j])
                    if chi2[i, j] < chi2Best:
                        chi2Best = chi2[i, j]
                        iBest, jBest = [i, j]
                        yBest = yShift
                        xBest = xShift
                        yBestStr = "yshift PCA = %.2f - %2f"%(y2,y1FracShift*9.0)
                        xBestStr = "xshift PCA = %.2f - %2f"%(x2,x1FracShift*9.0)                        
#             #print "y1 = "+str(y1)+", x1 = "+str(x1)
            #print "ibest = "+str(ibest)+", jbest = "+str(jbest)
            offsetsBest.append([y[iBest,jBest],x[iBest,jBest]])
            #print "yBest,xBest PCA = (%2f, %2f)"%(yBest,xBest)
            #print yBestStr+", "+xBestStr
            #print "OR in 1pix/element PSF= (%2f, %2f)"%(yBest/9.0,xBest/9.0)
            #print repr(chi2)
            centersUpdated2.append([y1+(1.0/9.0)*int(y1FracShift*9.0)+(y[iBest,jBest]/9.0),x1+(1.0/9.0)*int(x1FracShift*9.0)+(x[iBest,jBest]/9.0)])
            
            ################################################################################################################################################ 
            # Use a least squares fit to a parabola of (chi2,x,y) to find true x,y shift values instead of integer based ones above.  Only using central 3x3
            ################################################################################################################################################
            try:
                y0 = y[iBest,jBest]
                x0 = x[iBest,jBest]
                yPara = np.reshape(y[1+y0:y0-1,y01:y0-1],-1)
                xPara = np.reshape(x[1+y0:y0-1,y01:y0-1],-1)
                initGuess = [2.0, 2.0, chi2Best, y0, x0]
                chi2Para = chi2[1:-1,1:-1]
                print "shape chi2Para = "+repr(chi2Para.shape)
                bestFitVals = optimize.leastsq(residual, initGuess, args=(xPara,yPara,chi2Para))
            except:
                if (abs(y0)>1) or (abs(x0)>1):
                    recentSuccess.append(False)
                else:
                    recentSuccess.append(True)
            centersUpdated3.append(True)#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            log.debug("chi2Best = "+str(chi2Best))
            log.debug("Previous center = "+repr(centersLast[center])+", new ones are = "+repr(centersUpdated2[center])+"\n")
            
        #print repr(offsetsBest)
        meanDiff = abs(np.mean(centersLast)-np.mean(centersUpdated2))
        print "(np.mean(centersLast)) "+str(np.mean(centersLast))+" - "+str(np.mean(centersUpdated2))+" (np.mean(centersUpdated2)) = "+str(meanDiff)
        log.debug("PSF #50: Original center = "+repr(centersUpdated[50])+", new ones are = "+repr(centersUpdated2[50]))
        log.debug("PSF #1000: Original center = "+repr(centersUpdated[1000])+", new ones are = "+repr(centersUpdated2[1000]))
        centersLast = centersUpdated2
        #log.debug("Latest diff = "+str(meanDiff))
    print repr(offsetsBest)
    log.info("Finished PSF PCA based re-centering loop in "+str(iteration)+" iterations, resulting in a mean difference of "+str(meanDiff)+"\n")
    
def residual(p,x,y,chi2):
    a,b,c,d,xc,yc = p 
    return chi2 - (a*(x-xc)**2.0+b*(y-yc)**2.0+c*(x-xc)(y-yc)+d)
    
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
