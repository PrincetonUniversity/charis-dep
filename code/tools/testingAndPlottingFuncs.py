"""
Test plotting and such tools to test algorithms during development.  This toolbox and its parent folder should 
not be pushed into the "master" branch and should always remain in the "devel" branch.
"""
import numpy as np 
import tools 
import pylab
plt = pylab.matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

log = tools.getLogger('main.tools',lvl=0,addFH=False)

def loadChi2s(filename):
    
    f = open(filename)
    lines = f.readlines()
    ary2 = []
    ary = []
    for line in lines:
        if line!="\n":
            ary.append(float(line))
        else:
            ary2.append(ary)
            ary = []
    return ary2

def plotChiSquaredHists(chi2s, plotFilename=""):
    """
    plot histogram(s) of the chi squared values for the different PSFs fit in a certain number of iterations.  
    This can be stacked for different numbers of iterations using different colors.
    
    NOTE: for now this will be done largely all in this function, but could be broken up if useful parts
          can be split off for better use elsewhere.
    """
    colorsList =['Blue','BlueViolet','Chartreuse','Fuchsia','Crimson','Aqua','Gold','DarkCyan','OrangeRed','Plum','DarkGreen','Chocolate','SteelBlue ','Teal','Salmon','Brown']
    
    fig = plt.figure(1, figsize=(40,20) ,dpi=300)
    ## give the figure its title
    plotFileTitle = ""
    plt.suptitle(plotFileTitle,fontsize=30)
    ax = fig.add_subplot(111)
    for i in range(0,len(chi2s)):
        (n,bins,rectangles)=ax.hist(chi2s[i], bins=25, normed=True, label="Iter "+str(i+1), color=colorsList[i], histtype='step')
    
    # shrink box width by 20% to fit legend to right side
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.80,box.height])
    
    # Add legend to right of plot
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
    
    # save plot to file
    if plotFilename!='':
        plotFilename2 = plotFilename[:-4]+".png"
        plt.savefig(plotFilename2, dpi=300, orientation='landscape')
        s= '\nFigure saved to:\n'+plotFilename2
        log.info(s+'\n')
    else: 
        s= '\nWARNING: NO plotFilename provided, so NOT saving it to disk.'
        log.warning(s+'\n')
        
    plt.close()
    
def surfacePlotter(Z,X=0,Y=0,plotFilename=""):
    """
    Use meshgrid to make X and Y arrays representing pixel locations at 1pix resolution matching.
    Z is the 2D array of pixel values.
    assume input array is square.
    """
    debug = True
    if debug:
        print "\n\n in surfacePlotter \n\n" 
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)
    
    x = np.arange(-1*int((Z.shape[0])/2),int(Z.shape[1]/2)+1)
    X,Y = np.meshgrid(x,x)
    
    fig = plt.figure(1, figsize=(40,20) ,dpi=300)
    if debug:
        print "About to plot surface" 
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Z)
    if debug:
        print "Done making surface plot \n\n" 
    # save plot to file
    if plotFilename!='':
        plotFilename2 = plotFilename[:-4]+".png"
        plt.savefig(plotFilename2, dpi=300, orientation='landscape')
        s= '\nFigure saved to:\n'+plotFilename2
        log.info(s+'\n')
    else: 
        s= '\nWARNING: NO plotFilename provided, so NOT saving it to disk.'
        log.warning(s+'\n')
        
    plt.close()
    
    