"""
Test plotting and such tools to test algorithms during development.  This toolbox and its parent folder should 
not be pushed into the "master" branch and should always remain in the "devel" branch.
"""

import primitives as prims 
import tools 

def plotChiSquaredHists():
    """
    plot histogram(s) of the chi squared values for the different PSFs fit in a certain number of iterations.  
    This can be stacked for different numbers of iterations using different colors.
    
    NOTE: for now this will be done largely all in this function, but could be broken up if useful parts
          can be split off for better use elsewhere.
    """
    
    