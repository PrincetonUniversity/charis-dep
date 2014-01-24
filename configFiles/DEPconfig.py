"""
This file will hold the configurations/settings for the manual running of the 
DEP during the early testing and development stages.
Each new setting must have its arg information added to this docstring for the 
Sphinx docs, and will be used for the future versions/interfaces for the 
setting's documentation.

Args:
    inDataDir (str): Full path to the directory where the input data lives.
    
    outDirRoot (str): Full path to the root directory where the folders and
                        files output by the DEP will go for a certain run. The
                        subfolder created in this for a run will be based on
                        the input data file's name and the date.
                        
    inDataFile (str): Name of the input file name to reduce.  The selection of
                      this/these files will be updated in the future, but for 
                      current testing providing the single file string will 
                      suffice.
    
    
"""

inDataDir = "/mnt/Data1/Todai_Work/Data/data_CHARIS/testData"
outDirRoot = "/mnt/Data1/Todai_Work/Data/data_CHARIS/testOutputs/"
inDataFile = 'N20050227S0127.fits'
inBPMfile = 'combinedBPM.fits'
inFlatfile = 'NIRI_norm_flat.fits'



