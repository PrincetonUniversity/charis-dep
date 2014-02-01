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
                        
    inDataFilesRaw (list of strs): List of the name(s) of the input file name(s)
                                to reduce.  The selection of this/these file(s)
                                 will be updated in the future, but for current
                                testing providing the single file string will 
                                suffice.
    
    
"""

import pyfits as pf
import numpy as np
import os
import glob

inDataDir = "/mnt/Data1/Todai_Work/Data/data_CHARIS/testData"
outDirRoot = "/mnt/Data1/Todai_Work/Data/data_CHARIS/testOutputs/"
inDataFilesRaw = ['N20050227S0127.fits']
# Next line just converts the raw names to full path versions
inDataFiles = []
for file in inDataFilesRaw:
    inDataFiles.append(os.path.join(inDataDir,file))
ndrRoot = '*R1*.fits'
# get data list for NDRs and sort it
inputNDRs = np.sort(glob.glob(inDataDir +"/individual_reads/"+ndrRoot))
if False:
    bpmRoot = 'combinedBPM.fits'
    inBPMfile = os.path.join(inDataDir,bpmRoot)
if True:
    bpmRoot = 'ndrsFakeBPM.fits'
    inBPMfile = os.path.join(inDataDir +"/individual_reads/",bpmRoot)
flatRoot = 'NIRI_norm_flat.fits'
inFlatfile = os.path.join(inDataDir,flatRoot)

# List of Primitives to run
applyBPM = True
fitNDRs = True



