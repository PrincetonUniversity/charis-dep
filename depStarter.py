#!/usr/bin/env python
import pyfits as pf
import numpy as np
import os
import shutil
import sys

import tools
import primitives as prims
import configFiles as configs


def main():
    """
    This Function takes the place as the 'main' to start and run all the 
    requested reduction steps on the input data using the information set 
    in the configuration files.
    """  
    ## check if output folder exists, else create it
    if os.path.exists(configs.DEPconfig.outDirRoot):
        print '\n'+'$'*100+'\n'
        print 'WARNING!! the folder:'+configs.DEPconfig.outDirRoot+', all ready exists!'
        print 'You can overwrite the data in it, or exit this simulation.'
        YN = raw_input('\nOVERWRITE current folder (y/n):')
        if (('y' in YN) or ('Y' in YN)):
            print '\nDELETING all contents of folder:'+configs.DEPconfig.outDirRoot
            shutil.rmtree(configs.DEPconfig.outDirRoot)
            print 'MAKING new empty folder:'+configs.DEPconfig.outDirRoot
            os.mkdir(configs.DEPconfig.outDirRoot)
        elif (('n' in YN) or ('N' in YN)):
            sys.exit()
        print '\n'+'$'*100+'\n'
        print 'outputFolder all ready exists, so just adding new file to it.'
    else:
        os.mkdir(configs.DEPconfig.outDirRoot)
        
    # move to output directory, but copy current one to move back into after finished.
    pwd = os.curdir
    os.chdir(configs.DEPconfig.outDirRoot)
    
    #########################################################################
    # Set up the main and summary loggers, including system and process info.
    #########################################################################
    log = tools.getLogger('main')
    tools.logSystemInfo(log)
    tools.logFileProcessInfo(log)
    summaryLog = tools.setUpLogger('main.summary',addFH=False,addSH=False)
    tools.addFitsStyleHandler(summaryLog)
    
    # test the BPM primitive
    bpmData = tools.loadDataAry(configs.DEPconfig.inBPMfile)
    summaryLog.info('Masking hot pixels with file: '+os.path.basename(configs.DEPconfig.inBPMfile))
    bpmCorrDatas = []
    outHDUs = []
    i = 0
    inDataFiles = configs.DEPconfig.inDataFiles#configs.DEPconfig.inputNDRs
    print "in depStarter there are "+str(len(inDataFiles))+" datafiles"#$$$$$$$$$$
    for inputNDR in inDataFiles:
        log.debug('Currently applying BPM to file '+str(i+1)+'/'+str(len(inDataFiles)))
        outData = prims.maskHotPixels(inputNDR,bpmData)
        bpmCorrDatas.append(outData)
        outHDU = tools.loadHDU(inputNDR)
        outHDU[-1].data = outData
        outHDUs.append(outHDU)
        # load nparray back into original pf object?
    log.debug("Finished BPM corrections and total outputs were = "+str(len(bpmCorrDatas)))
    log.info('Finished BPM correction(s)')
    
    log.info("writing latest "+str(len(outHDUs))+" data to output files")
    for outHDU in outHDUs:
        tools.updateOutputFitsHeader(outHDU, 'main.summary.fitsFormat.log')
        split = os.path.splitext(os.path.basename(outHDU.filename()))
        outFileNameRoot = split[0]+'_OUT'+split[1]
        outFileName = os.path.join(configs.DEPconfig.outDirRoot,outFileNameRoot)
        outHDU.writeto(outFileName)
        outHDU.close()
        log.info("output HDUlist written to: "+outFileName)
    
    # move back into original working directory
    os.chdir(pwd)
#############################################################
# end
#############################################################
if __name__ == '__main__':
    main() 
