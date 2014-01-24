#!/usr/bin/env python

import tools
import primitives
import configFiles


def main():
    """
    This Function takes the place as the 'main' to start and run all the 
    requested reduction steps on the input data using the information set 
    in the configuration files.
    """
    
#     log = tools.setUpLogger('testLog')
#     tools.systemInfoMessages(log)
#     log.info('infoTest')    
#     log.maincritical('maincritTest')
#     log.maindebug('maindebugTest')
#     log.primcritical('primcriticalTest')
#     log.priminfo('priminfoTest')
#     log.toolerror('toolerrorTest',exc_info=True)
#     log.summary('summaryTest')
#     log.toolerror('toolerrorTest',exc_info=True)
#     
#     log.setLevel(100)
#     log.info('infoTest_afterLvlSetTo_100')   
#     
#     primitives.testCalPrim()
#     
#     log3 = tools.getLogger('testLog.summary')
#     log3.summary('summaryTestMsg')
    
    
    log = tools.getLogger('main')
    tools.logSystemInfo(log)
    tools.logFileProcessInfo(log)
    
    inputNDRs = []#!!!!!!!!!!!?????!!
    for inputNDR in inputNDRS:
        outData = maskHotPixels(inputNDR,BPM)
        # load nparray back into original pf object?
        
    
#     tools.systemInfoMessages(log1)
#     log1.setStreamLevel(10)
#     log1.info('TestInfoMsg')
#     
#     log2 = tools.getLogger('main.prims')
#     log2.setStreamLevel(50)
#     log2.info('TestInfoMsg')
#     
#     log22 = tools.getLogger('main.tools')
#     log22.setStreamLevel(80)
#     log22.info('TestInfoMsg')
#     
#     log23 = tools.getLogger('main.summary')
#     log23.setStreamLevel(80)
#     tools.addFitsStyleHandler(log23)
#     log23.info('TestInfoMsg')
#     
    
    
    
    
    
#############################################################
# end
#############################################################
if __name__ == '__main__':
    main() 
