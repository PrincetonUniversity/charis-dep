#!/usr/bin/env python

import tools
import primitives

def main():
    """
    """

    """
    Start a 'main' level logging object and load it up with the user's 
    system information for future reference.
    """
    log = tools.setUpLogger('testLog')
    tools.systemInfoMessages(log)
    log.info('infoTest')    
    log.maincritical('maincritTest')
    log.maindebug('maindebugTest')
    log.primcritical('primcriticalTest')
    log.priminfo('priminfoTest')
    log.toolerror('toolerrorTest',exc_info=True)
    log.summary('summaryTest')
    log.toolerror('toolerrorTest',exc_info=True)
    
    log.setLevel(100)
    log.info('infoTest_afterLvlSetTo_100')   
    
    primitives.testCalPrim()
    
    log3 = tools.getLogger('testLog.summary')
    log3.summary('summaryTestMsg')
    
    
    log1 = tools.getLogger('main')
    tools.systemInfoMessages(log1)
    log1.setStreamLevel(10)
    log1.info('TestInfoMsg')
    
    log2 = tools.getLogger('main.prims')
    log2.setStreamLevel(50)
    log2.info('TestInfoMsg')
    
    log22 = tools.getLogger('main.tools')
    log22.setStreamLevel(80)
    log22.info('TestInfoMsg')
    
    
    
    
#############################################################
# end
#############################################################
if __name__ == '__main__':
    main() 
