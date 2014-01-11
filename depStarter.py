#!/usr/bin/env python

import tools
import primitives

def main():
    """
    """
    
#     log = tools.CharisLogger('testLog')
#     
#     log.debug('tstDebgMsg')
#     log.info('tstInfoMsg')
#     log.error('tstErrMsg')
    
    
    
    #log2 = tools.CharisLogger2('testLog2')
    log2 = tools.setUpLogger('testLog')
    log2.info('infoTest')
    log2.maincritical('maincritTest')
    log2.maindebug('maindebugTest')
    log2.primcritical('primcriticalTest')
    log2.priminfo('priminfoTest')
    log2.toolerror('toolerrorTest',exc_info=True)
    log2.summary('summaryTest')
    log2.toolerror('toolerrorTest',exc_info=True)
    
    primitives.testCalPrim()
    
    log3 = tools.getLogger('testLog.summary')
    log3.summary('summaryTestMsg')
    
    
    
    
#############################################################
# end
#############################################################
if __name__ == '__main__':
    main() 
