#!/usr/bin/env python

import tools
import primitives

def main():
    """
    """
    
    log = tools.CharisLogger('testLog')
    
    log.debug('tstDebgMsg')
    log.info('tstInfoMsg')
    log.error('tstErrMsg')
    
    primitives.testCalPrim()
    
    
    
    
#############################################################
# end
#############################################################
if __name__ == '__main__':
    main() 
