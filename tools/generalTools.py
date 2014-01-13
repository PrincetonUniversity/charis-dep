import tools

def testToolFunc():
    """
    Just a test tool func.  Will be deleted once more real tools are written.
    """
    log = tools.setUpLogger('testLog.tools')
    print 'Inside testToolFunc'
    log.info('InfoMsgInsideTools')