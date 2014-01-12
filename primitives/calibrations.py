import tools

def testCalPrim():
    """
    Just a test prim during very early development.
    """
    log = tools.setUpLogger('testLog.prims')
    print("this is an empty test calibration primitive")
    log.info('testInfoMsgInsidePrim')
    
    tools.testToolFunc()