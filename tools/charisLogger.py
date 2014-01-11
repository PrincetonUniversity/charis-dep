import logging
import platform
import sys
import traceback

log_dict={}

class CharisLogger(logging.getLoggerClass()):
    """
    This is the advanced logging object used throughout the CHARIS
    Data Extraction Pipeline.  It inherits from the standard 
    Python library 'logging' and provides added features.
    The default log level for the output file will be DEBUG, ie ALL messages;
    while the default for the screen will be INFO, and can be changed easily 
    using the setLevel(lvl) member function.
    """       
    
    def newFunc(self,var):
        """Temp function to be placeholder for when an added function is 
        needed.
        """
        print 'var:'+repr(var)  
        
    # Add the new log levels needed for the 3 tier hierarchy plus the summary
    # level to the logging object.
    # Levels for the 'main', or top, tier.
    MAINCRITICAL = 80
    logging.addLevelName(MAINCRITICAL, 'MAINCRITICAL')
    def maincritical(self,msg,lvl=MAINCRITICAL, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.maincritical = maincritical
    MAINERROR = 75
    logging.addLevelName(MAINERROR, 'MAINERROR')
    def mainerror(self,msg,lvl=MAINERROR, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.mainerror = mainerror
    MAINWARNING = 70
    logging.addLevelName(MAINWARNING, 'MAINWARNING')
    def mainwarning(self,msg,lvl=MAINWARNING, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.mainwarning = mainwarning
    MAININFO = 65
    logging.addLevelName(MAININFO, 'MAININFO')
    def maininfo(self,msg,lvl=MAININFO, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.maininfo = maininfo
    MAINDEBUG = 60
    logging.addLevelName(MAINDEBUG, 'MAINDEBUG')
    def maindebug(self,msg,lvl=MAINDEBUG, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.maindebug = maindebug
    # Levels for the 'prims' tier.
    PRIMCRITICAL = 55
    logging.addLevelName(PRIMCRITICAL, 'PRIMCRITICAL')
    def primcritical(self,msg,lvl=PRIMCRITICAL, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.primcritical = primcritical
    PRIMERROR = 50
    logging.addLevelName(PRIMERROR, 'PRIMERROR')
    def primerror(self,msg,lvl=PRIMERROR, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.primerror = primerror
    PRIMWARNING = 45
    logging.addLevelName(PRIMWARNING, 'PRIMWARNING')
    def primwarning(self,msg,lvl=PRIMWARNING, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.primwarning = primwarning
    PRIMINFO = 40
    logging.addLevelName(PRIMINFO, 'PRIMINFO')
    def priminfo(self,msg,lvl=PRIMINFO, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.priminfo = priminfo
    PRIMDEBUG = 35
    logging.addLevelName(PRIMDEBUG, 'PRIMDEBUG')
    def primdebug(self,msg,lvl=PRIMDEBUG, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.primdebug = primdebug
    # Levels for the 'tools' tier.
    TOOLCRITICAL = 30
    logging.addLevelName(TOOLCRITICAL, 'TOOLCRITICAL')
    def toolcritical(self,msg,lvl=TOOLCRITICAL, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.toolcritical = toolcritical
    TOOLERROR = 25
    logging.addLevelName(TOOLERROR, 'TOOLERROR')
    def toolerror(self,msg,lvl=TOOLERROR, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.toolerror = toolerror
    TOOLWARNING = 20
    logging.addLevelName(TOOLWARNING, 'TOOLWARNING')
    def toolwarning(self,msg,lvl=TOOLWARNING, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.toolwarning = toolwarning
    TOOLINFO = 15
    logging.addLevelName(TOOLINFO, 'TOOLINFO')
    def toolinfo(self,msg,lvl=TOOLINFO, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.toolinfo = toolinfo
    TOOLDEBUG = 10
    logging.addLevelName(TOOLDEBUG, 'TOOLDEBUG')
    def tooldebug(self,msg,lvl=TOOLDEBUG, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.tooldebug = tooldebug
    # Level for the 'summary' info used for the main log file and the 
    # special summary file and the fits headers.
    SUMMARY = 5
    logging.addLevelName(SUMMARY, 'SUMMARY')
    def summary(self,msg,lvl=SUMMARY, *args, **kws):
        self.log(lvl,msg, *args, **kws)
    logging.Logger.summary = summary

def getLogger(name='generalLoggerName'):
    """This will either return the logging object already
    instantiated, or instantiate a new one and return it.
    
    Args:
        name (str): The name for the logging object and 
                    name.log will be the output file written to disk.
                    
    Returns:
        log (CharisLogger object): A CharisLogger object that was either 
                                  freshly instantiated or determined to 
                                  already exist, then returned.
    """
    log = False
    try:
        log = log_dict[name]
        #print repr(log_dict)
        #print 'found a log by the name already exists so returning it'
    except:
        log = setUpLogger(name)
    return log
    
def setUpLogger(name='generalLoggerName'):
    logging.setLoggerClass(CharisLogger)
    log = logging.getLogger(name)
    log_dict[name]=log
    #self.logger = logging.getLogger(name)
    log.setLevel(1)
    # make a file handler
    fh = logging.FileHandler(name+'.log')
    fh.setLevel(1)
    frmtString = '%(asctime)s - %(levelname)s - %(message)s'
    fFrmt = logging.Formatter(frmtString)
    fh.setFormatter(fFrmt)
    # add the Handler to the logger
    log.addHandler(fh)
    # make a stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(1)
    sFrmt = logging.Formatter('%(message)s')
    sh.setFormatter(sFrmt)
    # add the Handler to the logger
    log.addHandler(sh)
    # Call the system Info Message func to load up top of log with system Info.
    systemInfoMessages(log)
    
    return log

def systemInfoMessages(log):
    """ A function to be called just after a logging object is instantiated 
    for the DEP to load the log up with info about the computer it is 
    being ran on and the software version.
    """
    log.info('-'*50)
    log.info("-"*10+' System Information Summary '+'-'*10)
    log.info('Machine Type = '+platform.machine())
    log.info('Machine Version = '+platform.version())
    log.info('Machine Platform = '+platform.platform())
    log.info('Machine UserName = '+platform.uname()[1])
    log.info('Machine Processor = '+platform.processor())
    log.info('Python Version = '+repr(platform.python_version()))
    log.info('-'*50)
    
#     def __tracebacker(self):
#         """
#         Internal function for creating nicely formatted 
#         tracebacks for the ERROR level messages if requested.
#         """
#         ex_type, ex, tb = sys.exc_info()
#         tb_list = traceback.extract_tb(tb,6)
#         s='\nTraceback:\n'
#         for i in range(0,len(tb_list)):
#             line_str = ', Line: '+tb_list[i][3]
#             func_str = ', Function: '+tb_list[i][2]
#             ln_str = ', Line Number: '+str(tb_list[i][1])
#             file_str = ' File: '+tb_list[i][0]
#             s = s+file_str+func_str+ln_str+'\n'
#         return s