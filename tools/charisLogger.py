"""
.. moduleauthor:: Kyle Mede <kylemede@astron.s.u-tokyo.ac.jp>
"""

import logging
import sys
import traceback

log_dict={}

class CharisLogger(object):
    """
    This is the advanced logging object used throughout the CHARIS
    Data Extraction Pipeline.  It is a wrapper around the standard 
    Python library 'logging' with added features.
    Upon instantiation of the CharisLogger object, it will return 
    itself (ie. log = CharisLogger('blahblah_name')).
    The default log level for the output file will be DEBUG, ie ALL messages;
    while the default for the screen will be INFO, and can be changed easily 
    using the setLevel(lvl) member function.
    
    Args:
        name (str): The name for the logging object and 
                    name.log will be the output file written to disk.
                    
    Returns:
        log (CharisLogger object): A CharisLogger object that was either 
                                  freshly instantiated or determined to 
                                  already exist, then returned.
    """
    global log_dict
    
    def __init__(self,name='generalLoggerName'):
        self.logger = self.createLogger(name)
          
    def createLogger(self,name='generalLoggerName'):
        """An internal function that is called by __init__
        when a new charisLogger object is instantiated.
        It also has a failsafe to check if a logging object
        by that name already exists and just returns that.
        
        Args:
            name (str): The name for the logging object and 
                        name.log will be the output file written to disk.
        """
        log = False
        try:
            log = log_dict[name]
            #print repr(log_dict)
            #print 'found a log by the name already exists so returning it'
        except:
            #print 'Did not find a log by that name, so making a new one.'
            # create logger
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)
            # make a file handler
            fh = logging.FileHandler(name+'.log')
            fh.setLevel(logging.DEBUG)
            # create a formatter and set the formatter for the handler.
            frmtString = '%(asctime)s - %(levelname)s - %(message)s'
            fFrmt = logging.Formatter(frmtString)
            fh.setFormatter(fFrmt)
            # add the Handler to the logger
            self.logger.addHandler(fh)
            # make a stream handler
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            sFrmt = logging.Formatter('%(message)s')
            sh.setFormatter(sFrmt)
            self.logger.addHandler(sh)
            # add this new logging object to dict using name as key
            log_dict[name]=self.logger
            log = self.logger
        return log
    
    def getLogger(self,name='generalLoggerName'):
        """This will either return the logging object already
        instantiated, or instantiate a new one and return it.
        
        Args:
            name (str): The name for the logging object and 
                        name.log will be the output file written to disk.
        """
        log = False
        try:
            log = log_dict[name]
        except:
            log = self.createLogger(name)
        self.logger = log
        return self
        
    def debug(self,msg):
        """
        For 'DEBUG' level messages.
        
        Args:
            msg (str): The message to log.
        """
        self.logger.debug(msg)
    def info(self, msg):
        """
        For 'INFO' level messages.
        
        Args:
            msg (str): The message to log.
        """
        self.logger.info(msg)
    def warning(self,msg):
        """
        For 'WARNING' level messages.
        
        Args:
            msg (str): The message to log.
        """
        self.logger.warning(msg)
    def error(self,msg,exc_info=True):
        """
        For 'ERROR' level messages.
        
        Args:
            msg (str): The message to log.
            exc_info (bool): Also log a nicely formated traceback? Default=True
        """
        tb_str = ''
        if exc_info==True:
            tb_str = self.__tracebacker()
        self.logger.error(msg+tb_str)
    def critical(self,msg):
        """
        For 'CRITICAL' level messages.
        
        Args:
            msg (str): The message to log.
        """
        self.logger.critical(msg)
        
    def setLevel(self,lvl):
        """
        A function to set the level of prints to the screen.
        
        Args:
            lvl (str): level to set it to.  All messages from levels
                       ABOVE lvl will be printed to the screen.
                       Options are DEBUG, INFO, WARNING, ERROR, CRITICAL.
        """
        levelDict={'DEBUG':logging.DEBUG,
                   'INFO':logging.INFO,
                   'WARNING':logging.WARNING,
                   'ERROR':logging.ERROR,
                   'CRITICAL':logging.CRITICAL}
        try:
            self.logger.setLevel(levelDict[lvl])    
        except:
            print('The provided value for lvl "'.join(str(lvl)).join(' is not a valid value'))
            print('Please use a value from : "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".')
            
    
    def __tracebacker(self):
        """
        Internal function for creating nicely formatted 
        tracebacks for the ERROR level messages if requested.
        """
        ex_type, ex, tb = sys.exc_info()
        tb_list = traceback.extract_tb(tb,6)
        s='\nTraceback:\n'
        for i in range(0,len(tb_list)):
            line_str = ', Line: '+tb_list[i][3]
            func_str = ', Function: '+tb_list[i][2]
            ln_str = ', Line Number: '+str(tb_list[i][1])
            file_str = ' File: '+tb_list[i][0]
            s = s+file_str+func_str+ln_str+'\n'
        return s
    