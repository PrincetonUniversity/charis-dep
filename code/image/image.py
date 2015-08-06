import pyfits as pyf
import tools

log = tools.getLogger('main')

class Image:
    
    """
    Image is the basic class for raw and partially reduced CHARIS data.
    It must have at least the following boolean attribute references:
        self.destriped  (default False)
        self.flatfielded (default False)
    It must have at least the following other attribute references:
        self.data (default None)
        self.ivar (default None)
        self.header (default None)
    
    self.data, self.ivar, and self.header should be numpy ndarrays, 
    which can be read from and written to a fits file with the load 
    and write methods.  If not ndarrays, they should be None.

    Image may be initialized with the name of the raw file to read,
    through a call to Image.load().  
    """

    def __init__(self, filename=None, destriped=False, flatfielded=False):
        self.destriped = destriped
        self.flatfielded = flatfielded
        
        if filename is not None:
            self.load(filename)
        else:
            self.data = None
            self.ivar = None
            self.header = None
                
    def load(self, filename):
        """
        Image.load(outfilename)
        
        Read the first (index 0) HDU from filename into self.data, and
        HDU[0].header into self.header.  If there is more than one HDU,
        attempt to read HDU[1] into self.ivar.

        """
        try:
            hdulist = pyf.open(filename)
            self.data = hdulist[0].data
            self.header = hdulist[0].header
            log.info("Read data from HDU 0 of " + filename)
            if len(hdulist) > 1:
                self.ivar = hdulist[1].data
                if self.ivar.shape != self.data.shape:
                    log.error("Error: data (HDU 0) and inverse variance (HDU 1) have different shapes in file " + filename)
                    self.ivar = None
                else:
                    log.info("Read inverse variance from HDU 1 of " + filename)
            else:
                self.ivar = None
        except:
            log.error("Unable to read data and header from " + filename)
            self.data = None
            self.header = None
            self.ivar = None

    def write(self, filename, clobber=True):
        """
        Image.write(outfilename, clobber=True)
        
        Creates a primary HDU using self.data and self.header, and
        attempts to write to outfilename.  If self.ivar is not None,
        append self.ivar as a second HDU before writing to a file.       
        clobber is provided as a  keyword to pyf.HDUList.writeto. 
        """
        try:
            out = pyf.HDUList(pyf.PrimaryHDU(self.data, self.header))
            if self.ivar is not None:
                out.append(pyf.PrimaryHDU(self.ivar))
            try:
                out.writeto(filename, clobber=clobber)
                log.info("Writing data to " + filename)
                
            except:
                log.error("Unable to write FITS file " + filename)
        except:
            log.error("Unable to create HDU from data and header")
