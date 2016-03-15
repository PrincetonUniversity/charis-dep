try:
    from astropy.io import fits
except:
    import pyfits as fits
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

    def __init__(self, filename=None, data=None, ivar=None, chisq=None, header=None,
                 reads=None, flags=None, destriped=False, flatfielded=False):
        self.destriped = destriped
        self.flatfielded = flatfielded
        
        self.data = data
        self.ivar = ivar
        self.chisq = chisq
        self.header = header
        self.reads = reads
        self.flags = flags
        self.filename = filename

        if filename is not None:
            self.load(filename)
                
    def load(self, filename):
        """
        Image.load(outfilename)
        
        Read the first HDU with data from filename into self.data, and
        HDU[0].header into self.header.  If there is more than one HDU
        with data, attempt to read the second HDU with data into
        self.ivar.

        """
        try:
            self.filename = filename
            hdulist = fits.open(filename)
            self.header = hdulist[0].header
            if hdulist[0].data is not None:
                i_data = 0
            else:
                i_data = 1

            self.data = hdulist[i_data].data
            log.info("Read data from HDU " + str(i_data) + " of " + filename)

            if len(hdulist) > i_data + 1:
                self.ivar = hdulist[i_data + 1].data
                if self.ivar.shape != self.data.shape:
                    log.error("Error: data (HDU " + str(i_data) +\
                              ") and inverse variance (HDU " + str(i_data +\
                              1) + ") have different shapes in file " + filename)
                    self.ivar = None
                else:
                    log.info("Read inverse variance from HDU " + str(i_data + 1) + " of " + filename)
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
        clobber is provided as a  keyword to fits.HDUList.writeto.
        """
        try:
            out = fits.HDUList(fits.PrimaryHDU(self.data, self.header))
            if self.ivar is not None:
                out.append(fits.PrimaryHDU(self.ivar))
            if self.chisq is not None:
                out.append(fits.PrimaryHDU(self.chisq))
            if self.flags is not None:
                out.append(fits.PrimaryHDU(self.flags))
            try:
                out.writeto(filename, clobber=clobber)
                log.info("Writing data to " + filename)
                
            except:
                log.error("Unable to write FITS file " + filename)
        except:
            log.error("Unable to create HDU from data and header")
