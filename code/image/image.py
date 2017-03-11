
from astropy.io import fits
import numpy as np
import logging
from datetime import date

log = logging.getLogger('main')

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

    def __init__(self, filename='', data=None, ivar=None, chisq=None, 
                 header=fits.PrimaryHDU().header, extrahead=None,
                 reads=None, flags=None):
        '''
        Image initialization
        
        Parameters
        ----------
        filename: string
            Name of input file
        data: ndarray
            Numpy ndarray containing your data. Can be multi-dimensional.
        ivar: ndarray
            Numpy ndarray containing the inverse variance of the data. Should be same shape as data
        chisq: ndarray
            Numpy ndarray containing chisq value for each ramp fit to the data. Should be same shape as data
        header: `PrimaryHDU` header
            Empty header instance
        extraheader: `PrimaryHDU` header
            Placeholder for header from original ramp
        reads: ndarray
        flags: ndarray
        '''
        self.data = data
        self.ivar = ivar
        self.chisq = chisq
        self.header = header
        self.reads = reads
        self.flags = flags
        self.filename = filename
        self.extrahead = extrahead

        if data is None and filename != '':
            self.load(filename)
                
    def load(self, filename, loadbadpixmap=False):
        """
        Image.load(outfilename)
        
        Read the first HDU with data from filename into self.data, and
        HDU[0].header into self.header.  If there is more than one HDU
        with data, attempt to read the second HDU with data into
        self.ivar.
        
        Parameters
        ----------
        filename: string
            Name of input file
        loadbadpixmap: boolean
            When True, loads the bad pixel map at `calibrations/mask.fits`

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
            elif loadbadpixmap:
                self.ivar = fits.open('calibrations/mask.fits')[0].data
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
        clobber is provided as a keyword to fits.HDUList.writeto.
        
        Parameters
        ----------
        filename: string
            Name of destination file
        clobber: boolean
            When True, overwrites if file already exists
        """
        
        hdr = fits.PrimaryHDU().header
        today = date.today().timetuple()
        yyyymmdd = '%d%02d%02d' % (today[0], today[1], today[2])
        hdr['date'] = (yyyymmdd, 'File creation date (yyyymmdd)')

        for i, key in enumerate(self.header):
            hdr.append((key, self.header[i], self.header.comments[i]), end=True)

        out = fits.HDUList(fits.PrimaryHDU(None, hdr))
        out.append(fits.PrimaryHDU(self.data.astype(np.float32),hdr))
        if self.ivar is not None:
            out.append(fits.PrimaryHDU(self.ivar.astype(np.float32),hdr))
        if self.chisq is not None:
            out.append(fits.PrimaryHDU(self.chisq.astype(np.float32),hdr))
        if self.flags is not None:
            out.append(fits.PrimaryHDU(self.flags),hdr)

        if self.extrahead is not None:
            try:
                out.append(fits.PrimaryHDU(None, self.extrahead))
            except:
                log.warn("Extra header in image class must be a FITS header.")

        try:
            out.writeto(filename, clobber=clobber)
            log.info("Writing data to " + filename)
                
        except:
            log.error("Unable to write FITS file " + filename)
