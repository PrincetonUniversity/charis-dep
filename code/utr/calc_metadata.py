from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
import numpy as np
import re
import time

def _fetch(key, filename, comment=None, newkey=None):

    """
    Private function _fetch fetches a key from HDU 0 of a file.  It
    returns a tuple suitable for appending to a FITS header.

    Inputs:
    1. key       string, the name of the FITS header keyword
    2. filename  string, the file to be opened with astropy.io.fits.open(filename)
    Optional inputs:
    1. comment   string, comment to use for the entry.  Default None, 
                 i.e. use the comment in the original header
    2. newkey    string, keyword to return.  Default None, i.e., the
                 keyword to return is the same as the keyword in the 
                 original header.
    Ouptut: 
    1. (key, value, comment) tuple for use with fits.header.append

    If the file cannot be opened or the key is not in the header, the
    function will return a value of 'unavailable'.

    """

    try:
        head = fits.open(filename)[0].header
        val, comment0 = [head[key], head.comments[key]]
    except:
        if newkey is None:
            return (key, 'unavailable', comment)
        else:
            return (newkey, 'unavailable', comment)

    if comment is None and newkey is None:
        return (key, val, comment0)
    elif newkey is None:
        return (key, val, comment)
    elif comment is None:
        return (newkey, val, comment0)
    else:
        return (newkey, val, comment)
        

def metadata(filename, header=fits.PrimaryHDU().header, clear=True):
    
    """
    Function metadata populates a FITS header (creating a new one if
    necessary) with important information about an observation.  

    Input:
    1. filename  string to be called with astropy.io.fits.open(filename)
    Optional Input:
    1. header    fits header object to append to (default = create new)

    Output:
    header       fits header with additional data

    The main calculation in this routine is for the parallactic angle;
    other parameters are trivially computed or simply copied from the
    original FITS header.  Entries that cannot be computed or found in
    the original header will be given values of NaN or 'unavailable'.
    Times and parallactic angles are computed for the mean time of
    exposure, i.e, halfway between the time of the first and last
    reads.

    """

    if clear:
        header.clear()

    header.append(('comment', ''), end=True)
    header.append(('comment', '*'*60), end=True)
    header.append(('comment', '*'*18 + ' Time and Pointing Data ' + '*'*18), end=True)
    header.append(('comment', '*'*60), end=True)
    header.append(('comment', ''), end=True)

    try:
        origname = re.sub('.*CRSA', '', re.sub('.fits', '', filename))
        header.append(('origname', origname, 'Original file ID number'), end=True)
    except:
        pass

    ####################################################################
    # Attempt to get the mean time of the exposure.  Try three things:
    # 1. The mean of mjd-str and mjd-end in the main header (HDU 0)
    # 2. mjd in the main header (HDU 0)
    # 3. The mean acquisition time in the headers of the individual 
    #    reads, computed as acqtime in HDU 1 plus 1.48s/2*nreads
    ####################################################################

    mjd_ok = True
    try:
        head = fits.open(filename)[0].header
        try:
            mean_mjd = 0.5*(head['mjd-str'] + head['mjd-end'])
        except:
            try:
                mean_mjd = head['mjd'] + 1.48*0.5*len(fits.open(filename))/86400
            except:
                ########################################################
                # Note: acqtime is unreliable--doesn't always update.
                ########################################################
                #head1 = fits.open(filename)[1].header
                #mean_mjd = head1['acqtime'] - 2400000.5
                #mean_mjd += 1.48*0.5*len(fits.open(filename))/86400
                ########################################################
                # This is pretty bad: use the checksum time of the
                # middle read as the time stamp of last resort.
                ########################################################
                head1 = fits.open(filename)[len(fits.open(filename))//2].header
                t = head1.comments['checksum'].split()[-1]
                t = Time(t, format='isot')
                t.format = 'mjd'
                mean_mjd = float(str(t))                
    except:
        mjd_ok = False
        mean_mjd = np.nan
        utc_date = 'unavailable'
        utc_time = 'unavailable'

    pos_ok = True

    ####################################################################
    # Need RA and Dec to compute parallactic angle
    ####################################################################

    try:
        head = fits.open(filename)[0].header
        ra, dec = [head['ra'], head['dec']]
    except:
        #ra, dec = ['05:02:27.5438', '+07:27:39.265']
 	#ra, dec = ['04:37:36.182', '-02:28:25.87']
        pos_ok = False
        
    if mjd_ok:

        ################################################################
        # Subaru's coordinates in degrees
        ################################################################
        
        lng, lat = [-155.4760187, 19.825504]
        subaru = (str(lng) + 'd', str(lat) + 'd')
        t = Time(mean_mjd, format='mjd', location=subaru)
       
        if pos_ok:

            ############################################################
            # Precess from J2000 to the appropriate epoch
            ############################################################

            c = coord.SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='fk5')
        
            equinox = 'J%.5f' %(2000 + (mean_mjd - 51544.5)/365.25)
            c = c.transform_to(coord.FK5(equinox=equinox))

        ################################################################
        # Compute hour angle to get parallactic angle
        ################################################################

            ha =  (t.sidereal_time('apparent') - c.ra).rad
            lat = lat*np.pi/180
            
            pa = -np.arctan2(-np.sin(ha), np.cos(c.dec.rad)*np.tan(lat)
                              - np.sin(c.dec.rad)*np.cos(ha))
            pa = float(pa%(2*np.pi))
        else:
            pa = np.nan

        t.format = 'isot'
        utc_date = str(t).split('T')[0]
        utc_time = str(t).split('T')[1]
    else:
        pa = np.nan

    if not np.isfinite(mean_mjd):
        mean_mjd = utc_date = utc_time = 'unavailable'

    header['mjd'] = (mean_mjd, 'Mean MJD of exposure')    
    header['utc-date'] = (utc_date, 'UTC date of exposure')  
    header['utc-time'] = (utc_time, 'Mean UTC time of exposure')

    ####################################################################
    # Attempt to fetch useful/important keywords from the original
    # file's FITS header
    ####################################################################

    header.append(_fetch('ra', filename, comment='RA of telescope pointing'))
    header.append(_fetch('dec', filename, comment='DEC of telescope pointing'))

    if np.isfinite(pa):
        header['parang'] = (pa*180/np.pi, 'Mean parallactic angle (degrees)')
    else:
        header['parang'] = ('unavailable', 'Mean parallactic angle (degrees)')
    header.append(_fetch('d_imrpap', filename, comment='Image rotator pupil position angle (degrees)'))

    header.append(_fetch('HIERARCH CHARIS.FILTER.NAME', filename, 
                         comment='CHARIS filter name', newkey='filtname'))
    header.append(_fetch('HIERARCH CHARIS.FILTER.SLOT', filename, 
                         comment='CHARIS filter slot', newkey='filtpos'))
    header.append(_fetch('HIERARCH CHARIS.SHUTTER', filename, 
                         comment='CHARIS shutter position', newkey='shutter'))

    return header
