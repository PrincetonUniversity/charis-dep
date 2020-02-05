from __future__ import division
from builtins import str
from past.utils import old_div
import re
import time
import collections
import warnings


import numpy as np

import pandas as pd

from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from astropy.utils import iers
iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')

from charis.tools import compute_times, compute_angles


def get_headerval_with_exeption(header, keyword, default_val):
    try:
        val = header[keyword]
    except KeyError:
        val = default_val
    except IOError:
        val = default_val
    return val


def header_dataframe(filename):
    warnings.filterwarnings("ignore")

    header_list = collections.OrderedDict(
        [('OBJECT', ('OBJECT', 'N/A')),
         ('RA', ('RA', -10000)),
         ('DEC', ('DEC', -10000)),
         ('TEL TARG ALPHA', ('HIERARCH ESO TEL TARG ALPHA', -10000)),
         ('TEL TARG DELTA', ('HIERARCH ESO TEL TARG DELTA', -10000)),
         ('TEL TARG PMA', ('HIERARCH ESO TEL TARG PMA', -10000)),
         ('TEL TARG PMD', ('HIERARCH ESO TEL TARG PMD', -10000)),
         ('INS4 DROT2 RA', ('HIERARCH ESO INS4 DROT2 RA', -10000)),
         ('INS4 DROT2 DEC', ('HIERARCH ESO INS4 DROT2 DEC', -10000)),
         ('INS4 DROT2 MODE', ('HIERARCH ESO INS4 DROT2 MODE', 'N/A')),
         ('INS4 DROT2 BEGIN', ('HIERARCH ESO INS4 DROT2 BEGIN', -10000)),
         ('INS4 DROT2 END', ('HIERARCH ESO INS4 DROT2 END', -10000)),
         ('TEL GEOELEV', ('HIERARCH ESO TEL GEOELEV', -10000)),
         ('TEL GEOLAT', ('HIERARCH ESO TEL GEOLAT', -10000)),
         ('TEL GEOLON', ('HIERARCH ESO TEL GEOLON', -10000)),
         ('TEL ALT', ('HIERARCH ESO TEL ALT', -10000)),
         ('EXPTIME', ('EXPTIME', -10000)),
         ('DET SEQ1 DIT', ('HIERARCH ESO DET SEQ1 DIT', -10000)),
         ('DET DIT DELAY', ('HIERARCH ESO DET DITDELAY', -10000)),
         ('DET NDIT', ('HIERARCH ESO DET NDIT', -10000)),
         ('INS COMB IFLT', ('HIERARCH ESO INS COMB IFLT', 'N/A')),
         ('INS COMB ICOR', ('HIERARCH ESO INS COMB ICOR', 'N/A')),
         ('INS4 FILT2 NAME', ('HIERARCH ESO INS4 FILT2 NAME', 'N/A')),
         ('INS4 SHUT ST', ('HIERARCH ESO INS4 SHUT ST', 'N/A')),
         ('INS2 COMB IFS', ('HIERARCH ESO INS2 COMB IFS', 'N/A')),
         ('INS2 OPTI2 ID', ('HIERARCH ESO INS2 OPTI2 ID', 'N/A')),
         ('DPR TYPE', ('HIERARCH ESO DPR TYPE', 'N/A')),
         ('DPR TECH', ('HIERARCH ESO DPR TECH', 'N/A')),
         ('DET ID', ('HIERARCH ESO DET ID', 'ZPL')),
         ('SEQ ARM', ('HIERARCH ESO SEQ ARM', 'N/A')),
         ('INS4 COMB ROT', ('HIERARCH ESO INS4 COMB ROT', 'N/A')),
         ('INS1 PAC X', ('HIERARCH ESO INS1 PAC X', -10000)),
         ('INS1 PAC Y', ('HIERARCH ESO INS1 PAC Y', -10000)),
         ('DET READ CURNAME', ('HIERARCH ESO DET READ CURNAME', 'N/A')),
         ('OCS WAFFLE AMPL', ('HIERARCH ESO OCS WAFFLE AMPL', -10000)),
         ('OCS WAFFLE ORIENT', ('HIERARCH ESO OCS WAFFLE ORIENT', 'N/A')),
         ('INS4 OPTI22 NAME', ('HIERARCH ESO INS4 OPTI22 NAME', 'N/A')),
         ('TEL AMBI TAU0', ('HIERARCH ESO TEL AMBI TAU0', -10000)),
         ('TEL AMBI FWHM START', ('HIERARCH ESO TEL AMBI FWHM START', -10000)),
         ('TEL AMBI FWHM END', ('HIERARCH ESO TEL AMBI FWHM END', -10000)),
         ('TEL AIRM START', ('HIERARCH ESO TEL AIRM START', -10000)),
         ('OBS PROG ID', ('HIERARCH ESO OBS PROG ID', 'N/A')),
         ('OBS PI-COI NAME', ('HIERARCH ESO OBS PI-COI NAME', 'N/A')),
         ('OBS ID', ('HIERARCH ESO OBS ID', -10000)),
         ('ORIGFILE', ('ORIGFILE', 'N/A')),
         ('DET SEQ UTC', ('HIERARCH ESO DET SEQ UTC', 'N/A')),
         ('DET FRAM UTC', ('HIERARCH ESO DET FRAM UTC', 'N/A')),
         ('DATE-OBS', ('DATE-OBS', 'N/A')),
         ('DATE', ('DATE', 'N/A')),
         ('MJD-OBS', ('MJD-OBS', -10000))])

    orig_hdr = fits.getheader(filename)
    header = orig_hdr.copy()
    for key in header_list.keys():
        header[key] = get_headerval_with_exeption(
            orig_hdr, header_list[key][0], header_list[key][1])

    header_table = collections.OrderedDict()
    for key in header_list.keys():
        header_table[key] = []

    for key in header_list.keys():
        header_table[key].append(get_headerval_with_exeption(
            orig_hdr, header_list[key][0], header_list[key][1]))
    frames_info = pd.DataFrame(header_table)
    frames_info['DATE-OBS'] = pd.to_datetime(frames_info['DATE-OBS'], utc=True)
    frames_info['DATE'] = pd.to_datetime(frames_info['DATE'], utc=True)
    frames_info['DET FRAM UTC'] = pd.to_datetime(frames_info['DET FRAM UTC'], utc=True)

    return frames_info, header_table


def _fetch(key, filename, comment=None, newkey=None):
    """
    Private function _fetch fetches a key from HDU 0 of a file.  It
    returns a tuple suitable for appending to a FITS header.

    Parameters
    ----------

    key:       string
        The name of the FITS header keyword
    filename:  string
        The file to be opened with astropy.io.fits.open(filename)
    comment:   string
        Comment to use for the entry.  Default None, i.e. use the comment in the original header
    newkey:    string
        Keyword to return.  Default None, i.e., the keyword to return is the same as the keyword in the
                 original header.
    Returns
    -------
    (key, value, comment): tuple
        for use with fits.header.append

    Notes
    -----
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


def metadata(filename, header=None, clear=True, version=None):
    """
    Function metadata populates a FITS header (creating a new one if
    necessary) with important information about an observation.

    Parameters
    ----------
    filename:  string
        To be called with astropy.io.fits.open(filename)
    header:    fits header
        Fits header object to append to (default = create new)
    clear:    boolean
        Whether to clear the header or not

    Returns
    -------
    header:    fits header
        With additional data

    Notes
    -----
    The main calculation in this routine is for the parallactic angle;
    other parameters are trivially computed or simply copied from the
    original FITS header.  Entries that cannot be computed or found in
    the original header will be given values of NaN or 'unavailable'.
    Times and parallactic angles are computed for the mean time of
    exposure, i.e, halfway between the time of the first and last
    reads.

    """
    if header is None:
        header = fits.PrimaryHDU().header
    if clear:
        header.clear()
    if version is not None:
        header.append(('version', version, 'Pipeline Version'), end=True)

    header.append(('comment', ''), end=True)
    header.append(('comment', '*' * 60), end=True)
    header.append(('comment', '*' * 18 + ' Time and Pointing Data ' + '*' * 18), end=True)
    header.append(('comment', '*' * 60), end=True)
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
            mean_mjd = 0.5 * (head['mjd-str'] + head['mjd-end'])
        except:
            try:
                mean_mjd = head['mjd'] + 1.48 * 0.5 * len(fits.open(filename)) / 86400
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
                head1 = fits.open(filename)[len(fits.open(filename)) // 2].header
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
        #ra, dec = ['23:07:28.83', '+21:08:02.51']
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

            equinox = 'J%.5f' % (2000 + old_div((mean_mjd - 51544.5), 365.25))
            c = c.transform_to(coord.FK5(equinox=equinox))

            ############################################################
            # Compute hour angle to get parallactic angle.  Put this
            # within a try/except block in case astropy fails to fetch
            # the files necessary to compute the sidereal time.
            ############################################################

            try:
                ha = (t.sidereal_time('apparent') - c.ra).rad
                lat = lat * np.pi / 180

                pa = -np.arctan2(-np.sin(ha), np.cos(c.dec.rad) * np.tan(lat)
                                 - np.sin(c.dec.rad) * np.cos(ha))
                pa = float(pa % (2 * np.pi))
            except:
                pa = np.nan
        else:
            pa = np.nan

        t.format = 'isot'
        utc_date = str(t).split('T')[0]
        utc_time = str(t).split('T')[1]
    else:
        pa = np.nan

    if not np.isfinite(mean_mjd):
        mean_mjd = utc_date = utc_time = 'unavailable'

    for key in ['Name', 'object', 'imagetyp', 'telescop', 'exptime']:
        header.append(_fetch(key, filename))

    header.append(('mjd', mean_mjd, 'Mean MJD of exposure'))
    header.append(('utc-date', utc_date, 'UTC date of exposure'))
    header.append(('utc-time', utc_time, 'Mean UTC time of exposure'))

    ####################################################################
    # Attempt to fetch useful/important keywords from the original
    # file's FITS header
    ####################################################################

    header.append(_fetch('ra', filename, comment='RA of telescope pointing'))
    header.append(_fetch('dec', filename, comment='DEC of telescope pointing'))

    for key in ['azimuth', 'altitude']:
        header.append(_fetch(key, filename))

    #header['ra'] = (ra, 'RA of telescope pointing')
    #header['dec'] = (dec, 'DEC of telescope pointing')

    if np.isfinite(pa):
        header['parang'] = (pa * 180 / np.pi, 'Mean parallactic angle (degrees)')

    else:
        header['parang'] = ('unavailable', 'Mean parallactic angle (degrees)')
    header.append(_fetch('d_imrpap', filename, comment='Image rotator pupil position angle (degrees)'))

    filtnamekeys = ['Y_FLTNAM', 'FILTER01', 'HIERARCH CHARIS.FILTER.NAME']
    filtposkeys = ['Y_FLTSLT', 'HIERARCH CHARIS.FILTER.SLOT']
    prismnames = ['Y_PRISM', 'DISPERSR', 'Y_GRISM']
    shutterkeys = ['Y_SHUTTR', 'HIERARCH CHARIS.SHUTTER']

    for key in filtnamekeys:
        card = _fetch(key, filename, comment='CHARIS filter name', newkey='filtname')
        if card[1] != 'unavailable' or key == filtnamekeys[-1]:
            header.append(card)
            break
    for key in filtposkeys:
        card = _fetch(key, filename, comment='CHARIS filter slot', newkey='filtslot')
        if card[1] != 'unavailable' or key == filtposkeys[-1]:
            header.append(card)
            break
    for key in prismnames:
        card = _fetch(key, filename, comment='CHARIS prism (lo/hi/out)', newkey='prism')
        if card[1] != 'unavailable' or key == prismnames[-1]:
            header.append(card)
            break
    for key in shutterkeys:
        card = _fetch(key, filename, comment='CHARIS shutter position', newkey='shutter')
        if card[1] != 'unavailable' or key == shutterkeys[-1]:
            header.append(card)
            break

    for key in ['X_LYOT', 'X_CHAPKO', 'X_FPM', 'X_GRDST', 'X_GRDSEP', 'X_GRDAMP']:
        header.append(_fetch(key, filename))

    return header


def metadata_SPHERE(filename, dit_idx=None, header=None, clear=True, version=None):
    """
    Function metadata populates a FITS header (creating a new one if
    necessary) with important information about an observation.

    Parameters
    ----------
    filename:  string
        To be called with astropy.io.fits.open(filename)
    header:    fits header
        Fits header object to append to (default = create new)
    clear:    boolean
        Whether to clear the header or not

    Returns
    -------
    header:    fits header
        With additional data

    Notes
    -----
    The main calculation in this routine is for the parallactic angle;
    other parameters are trivially computed or simply copied from the
    original FITS header.  Entries that cannot be computed or found in
    the original header will be given values of NaN or 'unavailable'.
    Times and parallactic angles are computed for the mean time of
    exposure, i.e, halfway between the time of the first and last
    reads.

    """
    warnings.filterwarnings("ignore")
    if header is None:
        header = fits.PrimaryHDU().header
    if clear:
        header.clear()
    if version is not None:
        header.append(('version', version, 'Pipeline Version'), end=True)

    # try:
    #     origname = re.sub('.*CRSA', '', re.sub('.fits', '', filename))
    #     header.append(('origname', origname, 'Original file ID number'), end=True)
    # except:
    #     pass

    # COMPUTE PA HERE

    # if not np.isfinite(mean_mjd):
    #     mean_mjd = utc_date = utc_time = 'unavailable'
    #
    # for key in ['Name', 'object', 'imagetyp', 'telescop', 'exptime']:
    #     header.append(_fetch(key, filename))
    #
    # header.append(('mjd', mean_mjd, 'Mean MJD of exposure'))
    # header.append(('utc-date', utc_date, 'UTC date of exposure'))
    # header.append(('utc-time', utc_time, 'Mean UTC time of exposure'))
    #
    # ####################################################################
    # # Attempt to fetch useful/important keywords from the original
    # # file's FITS header
    # ####################################################################

    # header.append(_fetch('RA', filename, comment='RA of telescope pointing'))
    # header.append(_fetch('DEC', filename, comment='DEC of telescope pointing'))

    frames_info, header_table = header_dataframe(filename)
    header_table = Table(header_table)

    for key in header_table.keys():
        header[key] = header_table[0][key]

    # header.append(('comment', ''), end=True)
    # header.append(('comment', '*' * 60), end=True)
    # header.append(('comment', '*' * 18 + ' Time and Pointing Data ' + '*' * 18), end=True)
    # header.append(('comment', '*' * 60), end=True)
    # header.append(('comment', ''), end=True)

    if dit_idx == [1, None]:
        dit_idx = 1
    try:
        compute_times(frames_info, idx=dit_idx)
        compute_angles(frames_info)

        header['PARANG START'] = (frames_info['PARANG START'][0], '')
        header['PARANG'] = (frames_info['PARANG'][0], '')
        header['PARANG END'] = (frames_info['PARANG END'][0], '')
        header['PUPIL OFFSET'] = (frames_info['PUPIL OFFSET'][0], '')
        header['DEROT ANGLE'] = (frames_info['DEROT ANGLE'][0], '')
        header['RA2'] = (frames_info['RA'][0], 'Derotator adjusted')
        header['DEC2'] = (frames_info['DEC'][0], 'Derotator adjusted')
    except:
        print('Unable to compute parallactic angle for {}'.format(filename))

    return header


def addWCS(header, xpix, ypix, xpixscale=old_div(-0.015, 3600.), ypixscale=old_div(0.015, 3600.), extrarot=113.):
    '''
    Add the proper keywords to align the cube into the World Coordinate System.
    This modifies the variable `header` in place

    Parameters
    ----------

    header: FITS header
        Header to modify. Needs to already contain 'ra' and 'dec' keywords
    xpix:   float
        X coordinate of reference pixel
    ypix:   float
        Y coordinate of reference pixel
    xpixscale:   float
        Plate scale in the X direction in degrees
    Ypixscale:   float
        Plate scale in the Y direction in degrees
    extrarot:   float
        Additional rotation angle in degrees

    '''
    ra = header['ra']
    dec = header['dec']
    try:
        c = coord.SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='fk5')
    except:
        return

    ####################################################################
    # Compute the FITS header rotation and scale matrix to properly
    # align the image in FITS viewers
    ####################################################################

    try:
        header.append(('comment', ''), end=True)
        header.append(('comment', '*' * 60), end=True)
        header.append(('comment', '*' * 28 + ' WCS ' + '*' * 27), end=True)
        header.append(('comment', '*' * 60), end=True)
        header.append(('comment', ''), end=True)

        header.append(('XPIXSCAL', xpixscale, 'Degrees/pixel'), end=True)
        header.append(('YPIXSCAL', ypixscale, 'Degrees/pixel'), end=True)
        header.append(('CTYPE1', 'RA---TAN', 'first parameter RA  ,  projection TANgential'), end=True)
        header.append(('CTYPE2', 'DEC--TAN', 'second parameter DEC,  projection TANgential'), end=True)
        header.append(('CRVAL1', c.ra.deg, 'Reference X pixel value'), end=True)
        header.append(('CRVAL2', c.dec.deg, 'Reference Y pixel value'), end=True)
        header.append(('CRPIX1', xpix, 'Reference X pixel'), end=True)
        header.append(('CRPIX2', ypix, 'Reference Y pixel'), end=True)
        header.append(('EQUINOX', 2000, 'Equinox of coordinates'), end=True)
        header.append(('TOT_ROT', -1 * (header['PARANG'] + extrarot), 'Total rotation angle (degrees)'), end=True)

        angle = np.pi * (header['TOT_ROT']) / 180.
        header.append(('CD1_1', np.cos(angle) * xpixscale, 'Rotation matrix coefficient'), end=True)
        header.append(('CD1_2', np.sin(angle) * xpixscale, 'Rotation matrix coefficient'), end=True)
        header.append(('CD2_1', -np.sin(angle) * ypixscale, 'Rotation matrix coefficient'), end=True)
        header.append(('CD2_2', np.cos(angle) * ypixscale, 'Rotation matrix coefficient'), end=True)
    except:
        return

    return
