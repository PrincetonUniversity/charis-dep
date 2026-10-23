""" Partially adapted from A. Vigan's VLTPF Pipeline
Commit: f20dbcc on Feb 6 """

from builtins import zip

import astropy.coordinates as coord
import numpy as np
from astropy import units as u
from astropy.convolution import convolve
from astropy.time import Time
from scipy.stats import scoreatpercentile

global_cmap = 'inferno'


def expected_spectrum(stellar_temperature, wavelength, transmission):
    """Create normalized stellar spectrum after transmission.

    Parameters
    ----------
    stellar_temperature : class:`~astropy.units.Quantity`
        Temperature of the host star.
    wavelength : class:`~astropy.units.Quantity`
        Wavelength of transmission curve.
    transmission : array_like
        Total transmission of instrument + atmosphere.

    Returns
    -------
    array
        Black body spectrum multiplied by tranmission.

    """

    from astropy.modeling.physical_models import BlackBody
    bb = BlackBody(temperature=stellar_temperature)

    spectrum = bb(wavelength)
    spectrum /= np.max(spectrum)
    transmitted_spectrum = spectrum * transmission
    transmitted_spectrum /= np.max(transmitted_spectrum)

    transmission_curve = np.vstack(
        [wavelength.value, transmitted_spectrum.value]).T

    return transmission_curve


def sph_ifs_correct_spectral_xtalk(img, mask=None):
    '''
    Corrects a IFS frame from the spectral crosstalk

    This routines corrects for the SPHERE/IFS spectral crosstalk at
    small scales and (optionally) at large scales. This correction is
    necessary to correct the signal that is "leaking" between
    lenslets. See Antichi et al. (2009ApJ...695.1042A) for a
    theoretical description of the IFS crosstalk. Some informations
    regarding its correction are provided in Vigan et al. (2015), but
    this procedure still lacks a rigorous description and performance
    analysis.

    Since the correction of the crosstalk involves a convolution by a
    kernel of size 41x41, the values at the edges of the frame depend
    on how you choose to apply the convolution. Current implementation
    is EDGE_TRUNCATE. In other parts of the image (i.e. far from the
    edges), the result is identical to original routine by Dino
    Mesa. Note that in the original routine, the convolution that was
    coded did not treat the edges in a clean way defined
    mathematically. The scipy.ndimage.convolve() function offers
    different possibilities for the edges that are all documented.

    Parameters
    ----------
    img : array_like
        Input IFS science frame

    Returns
    -------
    img_corr : array_like
        Science frame corrected from the spectral crosstalk

    '''

    # definition of the dimension of the matrix
    sepmax = 20
    dim = sepmax*2+1
    bfac = 0.727986/1.8

    # defines a matrix to be used around each pixel
    # (the value of the matrix is lower for greater
    # distances form the center.
    x, y = np.meshgrid(np.arange(dim)-sepmax, np.arange(dim)-sepmax)
    rdist = np.sqrt(x**2 + y**2)
    kernel = 1 / (1+rdist**3 / bfac**3)
    kernel[(np.abs(x) <= 1) & (np.abs(y) <= 1)] = 0

    mask = mask.astype('bool')
    mask[0:4, :] = False
    mask[:, 0:4] = False
    mask[-4:, :] = False
    mask[:, -4:] = False

    # convolution and subtraction
    print('> compute convolution')
    conv = convolve(
        img, kernel, boundary='fill', fill_value=0.0,
        nan_treatment='interpolate',
        normalize_kernel=False, mask=mask, preserve_nan=False,
        normalization_zero_tol=1e-08)
    # conv_scipy = ndimage.convolve(img, kernel, mode='reflect')
    print('> subtract convolution')
    img_corr = img - conv

    return img_corr, conv


def sph_ifs_fix_badpix(img, bpm):
    '''
    Clean the bad pixels in an IFU image

    Extremely effective routine to remove bad pixels in IFS data. It
    goes through all bad pixels and fit a line beween the first good
    pixels encountered along the same column as the bad pixel,
    i.e. along the spectral axis of each micro-spectrum. Works very
    well because as zeroth-order the spectrum is very smooth and can
    be approximated by a line over one (or a few) bad pixels.

    Parameters
    ----------
    img : array_like
        The image to be cleaned

    bpm : array_like
        Bad pixel map

    logger : logHandler object
        Log handler for the reduction. Default is root logger

    Returns
    -------
    img_clean : array_like
        The cleaned image
    '''

    # copy the original image
    # print('> copy input image')
    img_clean = img.copy()

    # extension over which the good pixels will be looked for along
    # the spectral direction starting from the bad pixel
    ext = 10

    # remove edges in bad pixel map
    bpm[:ext+1, :] = 0
    bpm[:, :ext+1] = 0
    bpm[-ext-1:, :] = 0
    bpm[:, -ext-1:] = 0

    # use NaN for identifying bad pixels directly in the image
    img_clean[bpm == 1] = np.nan

    # static indices for searching good pixels and for the linear fit
    idx = np.arange(2*ext+1)
    idx_lh = np.arange(ext)+1

    # loop over bad pixels
    # print('> loop over bad pixels')
    badpix = np.where(bpm == 1)
    for y, x in zip(badpix[0], badpix[1]):
        # extract sub-region along the spectral direction
        sub = img_clean[y-ext:y+ext+1, x]

        # sub-regions "above" and "below" the bad pixel
        sub_low = np.flip(img_clean[y-ext:y, x], axis=0)
        sub_hig = img_clean[y+1:y+1+ext, x]

        # if any of the two is completely bad: skip
        # occurs only in the vignetted areas
        if np.all(np.isnan(sub_low)) or np.all(np.isnan(sub_hig)):
            continue

        # indices of the first good pixels "above" and "below" the bad pixel
        imin_low = idx_lh[~np.isnan(sub_low)].min()
        imin_hig = idx_lh[~np.isnan(sub_hig)].min()

        # linear fit
        xl = idx[ext-imin_low]
        yl = sub[ext-imin_low]

        xh = idx[ext+imin_hig]
        yh = sub[ext+imin_hig]

        a = (yh - yl) / (xh - xl)
        b = yh - a*xh

        fit = a*idx + b

        # replace bad pixel with the fit
        img_clean[y-imin_low+1:y+imin_hig, x] = fit[ext-imin_low+1:ext+imin_hig]

    # put back original value in regions that could not be corrected
    mask = np.isnan(img_clean)
    img_clean[mask] = img[mask]

    return img_clean


def fit_background(image, components, bgmask, outlier_percentiles=[2, 98]):
    arr = np.reshape(components, (components.shape[0], -1))
    # Mask: background mask and PCA components are reliable
    bgmask = np.logical_and(bgmask, components[1] != 0)
    non_outliers = (image > scoreatpercentile(image[bgmask], outlier_percentiles[0])) \
        * (image < scoreatpercentile(image[bgmask], outlier_percentiles[1]))
    bgmask = np.logical_and(bgmask, non_outliers)
    mask = np.reshape(bgmask, -1)

    coef = np.linalg.lstsq((arr[:, mask]).T, np.reshape(image, -1)[mask], rcond=None)[0]

    # This is the fit that we want
    bgfit = np.sum(components*coef[:, np.newaxis, np.newaxis], axis=0)

    return bgfit, coef


def _sexagesimal_to_deg(values, hours):
    """ESO ``HHMMSS.s`` (``hours=True``) or ``DDMMSS.s`` numbers in degrees, e.g. ``INS4 DROT2 RA`` or ``INS4 DROT2 DEC``.

    The sign is applied to the whole value, so ``-000530.0`` (-00d05m30s) stays negative.
    """
    values = np.asarray(values, dtype=float)
    sign = np.where(values < 0, -1.0, 1.0)
    values = np.abs(values)
    whole = np.floor(values / 1e4)
    minutes = np.floor((values - whole * 1e4) / 100.0)
    seconds = values - whole * 1e4 - minutes * 100.0
    degrees = sign * (whole + minutes / 60.0 + seconds / 3600.0)
    return degrees * 15.0 if hours else degrees


def parallatic_angle(ha, dec, geolat):
    '''
    Parallactic angle of a source in degrees

    Parameters
    ----------
    ha : array_like
        Hour angle, in hours. Any branch is accepted; it is wrapped to [-12h, 12h).

    dec : float
        Declination, in degrees

    geolat : float
        Observatory declination, in degrees

    Returns
    -------
    pa : array_like
        Parallactic angle values
    '''
    # The branch below needs the hour angle in [-12h, 12h). ``LST - RA`` spans (-24h, 24h),
    # so a sequence during which the sidereal time wraps past 0h would otherwise jump by
    # 360 deg.
    ha = coord.Angle(ha).wrap_at(12 * u.hourangle)
    pa = -np.arctan2(-np.sin(ha),
                     np.cos(dec) * np.tan(geolat) - np.sin(dec) * np.cos(ha))

    if (dec >= geolat):
        pa[ha < 0] += 360 * u.degree

    return np.degrees(pa)


def compute_times(frames_info, idx=None):
    '''
    Compute the various timestamps associated to frames

    Parameters
    ----------
    frames_info : dataframe
        The data frame with all the information on science frames
    '''

    # get necessary values
    time_start = frames_info['DATE-OBS'].values
    time_end = frames_info['DET FRAM UTC'].values
    time_delta = (time_end - time_start) / frames_info['DET NDIT'].values.astype(int)
    DIT = np.array(frames_info['DET SEQ1 DIT'].values.astype(
        float) * 1000, dtype='timedelta64[ms]')

    # calculate UTC time stamps
    if idx is None:
        idx = frames_info.index.get_level_values(0).values  # level 1 in original

    ts_start = time_start + time_delta * idx
    ts = time_start + time_delta * idx + DIT / 2
    ts_end = time_start + time_delta * idx + DIT

    # calculate mjd
    geolon = coord.Angle(frames_info['TEL GEOLON'].values[0], u.degree)
    geolat = coord.Angle(frames_info['TEL GEOLAT'].values[0], u.degree)
    geoelev = frames_info['TEL GEOELEV'].values[0]

    utc = Time(ts_start.astype(str), scale='utc', location=(geolon, geolat, geoelev))
    mjd_start = utc.mjd

    utc = Time(ts.astype(str), scale='utc', location=(geolon, geolat, geoelev))
    mjd = utc.mjd

    utc = Time(ts_end.astype(str), scale='utc', location=(geolon, geolat, geoelev))
    mjd_end = utc.mjd

    # update frames_info
    frames_info['TIME START'] = ts_start
    frames_info['TIME'] = ts
    frames_info['TIME END'] = ts_end

    frames_info['MJD START'] = mjd_start
    frames_info['MJD'] = mjd
    frames_info['MJD END'] = mjd_end


def compute_angles(frames_info, true_north=-1.75):
    '''
    Compute the various angles associated to frames: RA, DEC, parang,
    pupil offset, final derotation angle

    Parameters
    ----------
    frames_info : dataframe
        The data frame with all the information on science frames
    '''

    drot_mode = frames_info['INS4 DROT2 MODE'].unique()
    if len(drot_mode) != 1:
        raise ValueError('Derotator mode has several values in the sequence')

    # Derotator drift before 2016-07-12 (Maire et al. 2016, Eq. 1), from the
    # file-start header values. The angle is 0 mod 180 deg for a synchronised
    # derotator: ALT - 2 DROT2 in pupil tracking, ALT - PARANG - 2 DROT2 + POSANG
    # in field tracking.
    date_fix = Time('2016-07-12')
    drift = 0
    if np.any(frames_info['MJD'].values <= date_fix.mjd):
        try:
            angle = (frames_info['TEL ALT'].values.astype('float')
                     - 2. * frames_info['INS4 DROT2 BEGIN'].values.astype('float'))
            if drot_mode == 'SKY':
                angle += (frames_info['INS4 DROT2 POSANG'].values.astype('float')
                          - frames_info['TEL PARANG START'].values.astype('float'))
            drift = np.degrees(np.arctan(np.tan(np.radians(angle))))
        except KeyError:
            drift = 0
    # Field tracking does not use the parallactic angle, so its drift enters the
    # derotation angle directly.
    pa_correction = 0 if drot_mode == 'SKY' else drift

    # RA/DEC
    ra_drot = frames_info['INS4 DROT2 RA'].values.astype('float')
    ra_hour = coord.Angle(_sexagesimal_to_deg(ra_drot, hours=True) / 15.0, u.hour)
    frames_info['RA'] = ra_hour.degree

    dec_drot = frames_info['INS4 DROT2 DEC'].values.astype('float')
    dec = coord.Angle(_sexagesimal_to_deg(dec_drot, hours=False), u.degree)
    frames_info['DEC'] = dec.value

    geolon = coord.Angle(frames_info['TEL GEOLON'].values[0], u.degree)
    geolat = coord.Angle(frames_info['TEL GEOLAT'].values[0], u.degree)
    geoelev = frames_info['TEL GEOELEV'].values[0]

    location = (geolon, geolat, geoelev)
    # calculate parallactic angles
    utc = Time(frames_info['TIME'].values.astype(str), scale='utc', location=location)
    lst = utc.sidereal_time('apparent')
    ha = (lst - ra_hour).wrap_at(12 * u.hourangle)
    pa = parallatic_angle(ha, dec[0], geolat)
    frames_info['PARANG'] = pa.value + pa_correction
    frames_info['HOUR ANGLE'] = ha.value
    frames_info['LST'] = lst.value

    # Altitude and airmass
    # j2000 = coord.SkyCoord(ra=ra_hour, dec=dec, frame='icrs', obstime=utc)
    # altaz = j2000.transform_to(coord.AltAz(location=location))
    #
    # frames_info['ALTITUDE'] = altaz.alt.value
    # frames_info['AZIMUTH'] = altaz.az.value
    # frames_info['AIRMASS'] = altaz.secz.value

    utc = Time(frames_info['TIME START'].values.astype(str), scale='utc', location=location)
    lst = utc.sidereal_time('apparent')
    ha = (lst - ra_hour).wrap_at(12 * u.hourangle)
    pa = parallatic_angle(ha, dec[0], geolat)
    frames_info['PARANG START'] = pa.value + pa_correction
    frames_info['HOUR ANGLE START'] = ha.value
    frames_info['LST START'] = lst.value

    utc = Time(frames_info['TIME END'].values.astype(str), scale='utc', location=location)
    lst = utc.sidereal_time('apparent')
    ha = (lst - ra_hour).wrap_at(12 * u.hourangle)
    pa = parallatic_angle(ha, dec[0], geolat)
    frames_info['PARANG END'] = pa.value + pa_correction
    frames_info['HOUR ANGLE END'] = ha.value
    frames_info['LST END'] = lst.value

    # calculate parallactic angles
    # utc = Time(frames_info['TIME START'].values.astype(str), scale='utc', location=(geolon, geolat, geoelev))
    # lst = utc.sidereal_time('apparent')
    # ha = lst - ra_hour
    # pa = parallatic_angle(ha, dec[0], geolat)
    # frames_info['PARANG START'] = pa.value + pa_correction
    #
    # utc = Time(frames_info['TIME'].values.astype(str), scale='utc', location=(geolon, geolat, geoelev))
    # lst = utc.sidereal_time('apparent')
    # ha = lst - ra_hour
    # pa = parallatic_angle(ha, dec[0], geolat)
    # frames_info['PARANG'] = pa.value + pa_correction
    #
    # utc = Time(frames_info['TIME END'].values.astype(str), scale='utc', location=(geolon, geolat, geoelev))
    # lst = utc.sidereal_time('apparent')
    # ha = lst - ra_hour
    # pa = parallatic_angle(ha, dec[0], geolat)
    # frames_info['PARANG END'] = pa.value + pa_correction

    # Derotation angles
    #
    # DEROT ANGLE = PARANG + PUP_OFFSET + INSTRUMENT_OFFSET + TRUE_NORTH
    #  PUP_OFFSET = +135.99 +/- 0.11 (ELEV derotator mode, i.e. pupil tracking)
    # In field tracking (SKY) the field does not rotate and North is set by the
    # derotator position angle (Maire et al. 2016; IRDAP, van Holstein et al. 2020):
    # DEROT ANGLE = POSANG + DRIFT + INSTRUMENT_OFFSET + TRUE_NORTH
    #  INSTRUMENT_OFFSET
    #   IFS = -100.48 +/- 0.10
    #   IRD =    0.00 +/- 0.00
    #  TRUE_NORTH = -1.75 +/- 0.08
    #
    instru = frames_info['SEQ ARM'].unique()
    if len(instru) != 1:
        raise ValueError('Sequence is mixing different instruments: {0}'.format(instru))
    if instru == 'IFS':
        instru_offset = -100.48
    elif instru == 'IRDIS':
        instru_offset = 0.0
    else:
        raise ValueError('Unkown instrument {0}'.format(instru))

    parang = frames_info['PARANG']
    if drot_mode == 'ELEV':
        pupoff = 135.99
    elif drot_mode == 'SKY':
        pupoff = frames_info['INS4 DROT2 POSANG'] + drift
        parang = 0
    elif drot_mode == 'STAT':
        pupoff = -100.48
    else:
        raise ValueError('Unknown derotator mode {0}'.format(drot_mode))

    frames_info['PUPIL OFFSET'] = pupoff + instru_offset

    # final derotation value
    frames_info['DEROT ANGLE'] = parang + pupoff + instru_offset + true_north

