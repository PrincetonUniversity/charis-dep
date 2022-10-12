""" Partially adapted from A. Vigan's VLTPF Pipeline
Commit: f20dbcc on Feb 6 """

from builtins import range, zip

import astropy.coordinates as coord
from astropy import units as u
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling import fitting, models
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


def parallatic_angle(ha, dec, geolat):
    '''
    Parallactic angle of a source in degrees

    Parameters
    ----------
    ha : array_like
        Hour angle, in hours

    dec : float
        Declination, in degrees

    geolat : float
        Observatory declination, in degrees

    Returns
    -------
    pa : array_like
        Parallactic angle values
    '''
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
    time_delta = (time_end - time_start) / frames_info['DET NDIT'].values.astype(np.int)
    DIT = np.array(frames_info['DET SEQ1 DIT'].values.astype(
        np.float) * 1000, dtype='timedelta64[ms]')

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

    date_fix = Time('2016-07-12')
    if np.any(frames_info['MJD'].values <= date_fix.mjd):
        try:
            alt = frames_info['TEL ALT'].values.astype(np.float)
            drot2 = frames_info['INS4 DROT2 BEGIN'].values.astype(np.float)
            pa_correction = np.degrees(np.arctan(np.tan(np.radians(alt-2.*drot2))))
        except KeyError:
            pa_correction = 0
    else:
        pa_correction = 0

    # RA/DEC
    ra_drot = frames_info['INS4 DROT2 RA'].values.astype(np.float)
    ra_drot_h = np.floor(ra_drot / 1e4)
    ra_drot_m = np.floor((ra_drot - ra_drot_h * 1e4)/1e2)
    ra_drot_s = ra_drot - ra_drot_h*1e4 - ra_drot_m*1e2
    ra_hour = coord.Angle((ra_drot_h, ra_drot_m, ra_drot_s), u.hour)
    ra_deg = ra_hour*15
    frames_info['RA'] = ra_deg.value

    dec_drot = frames_info['INS4 DROT2 DEC'].values.astype(np.float)
    sign = np.sign(dec_drot)
    udec_drot = np.abs(dec_drot)
    dec_drot_d = np.floor(udec_drot / 1e4)
    dec_drot_m = np.floor((udec_drot - dec_drot_d * 1e4) / 1e2)
    dec_drot_s = udec_drot - dec_drot_d * 1e4 - dec_drot_m * 1e2
    dec_drot_d *= sign
    dec = coord.Angle((dec_drot_d, dec_drot_m, dec_drot_s), u.degree)
    frames_info['DEC'] = dec.value

    geolon = coord.Angle(frames_info['TEL GEOLON'].values[0], u.degree)
    geolat = coord.Angle(frames_info['TEL GEOLAT'].values[0], u.degree)
    geoelev = frames_info['TEL GEOELEV'].values[0]

    location = (geolon, geolat, geoelev)
    # calculate parallactic angles
    utc = Time(frames_info['TIME'].values.astype(str), scale='utc', location=location)
    lst = utc.sidereal_time('apparent')
    ha = lst - ra_hour
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
    ha = lst - ra_hour
    pa = parallatic_angle(ha, dec[0], geolat)
    frames_info['PARANG START'] = pa.value + pa_correction
    frames_info['HOUR ANGLE START'] = ha.value
    frames_info['LST START'] = lst.value

    utc = Time(frames_info['TIME END'].values.astype(str), scale='utc', location=location)
    lst = utc.sidereal_time('apparent')
    ha = lst - ra_hour
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

    #
    # Derotation angles
    #
    # PA_on-sky = PA_detector + PARANGLE + True_North + PUP_OFFSET + INSTRUMENT_OFFSET + TRUE_NORTH
    #  PUP_OFFSET = -135.99 +/- 0.11
    #  INSTRUMENT_OFFSET
    #   IFS = +100.48 +/- 0.10
    #   IRD =    0.00 +/- 0.00
    #   TRUE_NORTH = -1.75 +/- 0.08
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

    drot_mode = frames_info['INS4 DROT2 MODE'].unique()
    if len(drot_mode) != 1:
        raise ValueError('Derotator mode has several values in the sequence')
    if drot_mode == 'ELEV':
        pupoff = 135.99
    elif drot_mode == 'SKY':
        pupoff = -100.48 + frames_info['INS4 DROT2 POSANG']
    elif drot_mode == 'STAT':
        pupoff = -100.48
    else:
        raise ValueError('Unknown derotator mode {0}'.format(drot_mode))

    frames_info['PUPIL OFFSET'] = pupoff + instru_offset

    # final derotation value
    frames_info['DEROT ANGLE'] = frames_info['PARANG'] + pupoff + instru_offset + true_north


def compute_bad_pixel_map(bpm_files, dtype=np.uint8):
    '''
    Compute a combined bad pixel map provided a list of files

    Parameters
    ----------
    bpm_files : list
        List of names for the bpm files

    dtype : data type
        Data type for the final bpm

    Returns
    bpm : array_like
        Combined bad pixel map
    '''

    # check that we have files
    if len(bpm_files) == 0:
        raise ValueError('No bad pixel map files provided')

    # get shape
    shape = fits.getdata(bpm_files[0]).shape

    # star with empty bpm
    bpm = np.zeros((shape[-2], shape[-1]), dtype=np.uint8)

    # fill if files are provided
    for f in bpm_files:
        data = fits.getdata(f)
        bpm = np.logical_or(bpm, data)

    bpm = bpm.astype(dtype)

    return bpm


def collapse_frames_info(finfo, fname, collapse_type, coadd_value=2):
    '''
    Collapse frame info to match the collapse operated on the data

    Parameters
    ----------
    finfo : dataframe
        The data frame with all the information on science frames

    fname : str
       The name of the current file

    collapse_type : str
        Type of collapse. Possible values are mean or coadd. Default
        is mean.

    coadd_value : int
        Number of consecutive frames to be coadded when collapse_type
        is coadd. Default is 2

    Returns
    -------
    nfinfo : dataframe
        Collapsed data frame
    '''

    print('   ==> collapse frames information')

    nfinfo = None
    if collapse_type == 'none':
        nfinfo = finfo
    elif collapse_type == 'mean':
        index = pd.MultiIndex.from_arrays([[fname], [0]], names=['FILE', 'IMG'])
        nfinfo = pd.DataFrame(columns=finfo.columns, index=index)

        # get min/max indices
        imin = finfo.index.get_level_values(1).min()
        imax = finfo.index.get_level_values(1).max()

        # copy data
        nfinfo.loc[(fname, 0)] = finfo.loc[(fname, imin)]

        # update time values
        nfinfo.loc[(fname, 0), 'DET NDIT'] = 1
        nfinfo.loc[(fname, 0), 'TIME START'] = finfo.loc[(fname, imin), 'TIME START']
        nfinfo.loc[(fname, 0), 'TIME END'] = finfo.loc[(fname, imax), 'TIME END']
        nfinfo.loc[(fname, 0), 'TIME'] = finfo.loc[(fname, imin), 'TIME START'] + \
            (finfo.loc[(fname, imax), 'TIME END'] - finfo.loc[(fname, imin), 'TIME START']) / 2

        # recompute angles
        compute_angles(nfinfo)
    elif collapse_type == 'coadd':
        coadd_value = int(coadd_value)
        NDIT = len(finfo)
        NDIT_new = NDIT // coadd_value

        index = pd.MultiIndex.from_arrays(
            [np.full(NDIT_new, fname), np.arange(NDIT_new)], names=['FILE', 'IMG'])
        nfinfo = pd.DataFrame(columns=finfo.columns, index=index)

        for f in range(NDIT_new):
            # get min/max indices
            imin = int(f * coadd_value)
            imax = int((f + 1) * coadd_value - 1)

            # copy data
            nfinfo.loc[(fname, f)] = finfo.loc[(fname, imin)]

            # update time values
            nfinfo.loc[(fname, f), 'DET NDIT'] = 1
            nfinfo.loc[(fname, f), 'TIME START'] = finfo.loc[(fname, imin), 'TIME START']
            nfinfo.loc[(fname, f), 'TIME END'] = finfo.loc[(fname, imax), 'TIME END']
            nfinfo.loc[(fname, f), 'TIME'] = finfo.loc[(fname, imin), 'TIME START'] + \
                (finfo.loc[(fname, imax), 'TIME END'] - finfo.loc[(fname, imin), 'TIME START']) / 2

        # recompute angles
        compute_angles(nfinfo)
    else:
        raise ValueError('Unknown collapse type {0}'.format(collapse_type))

    return nfinfo


def lines_intersect(a1, a2, b1, b2):
    '''
    Determines the intersection point of two lines passing by points
    (a1,a2) and (b1,b2).

    See https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    Parameters
    ----------

    a, b : 2D tuples
        Coordinates of points on line 1

    c, d : 2D tuples
        Coordinates of points on line 2

    Returns
    -------
    val
        Returns None is lines are parallel, (cx,cy) otherwise.
    '''

    # make sure we have arrays
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    # test lines
    da = a2 - a1                # vector from A1 to A2
    db = b2 - b1                # vector from B1 to B2
    dp = a1 - b1
    pda = [-da[1], da[0]]       # perpendicular to A1-A2 vector

    # parallel lines
    if (pda * db).sum() == 0:
        return None

    # find intersection
    denom = np.dot(pda, db)
    num = np.dot(pda, dp)

    return (num / denom) * db + b1


def star_centers_from_PSF_img_cube(cube, wave, pixel, exclude_fraction=0.1, box_size=60,
                                   save_path=None):
    '''
    Compute star center from PSF images (IRDIS CI, IRDIS DBI, IFS)

    Parameters
    ----------
    cube : array_like
        IRDIFS PSF cube

    wave : array_like
        Wavelength values, in nanometers

    pixel : float
        Pixel scale, in mas/pixel

    exclude_fraction : float
        Exclude a fraction of the image borders to avoid getting
        biased by hot pixels close to the edges. Default is 10%

    box_size : int
        Size of the box in which the fit is performed. Default is 60 pixels

    save_path : str
        Path where to save the fit images. Default is None, which means
        that the plot is not produced


    Returns
    -------
    img_centers : array_like
        The star center in each frame of the cube

    '''

    # standard parameters
    nwave = wave.size
    loD = wave*1e-9 / 8 * 180/np.pi * 3600 * 1000 / pixel
    box = box_size // 2

    # spot fitting
    xx, yy = np.meshgrid(np.arange(2 * box), np.arange(2 * box))

    # loop over images
    img_centers = np.zeros((nwave, 2))
    # failed_centers = np.zeros(nwave, dtype=np.bool)
    for idx, (cwave, img) in enumerate(zip(wave, cube)):
        print('   ==> wave {0:2d}/{1:2d} ({2:4.0f} nm)'.format(idx + 1, nwave, cwave))

        # remove any NaN
        img = np.nan_to_num(img)

        # center guess
        cy, cx = np.unravel_index(np.argmax(img), img.shape)

        # check if we are really too close to the edge
        dim = img.shape
        lf = exclude_fraction
        hf = 1 - exclude_fraction
        if (cx <= lf*dim[-1]) or (cx >= hf*dim[-1]) or \
           (cy <= lf*dim[0]) or (cy >= hf*dim[0]):
            nimg = img.copy()
            nimg[:, :int(lf*dim[-1])] = 0
            nimg[:, int(hf*dim[-1]):] = 0
            nimg[:int(lf*dim[0]), :] = 0
            nimg[int(hf*dim[0]):, :] = 0

            cy, cx = np.unravel_index(np.argmax(nimg), img.shape)

        # sub-image
        sub = img[cy-box:cy+box, cx-box:cx+box]

        # fit peak with Gaussian + constant
        imax = np.unravel_index(np.argmax(sub), sub.shape)
        g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
                                   x_stddev=loD[idx], y_stddev=loD[idx]) + \
            models.Const2D(amplitude=sub.min())
        fitter = fitting.LevMarLSQFitter()
        par = fitter(g_init, xx, yy, sub)

        cx_final = cx - box + par[0].x_mean
        cy_final = cy - box + par[0].y_mean

        img_centers[idx, 0] = cx_final
        img_centers[idx, 1] = cy_final

    # look for outliers and replace by a linear fit to all good ones
    # Ticket #81
    ibad = []
    if nwave > 2:
        c_med = np.median(img_centers, axis=0)
        c_std = np.std(img_centers, axis=0)
        bad = np.any(np.logical_or(img_centers < (c_med-3*c_std),
                                   img_centers > (c_med+3*c_std)), axis=1)
        ibad = np.where(bad)[0]
        igood = np.where(np.logical_not(bad))[0]
        nbad = len(ibad)

        if nbad != 0:
            print('   ==> found {} outliers. Will replace them with a linear fit.'.format(nbad))

            idx = np.arange(nwave)

            # x
            lin = np.polyfit(idx[igood], img_centers[igood, 0], 1)
            pol = np.poly1d(lin)
            img_centers[ibad, 0] = pol(idx[ibad])

            # y
            lin = np.polyfit(idx[igood], img_centers[igood, 1], 1)
            pol = np.poly1d(lin)
            img_centers[ibad, 1] = pol(idx[ibad])

    #
    # Generate summary plot
    #

    # multi-page PDF to save result
    if save_path is not None:
        pdf = PdfPages(save_path)

        for idx, (cwave, img) in enumerate(zip(wave, cube)):
            cx_final = img_centers[idx, 0]
            cy_final = img_centers[idx, 1]

            failed = (idx in ibad)
            if failed:
                mcolor = 'r'
                bcolor = 'r'
            else:
                mcolor = 'b'
                bcolor = 'w'

            plt.figure('PSF center - imaging', figsize=(8.3, 8))
            plt.clf()

            plt.subplot(111)
            plt.imshow(img/np.nanmax(img), aspect='equal', norm=colors.LogNorm(vmin=1e-6, vmax=1),
                       interpolation='nearest', cmap=global_cmap)
            plt.plot([cx_final], [cy_final], marker='D', color=mcolor)
            plt.gca().add_patch(patches.Rectangle((cx-box, cy-box), 2*box, 2*box, ec=bcolor, fc='none'))
            if failed:
                plt.text(cx, cy+box, 'Fit failed', color='r', weight='bold', fontsize='x-small',
                         ha='center', va='bottom')
            plt.title(r'Image #{0} - {1:.0f} nm'.format(idx+1, cwave))

            ext = 1000 / pixel
            plt.xlim(cx_final-ext, cx_final+ext)
            plt.xlabel('x position [pix]')
            plt.ylim(cy_final-ext, cy_final+ext)
            plt.ylabel('y position [pix]')

            plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.95)

            pdf.savefig()

        pdf.close()

    return img_centers


def star_centers_from_waffle_cube(cube, wave, instrument, waffle_orientation,
                                  high_pass=False, center_offset=(0, 0), smooth=0,
                                  coro=True, display=False, save_path=None):
    '''
    Compute star center from waffle images
    Parameters
    ----------
    cube : array_like
        Waffle IRDIS cube
    wave : array_like
        Wavelength values, in nanometers
    instrument : str
        Instrument, IFS or IRDIS

    waffle_orientation : str
        String giving the waffle orientation '+' or 'x'
    high_pass : bool
        Apply high-pass filter to the image before searching for the
        satelitte spots
    smooth : int
        Apply a gaussian smoothing to the images to reduce noise. The
        value is the sigma of the gaussian in pixel.  Default is no
        smoothing

    center_offset : tuple
        Apply an (x,y) offset to the default center position. Default is no offset

    coro : bool
        Observation was performed with a coronagraph. Default is True
    display : bool
        Display the fit of the satelitte spots
    save_path : str
        Path where to save the fit images

    Returns
    -------
    spot_center : array_like
        Centers of each individual spot in each frame of the cube
    spot_dist : array_like
        The 6 possible distances between the different spots
    img_center : array_like
        The star center in each frame of the cube
    '''

    # instrument
    if instrument == 'IFS':
        pixel = 7.46
        offset = 102
    elif instrument == 'IRDIS':
        pixel = 12.25
        offset = 0
    else:
        raise ValueError('Unknown instrument {0}'.format(instrument))

    # standard parameters
    dim = cube.shape[-1]
    nwave = wave.size
    loD = wave * 1e-6 / 8 * 180 / np.pi * 3600 * 1000 / pixel

    # waffle parameters
    freq = 10 * np.sqrt(2) * 0.97
    box = 8
    if waffle_orientation == '+':
        orient = offset * np.pi / 180
    elif waffle_orientation == 'x':
        orient = offset * np.pi / 180 + np.pi / 4

    # spot fitting
    xx, yy = np.meshgrid(np.arange(2 * box), np.arange(2 * box))

    # multi-page PDF to save result
    if save_path is not None:
        pdf = PdfPages(save_path)

    # center guess
    if instrument == 'IFS':
        center_guess = np.full((nwave, 2), ((dim // 2) + 3, (dim // 2) - 1))
    elif instrument == 'IRDIS':
        center_guess = np.array(((485, 520), (486, 508)))

    # loop over images
    spot_center = np.zeros((nwave, 4, 2))
    spot_dist = np.zeros((nwave, 6))
    img_center = np.zeros((nwave, 2))
    for idx, (wave, img) in enumerate(zip(wave, cube)):
        print('  wave {0:2d}/{1:2d} ({2:.3f} micron)'.format(idx + 1, nwave, wave))

        # remove any NaN
        img = np.nan_to_num(img)

        # center guess (+offset)
        cx_int = int(center_guess[idx, 0]) + center_offset[0]
        cy_int = int(center_guess[idx, 1]) + center_offset[1]

        # optional high-pass filter
        if high_pass:
            img = img - ndimage.median_filter(img, 15, mode='mirror')

        # optional smoothing
        if smooth > 0:
            img = ndimage.gaussian_filter(img, smooth)

        # mask for non-coronagraphic observations
        if not coro:
            mask = aperture.disc(cube[0].shape[-1], 5 * loD[idx], diameter=False,
                                 center=(cx_int, cy_int), invert=True)
            img *= mask

        # create plot if needed
        if save_path or display:
            fig = plt.figure(0, figsize=(8, 8))
            plt.clf()
            col = ['red', 'blue', 'magenta', 'purple']
            ax = fig.add_subplot(111)
            ax.imshow(img / img.max(), aspect='equal', vmin=1e-2, vmax=1, norm=colors.LogNorm())
            ax.set_title(r'Image #{0} - {1:.3f} $\mu$m'.format(idx + 1, wave))

        # satelitte spots
        for s in range(4):
            cx = int(cx_int + freq * loD[idx] * np.cos(orient + np.pi / 2 * s))
            cy = int(cy_int + freq * loD[idx] * np.sin(orient + np.pi / 2 * s))

            sub = img[cy - box:cy + box, cx - box:cx + box]

            # fit: Gaussian + constant
            imax = np.unravel_index(np.argmax(sub), sub.shape)
            g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
                                       x_stddev=loD[idx], y_stddev=loD[idx]) + \
                models.Const2D(amplitude=sub.min())
            fitter = fitting.LevMarLSQFitter()
            par = fitter(g_init, xx, yy, sub)
            fit = par(xx, yy)

            cx_final = cx - box + par[0].x_mean
            cy_final = cy - box + par[0].y_mean

            spot_center[idx, s, 0] = cx_final
            spot_center[idx, s, 1] = cy_final

            # plot sattelite spots and fit
            if save_path or display:
                ax.plot([cx_final], [cy_final], marker='D', color=col[s])
                ax.add_patch(patches.Rectangle((cx - box, cy - box),
                             2 * box, 2 * box, ec='white', fc='none'))

                axs = fig.add_axes((0.17 + s * 0.2, 0.17, 0.1, 0.1))
                axs.imshow(sub, aspect='equal', vmin=0, vmax=sub.max())
                axs.plot([par[0].x_mean], [par[0].y_mean], marker='D', color=col[s])
                axs.set_xticks([])
                axs.set_yticks([])

                axs = fig.add_axes((0.17 + s * 0.2, 0.06, 0.1, 0.1))
                axs.imshow(fit, aspect='equal', vmin=0, vmax=sub.max())
                axs.set_xticks([])
                axs.set_yticks([])

        # lines intersection
        intersect = lines_intersect(spot_center[idx, 0, :], spot_center[idx, 2, :],
                                    spot_center[idx, 1, :], spot_center[idx, 3, :])
        img_center[idx] = intersect

        # scaling
        spot_dist[idx, 0] = np.sqrt(np.sum((spot_center[idx, 0, :] - spot_center[idx, 2, :])**2))
        spot_dist[idx, 1] = np.sqrt(np.sum((spot_center[idx, 1, :] - spot_center[idx, 3, :])**2))
        spot_dist[idx, 2] = np.sqrt(np.sum((spot_center[idx, 0, :] - spot_center[idx, 1, :])**2))
        spot_dist[idx, 3] = np.sqrt(np.sum((spot_center[idx, 0, :] - spot_center[idx, 3, :])**2))
        spot_dist[idx, 4] = np.sqrt(np.sum((spot_center[idx, 1, :] - spot_center[idx, 2, :])**2))
        spot_dist[idx, 5] = np.sqrt(np.sum((spot_center[idx, 2, :] - spot_center[idx, 3, :])**2))

        # finalize plot
        if save_path or display:
            ax.plot([spot_center[idx, 0, 0], spot_center[idx, 2, 0]],
                    [spot_center[idx, 0, 1], spot_center[idx, 2, 1]],
                    color='w', linestyle='dashed')
            ax.plot([spot_center[idx, 1, 0], spot_center[idx, 3, 0]],
                    [spot_center[idx, 1, 1], spot_center[idx, 3, 1]],
                    color='w', linestyle='dashed')

            ax.plot([intersect[0]], [intersect[1]], marker='+', color='w', ms=15)

            ext = 1000 / pixel
            ax.set_xlim(intersect[0] - ext, intersect[0] + ext)
            ax.set_ylim(intersect[1] - ext, intersect[1] + ext)

            plt.tight_layout()

            if save_path:
                pdf.savefig()

            if display:
                plt.pause(1e-3)

    if save_path:
        pdf.close()

    return spot_center, spot_dist, img_center
