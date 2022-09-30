import os
# from abc import ABCMeta, abstractmethod, abstractproperty
from builtins import input, object

import numpy as np
import pkg_resources
from astropy import units as u
from astropy.coordinates import EarthLocation

__all__ = ['CHARIS', 'SPHERE', 'instrument_from_data']


class CHARIS(object):
    """Class containing instrument properties of CHARIS

    """

    __valid_observing_modes = ['J', 'H', 'K',
                               'Broadband', 'ND']

    __wavelength_range = {'J': [1155., 1340.] * u.nanometer,
                          'H': [1470., 1800.] * u.nanometer,
                          'K': [2005., 2380.] * u.nanometer,
                          'Broadband': [1140., 2410.] * u.nanometer,
                          'ND': [1140., 2410.] * u.nanometer}

    __resolution = {'J': 100,
                    'H': 100,
                    'K': 100,
                    'Broadband': 30,
                    'ND': 30}

    def wavelengths(self, lower_wavelength_limit, upper_wavelength_limit, R):
        Nspec = int(np.log(upper_wavelength_limit * 1. / lower_wavelength_limit) * R + 1.5)
        loglam_endpts = np.linspace(np.log(lower_wavelength_limit),
                                    np.log(upper_wavelength_limit), Nspec)
        loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1]) / 2.
        lam_endpts = np.exp(loglam_endpts)
        lam_midpts = np.exp(loglam_midpts)

        return lam_midpts, lam_endpts

    def __init__(self, observing_mode):
        self.instrument_name = 'CHARIS'
        if observing_mode in self.__valid_observing_modes:
            self.observing_mode = observing_mode
        self.wavelength_range = self.__wavelength_range[observing_mode]
        self.resolution = self.__resolution[observing_mode]
        self.lenslet_geometry = 'rectilinear'
        self.pixel_scale = 0.015 * u.arcsec / u.pixel
        self.gain = 2.
        self.wavelengthpolyorder = 3
        self.offsets = np.arange(-5, 6)
        index_range = np.arange(-100, 101, dtype='float')
        self.lenslet_ix, self.lenslet_iy = np.meshgrid(index_range, index_range)

        longitude, latitude = [-155.4760187, 19.825504] * u.degree
        self.location = EarthLocation(longitude, latitude)

        if self.observing_mode == 'ND':
            observing_mode = 'Broadband'
        self.calibration_path = \
            os.path.join(
                pkg_resources.resource_filename('charis', 'calibrations'),
                'CHARIS', observing_mode)

        self.transmission = np.loadtxt(os.path.join(
            self.calibration_path, observing_mode + '_tottrans.dat'))

        self.lam_midpts, self.lam_endpts = \
            self.wavelengths(self.wavelength_range[0].value,
                             self.wavelength_range[1].value,
                             self.resolution)

    def __repr__(self):
        return "{} {}".format(self.instrument_name, self.observing_mode)


class SPHERE(object):
    """Class containing instrument properties of SPHERE

    """

    __valid_observing_modes = ['YJ', 'YH']

    __wavelength_range = {'YJ': [940., 1370.] * u.nanometer,
                          'YH': [920., 1700.] * u.nanometer}

    __resolution = {'YJ': 55,
                    'YH': 35}

    __calibration_wavelength = {
        'YJ': [987.72, 1123.71, 1309.37] * u.nanometer,
        'YH': [987.72, 1123.71, 1309.37, 1545.07] * u.nanometer}

    __wavelengthpolyorder = {
        'YJ': 2,
        'YH': 3}

    # def wavelengths(self, lower_wavelength_limit, upper_wavelength_limit, R):
    #     Nspec = int(np.log(upper_wavelength_limit * 1. / lower_wavelength_limit) * R + 1.5)
    #     loglam_midpts = np.linspace(np.log(lower_wavelength_limit), np.log(upper_wavelength_limit), Nspec)
    #     loglam_binsize = np.diff(loglam_midpts)
    #     loglam_endpts = np.zeros(len(loglam_midpts) + 1)
    #     for i in range(loglam_binsize.shape[0]):
    #         loglam_endpts[i] = loglam_midpts[i] - loglam_binsize[i] / 2.
    #     loglam_endpts[-2] = loglam_midpts[-1] - loglam_binsize[-1] / 2.
    #     loglam_endpts[-1] = loglam_midpts[-1] + loglam_binsize[-1] / 2.
    #
    #     lam_endpts = np.exp(loglam_endpts)
    #     lam_midpts = np.exp(loglam_midpts)
    #
    #     return lam_midpts, lam_endpts

    def wavelengths(self, lower_wavelength_limit, upper_wavelength_limit, R):
        Nspec = int(np.log(upper_wavelength_limit * 1. / lower_wavelength_limit) * R + 1.5)
        loglam_endpts = np.linspace(np.log(lower_wavelength_limit),
                                    np.log(upper_wavelength_limit), Nspec)
        loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1]) / 2.
        lam_endpts = np.exp(loglam_endpts)
        lam_midpts = np.exp(loglam_midpts)

        return lam_midpts, lam_endpts

    # def wavelengths(self, lower_wavelength_limit, upper_wavelength_limit, R):
    #     lam_midpts = np.linspace(957.478, 1635.75, 39)
    #     binsize = np.diff(lam_midpts)[0]
    #     lam_endpts = np.append(lam_midpts - binsize / 2., [lam_midpts[-1] + binsize])
    #
    #     return lam_midpts, lam_endpts

    def __init__(self, observing_mode):
        self.instrument_name = 'SPHERE'
        if observing_mode in self.__valid_observing_modes:
            self.observing_mode = observing_mode
        self.wavelength_range = self.__wavelength_range[observing_mode]
        self.resolution = self.__resolution[observing_mode]
        self.lenslet_geometry = 'hexagonal'
        self.pixel_scale = 0.00746 * u.arcsec / u.pixel
        self.gain = 1.8
        self.calibration_wavelength = self.__calibration_wavelength[observing_mode]
        self.wavelengthpolyorder = self.__wavelengthpolyorder[observing_mode]
        self.offsets = np.arange(-5, 6)
        index_range = np.arange(-100, 101, dtype='float')
        self.lenslet_ix, self.lenslet_iy = np.meshgrid(index_range, index_range)
        self.lenslet_ix[::2] += 0.5
        self.lenslet_iy *= np.sqrt(3) / 2.

        ################################################################
        # Flip the horizontal axis in the resulting cubes to match the
        # orientation of the SPHERE pipeline
        ################################################################

        self.lenslet_ix = self.lenslet_ix[:, ::-1]
        self.lenslet_iy = self.lenslet_iy[:, ::-1]

        longitude, latitude = [-70.4045, -24.6268] * u.degree
        self.location = EarthLocation(longitude, latitude)
        self.calibration_path = \
            os.path.join(
                pkg_resources.resource_filename('charis', 'calibrations'),
                'SPHERE', observing_mode)
        self.transmission = np.loadtxt(os.path.join(
            self.calibration_path, self.observing_mode + '_tottrans.dat'))

        self.lam_midpts, self.lam_endpts = \
            self.wavelengths(self.wavelength_range[0].value,
                             self.wavelength_range[1].value,
                             self.resolution)

    def __repr__(self):
        return "{} {}".format(self.instrument_name, self.observing_mode)


def instrument_from_data(header, calibration=True, interactive=False, verbose=False):
    correct_header = True

    if 'CHARIS' in header['INSTRUME']:
        if 'Y_FLTNAM' in header and 'OBJECT' in header:
            observing_mode = header['Y_FLTNAM']
            instrument = CHARIS(observing_mode)
            if verbose:
                print("Instrument: {}".format(instrument.instrument_name))
                print("Mode: {}".format(instrument.observing_mode))

            if calibration:
                if header['OBJECT'] in ['1200nm', '1550nm', '2346nm']:
                    calibration_wavelength = [int(header['OBJECT'].split('n')[0])] * u.nanometer
                    print(("     Wavelength detected: ", calibration_wavelength))
                else:
                    print("Invalid wavelength keyword")
                    correct_header = False
            else:
                return instrument, None, correct_header
        else:
            correct_header = False

        if not correct_header and interactive:
            print("\n" + "*" * 60)
            print("The file you selected doesn't appear to have the correct header keywords set")
            print("This can happen for files taken before Apr 1st, 2017. Please enter them manually.")
            print("*" * 60)
            while True:
                observing_mode = input("     Band? [J/H/K/Broadband/ND]: ")
                if observing_mode in ["J", "H", "K", "Broadband", "ND"]:
                    break
                else:
                    print("Invalid input.")
            while True:
                calibration_wavelength = input("     Wavelength? [1200/1550/2346]: ")
                if calibration_wavelength in ["1200", "1550", "2346"]:
                    calibration_wavelength = [int(calibration_wavelength)] * u.nanometer
                    break
                else:
                    print("Invalid input")
            instrument = CHARIS(observing_mode)

    elif 'SPHERE' in header['INSTRUME']:
        if 'IFS' in header['HIERARCH ESO SEQ ARM']:
            if 'YJ' in header['HIERARCH ESO INS2 COMB IFS']:
                observing_mode = 'YJ'
            else:
                observing_mode = 'YH'
        else:
            raise ValueError("Data is not for IFS")
        instrument = SPHERE(observing_mode)
        calibration_wavelength = instrument.calibration_wavelength

    else:
        raise NotImplementedError("The instrument is not supported.")

    return instrument, calibration_wavelength, correct_header
