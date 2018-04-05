# from abc import ABCMeta, abstractmethod, abstractproperty
import os

import numpy as np
import pkg_resources
from astropy import units as u
from astropy.coordinates import EarthLocation

__all__ = ['CHARIS', 'SPHERE', 'instrument_from_data']


class CHARIS(object):
    """Class containing instrument properties of CHARIS

    """

    __valid_observing_modes = ['J', 'H', 'K',
                               'Broadband']

    __wavelength_range = {'J': [1155., 1340.] * u.nanometer,
                          'H': [1470., 1800.] * u.nanometer,
                          'K': [2005., 2380.] * u.nanometer,
                          'Broadband': [1140., 2410.] * u.nanometer}

    __resolution = {'J': 100,
                    'H': 100,
                    'K': 100,
                    'Broadband': 30}

    def __init__(self, observing_mode):
        self.instrument_name = 'CHARIS'
        if observing_mode in self.__valid_observing_modes:
            self.observing_mode = observing_mode
        self.wavelength_range = self.__wavelength_range[observing_mode]
        self.resolution = self.__resolution[observing_mode]
        self.lenslet_geometry = 'rectilinear'
        self.pixel_scale = 0.015 * u.arcsec / u.pixel
        self.gain = 2.
        index_range = np.arange(-100, 101, dtype='float')
        self.lenslet_ix, self.lenslet_iy = np.meshgrid(index_range, index_range)

        longitude, latitude = [-155.4760187, 19.825504] * u.degree
        self.location = EarthLocation(longitude, latitude)
        self.calibration_path = \
            os.path.join(
                pkg_resources.resource_filename('charis', 'calibrations'),
                'CHARIS', observing_mode)
        self.transmission = np.loadtxt(os.path.join(
            self.calibration_path, self.observing_mode + '_tottrans.dat'))

    def __repr__(self):
        return "{} {}".format(self.instrument_name, self.observing_mode)


class SPHERE(object):
    """Class containing instrument properties of SPHERE

    """

    __valid_observing_modes = ['YJ', 'YH']

    __wavelength_range = {'YJ': [900., 1350.] * u.nanometer,
                          'YH': [900., 1650.] * u.nanometer}

    __resolution = {'YJ': 55,
                    'YH': 35}

    __calibration_wavelength = {'YJ': [987.7, 1123.7, 1309.4] * u.nanometer,
                                'YH': [987.7, 1123.7, 1309.4, 1545.1] * u.nanometer}

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

        index_range = np.arange(-100, 101, dtype='float')
        self.lenslet_ix, self.lenslet_iy = np.meshgrid(index_range, index_range)
        self.lenslet_ix[::2] += 0.5
        self.lenslet_iy *= np.sqrt(3) / 2.

        longitude, latitude = [-70.4045, -24.6268] * u.degree
        self.location = EarthLocation(longitude, latitude)
        self.calibration_path = \
            os.path.join(
                pkg_resources.resource_filename('charis', 'calibrations'),
                'SPHERE', observing_mode)
        self.transmission = np.loadtxt(os.path.join(
            self.calibration_path, self.observing_mode + '_tottrans.dat'))

    def __repr__(self):
        return "{} {}".format(self.instrument_name, self.observing_mode)


def instrument_from_data(header, interactive=False):
    correct_header = True

    if 'CHARIS' in header['INSTRUME']:
        if 'Y_FLTNAM' in header and 'OBJECT' in header:
            observing_mode = header['Y_FLTNAM']
            instrument = CHARIS(observing_mode)
            print("Instrument: {}".format(instrument.instrument_name))
            print("Mode: {}".format(instrument.observing_mode))

            if header['OBJECT'] in ['1200nm', '1550nm', '2346nm']:
                calibration_wavelength = [int(header['OBJECT'].split('n')[0])] * u.nanometer
                print("     Wavelength detected: ", calibration_wavelength)
            else:
                print("Invalid wavelength keyword")
                correct_header = False
        else:
            correct_header = False

        if not correct_header and interactive:
            print("\n" + "*" * 60)
            print("The file you selected doesn't appear to have the correct header keywords set")
            print("This can happen for files taken before Apr 1st, 2017. Please enter them manually.")
            print("*" * 60)
            while True:
                observing_mode = raw_input("     Band? [J/H/K/Broadband]: ")
                if observing_mode in ["J", "H", "K", "Broadband"]:
                    break
                else:
                    print("Invalid input.")
            while True:
                calibration_wavelength = raw_input("     Wavelength? [1200/1550/2346]: ")
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
