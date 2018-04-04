# from abc import ABCMeta, abstractmethod, abstractproperty
import os
import pkg_resources
from astropy import units as u
from astropy.coordinates import EarthLocation


__all__ = ['CHARIS', 'SPHERE']


class CHARIS(object):
    """Class containing instrument properties of Charis

    """

    __valid_observing_modes = ['highres_J', 'highres_H', 'highres_K',
                               'lowres']

    __wavelength_range = {'highres_J': [1155., 1340.] * u.nanometer,
                          'highres_H': [1470., 1800.] * u.nanometer,
                          'highres_K': [2005., 2380.] * u.nanometer,
                          'lowres': [1140., 2410.] * u.nanometer}

    __resolution = {'highres_J': 100,
                    'highres_H': 100,
                    'highres_K': 100,
                    'lowres': 30}

    def __init__(self, observing_mode):
        self.instrument_name = 'CHARIS'
        if observing_mode in self.__valid_observing_modes:
            self.observing_mode = observing_mode
        self.wavelength_range = self.__wavelength_range[observing_mode]
        self.resolution = self.__resolution[observing_mode]
        self.lenslet_geometry = 'rectilinear'

        longitude, latitude = [-155.4760187, 19.825504] * u.degree
        self.location = EarthLocation(longitude, latitude)
        self.calibration_path = \
            os.path.join(
                pkg_resources.resource_filename('charis', 'calibrations'),
                observing_mode)

    def __repr__(self):
        return "{} {}".format(self.instrument_name, self.observing_mode)


class SPHERE(object):
    """Class containing instrument properties of Sphere

    """

    __valid_observing_modes = ['SPHERE_YJ', 'SPHERE_YH']

    __wavelength_range = {'SPHERE_YJ': [950., 1350.] * u.nanometer,
                          'SPHERE_YH': [950., 1650.] * u.nanometer}

    __resolution = {'SPHERE_YJ': 54,
                    'SPHERE_YH': 33}

    def __init__(self, observing_mode):
        self.instrument_name = 'SPHERE'
        if observing_mode in self.__valid_observing_modes:
            self.observing_mode = observing_mode
        self.wavelength_range = self.__wavelength_range[observing_mode]
        self.resolution = self.__resolution[observing_mode]
        self.lenslet_geometry = 'hexagonal'

        longitude, latitude = [-70.4045, -24.6268] * u.degree
        self.location = EarthLocation(longitude, latitude)
        self.calibration_path = \
            os.path.join(
                pkg_resources.resource_filename('charis', 'calibrations'),
                observing_mode)

    def __repr__(self):
        return "{} {}".format(self.instrument_name, self.observing_mode)
