#!/usr/bin/env python

import glob

from setuptools import setup, Extension
from Cython.Distutils import build_ext

ext_modules_openMP = []
ext_modules_openMP += [Extension("charis.primitives.matutils",
                                 ['charis/primitives/matutils.pyx'],
                                 extra_compile_args=['-fopenmp'],
                                 extra_link_args=['-fopenmp'],
                                 )]

ext_modules_openMP += [Extension("charis.utr.fitramp",
                                 ['charis/utr/fitramp.pyx'],
                                 extra_compile_args=['-fopenmp'],
                                 extra_link_args=['-fopenmp'],
                                 )]

ext_modules_noopenMP = []
ext_modules_noopenMP += [Extension("charis.primitives.matutils",
                                   ['charis/primitives/matutils.pyx'],
                                   )]

ext_modules_noopenMP += [Extension("charis.utr.fitramp",
                                   ['charis/utr/fitramp.pyx'],
                                   )]


def setup_charis(ext_modules):
    setup(
        name='charis',
        version='1.0.1',
        description='The Data Reduction Pipeline for the CHARIS Integral-Field Spectrograph',
        author='Timothy Brandt, Maxime Rizzo',
        author_email='timothy.d.brandt@gmail.com',
        packages={'charis', 'charis.primitives', 'charis.utr', 'charis.image',
                    'charis.parallel'},
        package_dir={'charis': 'charis', 'charis.primitives': 'charis/primitives',
                     'charis.image': 'charis/image', 'charis.utr': 'charis/utr',
                     'charis.parallel': 'charis/parallel'},
        data_files=[('charis/calibrations', ['charis/calibrations/lowres/mask.fits',
                                             'charis/calibrations/lowres/pixelflat.fits']),
                    ('charis/calibrations/lowres', glob.glob('charis/calibrations/lowres/hires_psflets*') +
                     ['charis/calibrations/lowres/lensletflat.fits',
                        'charis/calibrations/lowres/lowres_tottrans.dat',
                        'charis/calibrations/lowres/lamsol.dat']),
                    ('charis/calibrations/highres_J', glob.glob('charis/calibrations/highres_J/hires_psflets*') +
                     ['charis/calibrations/highres_J/lensletflat.fits',
                        'charis/calibrations/highres_J/J_tottrans.dat',
                        'charis/calibrations/highres_J/lamsol.dat']),
                    ('charis/calibrations/highres_H', glob.glob('charis/calibrations/highres_H/hires_psflets*') +
                     ['charis/calibrations/highres_H/lensletflat.fits',
                        'charis/calibrations/highres_H/H_tottrans.dat',
                        'charis/calibrations/highres_H/lamsol.dat']),
                    ('charis/calibrations/highres_K', glob.glob('charis/calibrations/highres_K/hires_psflets*') +
                     ['charis/calibrations/highres_K/lensletflat.fits',
                        'charis/calibrations/highres_K/K_tottrans.dat',
                        'charis/calibrations/highres_K/lamsol.dat'])],
        install_requires=['numpy', 'scipy', 'astropy', 'cython>=0.x'],
        scripts=['scripts/buildcal', 'scripts/extractcube'],
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )


try:
    setup_charis(ext_modules_openMP)
except:
    setup_charis(ext_modules_noopenMP)
