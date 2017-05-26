#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import glob


ext_modules = [ ]
ext_modules += [Extension("charis.primitives.matutils", 
                         ['code/primitives/matutils.pyx'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                      )]

ext_modules += [Extension("charis.utr.fitramp", 
                          ['code/utr/fitramp.pyx'],
                          extra_compile_args=['-fopenmp'],
                          extra_link_args=['-fopenmp'],
                      )]


setup(    
    name='charis', 
    version='1.0.0',
    description='The Data Reduction Pipeline for the CHARIS Integral-Field Spectrograph',
    author='Timothy Brandt, Maxime Rizzo',
    author_email='timothy.d.brandt@gmail.com',
    packages = {'charis', 'charis.primitives', 'charis.utr', 'charis.image',
                'charis.parallel'},
    package_dir = {'charis':'code', 'charis.primitives':'code/primitives',
                   'charis.image':'code/image', 'charis.utr':'code/utr',
                   'charis.parallel':'code/parallel'},
    data_files = [('charis/calibrations', ['code/calibrations/lowres/mask.fits',
                                           'code/calibrations/lowres/pixelflat.fits']),
                  ('charis/calibrations/lowres', glob.glob('code/calibrations/lowres/hires_psflets*') + 
                                                  ['code/calibrations/lowres/lensletflat.fits',
                                                  'code/calibrations/lowres/lowres_tottrans.dat',
                                                  'code/calibrations/lowres/lamsol.dat']),
                  ('charis/calibrations/highres_J', glob.glob('code/calibrations/highres_J/hires_psflets*') + 
                                                  ['code/calibrations/highres_J/lensletflat.fits',
                                                  'code/calibrations/highres_J/J_tottrans.dat',
                                                  'code/calibrations/highres_J/lamsol.dat']),
                  ('charis/calibrations/highres_H', glob.glob('code/calibrations/highres_H/hires_psflets*') + 
                                                  ['code/calibrations/highres_H/lensletflat.fits',
                                                  'code/calibrations/highres_H/H_tottrans.dat',
                                                  'code/calibrations/highres_H/lamsol.dat']),
                  ('charis/calibrations/highres_K', glob.glob('code/calibrations/highres_K/hires_psflets*') + 
                                                  ['code/calibrations/highres_K/lensletflat.fits',
                                                  'code/calibrations/highres_K/K_tottrans.dat',
                                                  'code/calibrations/highres_K/lamsol.dat'])],
    install_requires = ['numpy', 'scipy', 'astropy'],
    scripts=['scripts/buildcal', 'scripts/extractcube'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
) 
