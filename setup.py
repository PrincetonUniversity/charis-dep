#!/usr/bin/env python

import glob

from setuptools import setup, find_packages, Extension
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
        description='The Data Reduction Pipeline for the CHARIS and SPHERE Integral-Field Spectrographs',
        author='Timothy Brandt, Maxime Rizzo, Matthias Samland',
        author_email='timothy.d.brandt@gmail.com',
        packages=find_packages(),
        package_data={"": ["calibrations/**/*.fits",
                           "calibrations/**/*.json",
                           "calibrations/**/**/*.fits",
                           "calibrations/**/**/*.dat"]},
        python_requires='>=3.7',  # , <3.10.0',
        install_requires=['numpy', 'scipy', 'astropy',
                          'pandas', 'tqdm', 'future',
                          'bokeh', 'bottleneck', 'psutil'],
        scripts=['scripts/buildcal', 'scripts/extractcube', 'scripts/hexplot'],
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )


try:
    setup_charis(ext_modules_openMP)
except Exception:
    setup_charis(ext_modules_noopenMP)
