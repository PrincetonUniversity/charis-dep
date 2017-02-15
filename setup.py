#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


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
    packages = {'charis', 'charis.primitives', 'charis.utr', 'charis.image'},
    package_dir = {'charis': 'code', 'charis.primitives':'code/primitives',
                   'charis.image':'code/image', 'charis.utr':'code/utr'},
    install_requires = ['numpy', 'scipy', 'astropy'],
    scripts=['scripts/buildcal', 'scripts/extractcube'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
) 
