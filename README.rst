charis
====
Data Reduction Pipeline for the CHARIS and SPHERE Integral-Field Spectrographs
-------------------------------------------------------------------------------------------

The charis pipeline is capable of extracting spectral data cubes from both the Subaru/CHARIS as well as the SPHERE/IFS integral field spectrographs for high-contrast imaging.

For a detailed description of the pipeline please refer to `Brandt et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract>`_. For a description of the pipeline's adaption to SPHERE/IFS please refer to Samland et al. 2022.


Requirements
------------
Python >3.7
Cython with a C compiler and OpenMP


Dependencies
------------
The charis pipeline requires the following packages in a reasonably up-to-date version
to function:

- numpy, scipy, astropy, pandas, tqdm, matplotlib, cython, bokeh
- bottleneck, psutil


Contributing
------------

Please open a new issue or new pull request for bugs, feedback, or new features you would like to see.   If there is an issue you would like to work on, please leave a comment and we will be happy to assist.   New contributions and contributors are very welcome!

New to github or open source projects?  If you are unsure about where to start or haven't used github before, please feel free to email `@t-brandt`_ or `@m-samland`_.

Feedback and feature requests?  Is there something missing you would like to see?  Please open an issue or send an email to `@t-brandt`_ or `@m-samland`_.


Acknowledgements
----------------

If you have made use of the charis pipeline in your research, please cite:

- `Brandt et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract>`_
- Samland et al. 2022


Documentation
-------------
http://princetonuniversity.github.io/charis-dep/


Your computer should have at least ~2 GB of RAM to extract data cubes, and at least 2 GB/core (and at least 4 GB total) to build the calibration files.  The calibration files can take a long time to generate if you do not have multiple processors.


Installation:
The easy way to install is to use the setup.py in this directory with
python setup.py install
We recommend that you first install the newest anaconda Python if you are not already using anaconda:
https://www.continuum.io/downloads

** CAUTIONARY NOTE: the method below may or may not work with the latest versions of Anaconda and Xcode; it may break either the pipeline or numpy/scipy by linking incompatible libraries.  I recommend not following these instructions.  The setup script as invoked above will attempt to install with openMP support, but will default to an installation without openMP support. **
If you are running this on a Mac, you need gcc from Xcode, and you probably need a homebrew installation of gcc-5 to enable OpenMP linking.  Follow the instructions here:
http://mathcancer.blogspot.com/2016/01/PrepOSXForCoding-Homebrew.html
You may need to specify the C compiler when running the setup script using something like
CC=gcc-5 python setup.py install
or
CC=gcc-mp-5 python setup.py install
Type gcc [tab][tab] in a terminal to see your available gcc compilers.  If you use tcsh instead of bash, your export command will be different, but something like this should work:
set CC = gcc-5
python setup.py install
** END CAUTIONARY NOTE **

Quick-start tutorial, basic usage with sample files:
(note: sample files of the M5 globular cluster are currently hosted at http://web.physics.ucsb.edu/~tbrandt/charis_sample_data/)
** NOTE: file hosting location changed December 2017 **

Example:
CRSA00000001.fits is a 1550-nm monochromatic flat taken in broadband/lowres mode
CRSA00000002.fits through CRSA00000005.fits are broadband/lowres darks
CRSA10000000.fits is a science frame in broadband/lowres mode

First, we must build the calibration files.  Create a directory where they will live, make this the current working directory, and run
buildcal /path/to/CRSA00000001.fits 1550 lowres /path/to/CRSA0000000[2-5].fits
Follow the prompts to create (or not) oversampled PSFlet images.  This routine will create calibration files in the current working directory.  The arguments to buildcal are:
1. The monochromatic flat, as a raw sequence of CHARIS reads
2. The wavelength (in nm) of the monochromatic flat
3. The CHARIS mode (J/H/K/lowres)
4. Background frame(s)/dark(s) (optional)

Newer calibration files require simply, e.g.,
buildcal /path/to/CRSA00000000.fits
with the wavelength and observing mode encoded in the fits header.

Now extract a cube.  First, you need to create an appropriate .ini file by modifying sample.ini in the code directory.  With the example file names given above and with your modified .ini file in the current working directory, you would run
extractcube /path/to/CRSA00000000.fits modified.ini
The arguments are simply
1. The raw file(s) to extract into cubes
2. The configuration file
The extracted cubes will be written to the current working directory.  The first HDU is simply the header with some basic information, the second HDU is the cube, the third HDU is the inverse variance on the cube, and the fourth HDU has no data but saves the original header on HDU0 of the raw reads.
