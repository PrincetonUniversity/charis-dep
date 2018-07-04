#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import logging
import multiprocessing
import re
import time

import numpy as np
from astropy.io import fits

from . import fitramp

try:
    from image import Image
except:
    from charis.image import Image

log = logging.getLogger('main')
from pdb import set_trace


def getreads(filename, header=fits.PrimaryHDU().header, read_idx=[1, None]):
    """
    Get reads from fits file and put them in the correct
    format for the up-the-ramp functions.

    Parameters
    ----------
    filename:  string
        Name of the fits file
    header:    FITS header object
        For recording the reads used.
    read_idx:  list
        [first index, last index] to extract from
                  the hdulist: index = 1 is the first read, use
                  last index = None to use all reads after the
                  first index.  Default [1, None].

    Returns
    -------
    reads:     3D float32 ndarray
        shape = (nreads, ny, nx)

    Notes
    -----
    The input header, if given, will contain keywords for the first
    and last reads used.

    """

    log.info("Getting reads from " + filename)

    hdulist = fits.open(filename)
    shape = hdulist[1].data.shape

    if read_idx[1] is not None:
        if read_idx[1] > read_idx[0]:
            idx1 = read_idx[1] + 1
    else:
        idx1 = read_idx[1]

    reads = np.zeros((len(hdulist[read_idx[0]:idx1]), shape[0], shape[1]),
                     np.float32)

    header.append(('firstrd', read_idx[0], 'First HDU of original file used'), end=True)

    for i, r in enumerate(hdulist[read_idx[0]:idx1]):
        header['lastrd'] = (i + read_idx[0], 'Last HDU of original file used')
        reads[i] = r.data
    return reads


def calcramp(filename, mask=None, gain=2., noisefac=0,
             header=fits.PrimaryHDU().header, read_idx=[1, None],
             maxcpus=multiprocessing.cpu_count(),
             fitnonlin=True, fitexpdecay=True):
    """
    Function calcramp computes the up-the-ramp count rates and their
    inverse variance using the cython function fit_ramp.

    Parameters
    ----------
    filename:  string
        name of the file containing the reads, to be
                 opened by astropy.io.fits.open().  Reads are assumed
                 to reside in HDU 1, 2, ..., N.
    mask:      boolean array
        nonzero for good pixels and zero for
                 bad pixels.  Default None (no masking)
    gain:      float
        electrons/ADU, used to compute shot noise on
                 count rates.  Default 2.
    noisefac:  float
        extra noise to add to derived count rates.
                 This noise will be added in quadrature but will be
                 assumed to be a fixed fraction of the count rate;
                 it is appropriate for handing imperfect PSFlet
                 models.  Default 0.
    header:    FITS header
        Header to hold information on the ramp.  Create
                 a new header if this is not given.
    read_idx:  list
        Reads to use to compute the ramp.  Default [1, None],
                 i.e., use the first and all subsequent reads.
    maxcpus:   Maximum number of threads for OpenMP parallelization
                 in Cython.  Default multiprocessing.cpu_count().
    fitnonlin:  boolean
        fit an approximately measured nonlinear
                 response to each pixel's count rate?  Adds very
                 little to the computational cost.  Default True.
    fitexpdecay:  boolean
        fit for the exponential decay of the
                 reference voltage in the first read (if using the
                 first read)?  Strongly recommended for CHARIS data.
                 Default True.  Only possible if there are at least
                 three reads.

    Returns
    -------
    image:     Image instance
        An Image class containing the derived count rates,
                 their inverse variance, and a FITS header.


    This function is largely a wrapper for the cython function
    fit_ramp, which is calls after first calling getreads().

    """

    header.append(('comment', ''), end=True)
    header.append(('comment', '*' * 60), end=True)
    header.append(('comment', '*' * 19 + ' Ramp, Masking, Noise ' + '*' * 19), end=True)
    header.append(('comment', '*' * 60), end=True)
    header.append(('comment', ''), end=True)

    reads = getreads(filename, header, read_idx)

    try:
        read0 = header['firstrd']
    except:
        read0 = 0

    maskarr = np.ones(reads[0].shape, np.uint16)
    if mask is not None:
        maskarr[:] = mask != 0

    header.append(('pixmask', mask is not None, 'Mask known bad/hot pixels?'), end=True)

    data, ivar = fitramp.fit_ramp(reads, maskarr, tol=1e-5, read0=read0,
                                  gain=gain, maxproc=maxcpus,
                                  fitnonlin=fitnonlin, fitexpdecay=fitexpdecay,
                                  returnivar=True)

    if noisefac > 0:
        ivar[:] = old_div(1., (old_div(1., (ivar + 1e-100)) + (noisefac * data)**2))

    header.append(('gain', gain, 'Assumed detector gain for Poisson variance'), end=True)
    header['noisefac'] = (noisefac, 'Added noise (as fraction of abs(ct rate))')

    header['fitdecay'] = (fitexpdecay, 'Fit exponential decay of ref. volt. in read 1?')
    header['nonlin'] = (fitnonlin, 'Fit nonlinear pixel response?')

    return Image(data=data, ivar=ivar, header=header)
