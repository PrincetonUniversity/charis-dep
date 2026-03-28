#!/usr/bin/env python

import logging
import multiprocessing
from builtins import range
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from past.utils import old_div
from scipy import linalg, ndimage, signal

from . import matutils

log = logging.getLogger('main')

# Per-chunk status codes
CHUNK_OK = 0         # local fit succeeded
CHUNK_NO_SIGNAL = 1  # no cross-correlation signal (fallback to global shift)
CHUNK_BOUNDARY = 2   # local fit OK but peak was at offset boundary
CHUNK_NONFINITE = 3  # parabola fit produced inf/nan (fallback to global shift)
CHUNK_CLAMPED = 4    # shift outside offset range, clamped


@dataclass
class FitshiftResult:
    """Diagnostics and result of a fitshift (`calc_offset`) call.

    Attributes
    ----------
    psflets : ndarray, shape (nlam, ny, nx)
        PSFlet templates shifted by the fitted position-dependent offset.
    global_shift : float
        Image-wide sub-pixel shift computed over the full detector.
    shiftarr : ndarray, shape (nchunks_y, nchunks_x)
        Per-chunk sub-pixel shifts.  Failed chunks carry the global shift.
    status_arr : ndarray of int8, shape (nchunks_y, nchunks_x)
        Per-chunk status codes: ``CHUNK_OK=0``, ``CHUNK_NO_SIGNAL=1``,
        ``CHUNK_BOUNDARY=2``, ``CHUNK_NONFINITE=3``, ``CHUNK_CLAMPED=4``.
    fullshiftarr : ndarray, shape (ny, nx)
        Interpolated shift map at full detector resolution.
    nchunks : tuple of int
        ``(nchunks_y, nchunks_x)`` — number of chunks per axis.
    dx : int
        Chunk size in pixels.
    n_good : int
        Number of chunks whose local fit succeeded (status 0 or 2 or 4).
    n_total : int
        Total number of chunks.
    n_no_signal : int
        Chunks with no cross-correlation signal.
    n_boundary_peak : int
        Chunks where the correlation peak was at the edge of the offset range.
    n_nonfinite : int
        Chunks where the parabola fit produced a non-finite shift.
    n_clamped : int
        Chunks where the fitted shift was outside the offset range and was clamped.
    """

    psflets: np.ndarray
    global_shift: float
    shiftarr: np.ndarray
    status_arr: np.ndarray
    fullshiftarr: np.ndarray
    nchunks: tuple
    dx: int
    n_good: int
    n_total: int
    n_no_signal: int
    n_boundary_peak: int
    n_nonfinite: int
    n_clamped: int

    @property
    def fraction_good(self) -> float:
        """Fraction of chunks with usable local signal."""
        return self.n_good / self.n_total if self.n_total > 0 else 0.0

    @property
    def local_shift_std(self) -> float:
        """Standard deviation of local shifts in good chunks."""
        good = (self.status_arr == CHUNK_OK) | (self.status_arr == CHUNK_BOUNDARY) | (self.status_arr == CHUNK_CLAMPED)
        if np.any(good):
            return float(np.std(self.shiftarr[good]))
        return 0.0

    def to_fits(self) -> fits.HDUList:
        """Serialise diagnostics to an ``astropy`` ``HDUList``.

        Returns
        -------
        hdul : fits.HDUList
            Extension 0: full interpolated shift map (ny, nx).
            Extension 1 (CHUNKS): per-chunk shifts (nchunks_y, nchunks_x).
            Extension 2 (STATUS): per-chunk status codes.
        """
        hdr = fits.Header()
        hdr['GSHIFT'] = (self.global_shift, 'Image-wide (global) sub-pixel shift')
        hdr['NGOOD'] = (self.n_good, 'Chunks with usable local fit')
        hdr['NTOTAL'] = (self.n_total, 'Total number of chunks')
        hdr['DX'] = (self.dx, 'Chunk size [px]')
        hdr['NNOSIG'] = (self.n_no_signal, 'Chunks with no signal (used global shift)')
        hdr['NBNDRY'] = (self.n_boundary_peak, 'Chunks with peak at offset boundary')
        hdr['NNONFIN'] = (self.n_nonfinite, 'Chunks with non-finite parabola fit')
        hdr['NCLAMP'] = (self.n_clamped, 'Chunks with shift clamped to offset range')

        hdul = fits.HDUList([
            fits.PrimaryHDU(self.fullshiftarr.astype(np.float32), header=hdr),
            fits.ImageHDU(self.shiftarr.astype(np.float32), name='CHUNKS'),
            fits.ImageHDU(self.status_arr, name='STATUS'),
        ])
        return hdul


def _fit_parabola(corrvals, offsets):
    """Fit a parabola to the peak region of a cross-correlation curve.

    Parameters
    ----------
    corrvals : ndarray, shape (n_offsets,)
        Cross-correlation values at each offset.
    offsets : ndarray, shape (n_offsets,)
        Integer offsets corresponding to each value.

    Returns
    -------
    shift : float
        Sub-pixel offset at the parabola maximum.  ``nan`` if the fit is
        degenerate (zero leading coefficient).
    at_boundary : bool
        ``True`` if the integer peak was at the first or last offset.
    clamped : bool
        ``True`` if the fitted shift was outside the offset range and was
        clipped to the nearest boundary.
    """
    icen = int(np.argmax(corrvals))
    at_boundary = (icen == 0 or icen == offsets.shape[0] - 1)

    imin = max(0, icen - 2)
    imax = min(offsets.shape[0], icen + 3)
    cv = corrvals[imin:imax]

    arr = np.ones((imax - imin, 3))
    arr[:, 1] = offsets[imin:imax]
    arr[:, 2] = offsets[imin:imax] ** 2
    coef = linalg.lstsq(arr, cv)[0]

    if coef[2] == 0:
        return float('nan'), at_boundary, False

    shift = -coef[1] / (2.0 * coef[2])
    if not np.isfinite(shift):
        return float('nan'), at_boundary, False

    clipped = float(np.clip(shift, float(offsets[0]), float(offsets[-1])))
    clamped = clipped != shift
    return clipped, at_boundary, clamped


def calc_offset(psflets, image, offsets, dx=64,
                maxcpus=multiprocessing.cpu_count()):
    """Compute position-dependent sub-pixel shift in the PSFlet spot grid.

    Uses cross-correlation between oversampled PSFlet templates and the
    science image, evaluated in spatial chunks of size ``dx × dx`` pixels,
    to find a position-dependent sub-pixel offset that maximises the match.

    A global (image-wide) shift is always computed first and used as the
    fallback value for chunks where the local fit fails or has no signal.

    Parameters
    ----------
    psflets : ndarray, shape (nlam, ny, upsamp * nx)
        Oversampled PSFlet templates. The last dimension is oversampled by
        factor ``upsamp`` perpendicular to the dispersion direction.
    image : Image
        Input science frame with ``image.data`` (ny, nx) and ``image.ivar``
        (ny, nx).
    offsets : ndarray of int, shape (n_offsets,)
        Integer offsets (in oversampled pixels) at which to evaluate the
        cross-correlation.
    dx : int, optional
        Chunk size in pixels. The detector is split into ``ceil(ny/dx)`` by
        ``ceil(nx/dx)`` chunks. Default 64.
    maxcpus : int, optional
        Number of OpenMP threads. Default ``multiprocessing.cpu_count()``.

    Returns
    -------
    result : FitshiftResult
        Shifted PSFlet templates and full diagnostics (global shift, per-chunk
        shifts, per-chunk status codes, interpolated shift map, counters).

    Raises
    ------
    RuntimeError
        If the image-wide cross-correlation returns no usable signal
        (e.g., the entire image has zero inverse variance).

    Notes
    -----
    Per-chunk status codes are defined as module-level constants:
    ``CHUNK_OK=0``, ``CHUNK_NO_SIGNAL=1``, ``CHUNK_BOUNDARY=2``,
    ``CHUNK_NONFINITE=3``, ``CHUNK_CLAMPED=4``.
    """

    mask = (image.ivar > 0).astype(np.uint16)

    if psflets.dtype != 'float32':  # Ensure correct dtype, native byte order
        psflets2 = np.empty(psflets.shape, np.float32)
        psflets2[:] = psflets
        psflets = psflets2

    ny, nx = image.data.shape
    nchunks_y = int(np.ceil(ny * 1. / dx))
    nchunks_x = int(np.ceil(nx * 1. / dx))

    #####################################################################
    # Compute the global (image-wide) shift first.  This is used as the
    # fallback for chunks where the local cross-correlation fails.
    #####################################################################

    corrvals_global = matutils.crosscorr(
        psflets, image.data.astype('float64'),
        image.ivar.astype('float64'),
        offsets, maxproc=maxcpus, m1=0, m2=nx)

    if corrvals_global is None:
        raise RuntimeError(
            "fitshift failed: image-wide cross-correlation returned no data "
            "(image may be empty or entirely masked)")

    corrvals_global_sum = np.sum(corrvals_global, axis=1)
    if np.all(corrvals_global_sum == 0) or not np.any(np.isfinite(corrvals_global_sum)):
        raise RuntimeError(
            "fitshift failed: image-wide cross-correlation has no signal "
            "(image may be empty or entirely masked)")

    global_shift, global_at_boundary, _ = _fit_parabola(corrvals_global_sum, offsets)
    if not np.isfinite(global_shift):
        raise RuntimeError(
            "fitshift failed: global parabola fit produced a non-finite shift "
            "(degenerate cross-correlation)")

    log.info("fitshift: global (image-wide) shift = %.4f%s",
             global_shift, " [at boundary]" if global_at_boundary else "")

    #####################################################################
    # Calculate the cross-correlation per chunk.  Initialise the shift
    # array with the global shift so that failed chunks fall back to it
    # automatically without any special-casing.
    #####################################################################

    shiftarr = np.full((nchunks_y, nchunks_x), global_shift)
    status_arr = np.full((nchunks_y, nchunks_x), CHUNK_NO_SIGNAL, dtype=np.int8)

    n_total = nchunks_y * nchunks_x
    n_no_signal = 0
    n_boundary_peak = 0
    n_nonfinite = 0
    n_clamped = 0

    log.info(
        "fitshift: computing cross-correlation on %d x %d grid "
        "(%d px chunks, %d x %d detector)",
        nchunks_y, nchunks_x, dx, ny, nx)

    for i in range(0, nx, dx):
        corrvals_all = matutils.crosscorr(psflets, image.data.astype('float64'),
                                          image.ivar.astype('float64'),
                                          offsets, maxproc=maxcpus,
                                          m1=i, m2=i + dx)

        if corrvals_all is None:
            # crosscorr returns None when the column strip is too narrow
            # or has no valid pixels — chunks already carry global_shift.
            log.debug(
                "fitshift: crosscorr returned None for column strip "
                "[%d:%d], using global shift (%.4f)", i, i + dx, global_shift)
            for j in range(0, ny, dx):
                n_no_signal += 1
            continue

        for j in range(0, ny, dx):

            corrvals = np.sum(corrvals_all[:, j:j + dx], axis=1)

            # Check for zero signal in this chunk
            if np.all(corrvals == 0) or not np.any(np.isfinite(corrvals)):
                log.debug(
                    "fitshift: no signal in chunk (row %d-%d, col %d-%d), "
                    "using global shift (%.4f)",
                    j, j + dx, i, i + dx, global_shift)
                n_no_signal += 1
                # shiftarr already holds global_shift; status already CHUNK_NO_SIGNAL
                continue

            shift, at_boundary, clamped = _fit_parabola(corrvals, offsets)

            if not np.isfinite(shift):
                n_nonfinite += 1
                status_arr[j // dx, i // dx] = CHUNK_NONFINITE
                # shiftarr already holds global_shift as fallback
                log.debug(
                    "fitshift: non-finite shift in chunk (row %d-%d, col %d-%d), "
                    "using global shift (%.4f)", j, j + dx, i, i + dx, global_shift)
            elif clamped:
                n_clamped += 1
                if at_boundary:
                    n_boundary_peak += 1
                status_arr[j // dx, i // dx] = CHUNK_CLAMPED
                shiftarr[j // dx, i // dx] = shift
            elif at_boundary:
                n_boundary_peak += 1
                status_arr[j // dx, i // dx] = CHUNK_BOUNDARY
                shiftarr[j // dx, i // dx] = shift
            else:
                status_arr[j // dx, i // dx] = CHUNK_OK
                shiftarr[j // dx, i // dx] = shift

    # Log summary of chunk quality
    n_good = n_total - n_no_signal - n_nonfinite
    log.info(
        "fitshift: %d/%d chunks with usable local fit, shift range [%.3f, %.3f]",
        n_good, n_total,
        float(np.min(shiftarr)), float(np.max(shiftarr)))
    if n_boundary_peak > 0:
        log.info(
            "fitshift: %d/%d good chunks had correlation peak at offset boundary",
            n_boundary_peak, n_good if n_good > 0 else 1)
    if n_nonfinite > 0:
        log.warning(
            "fitshift: %d/%d chunks had non-finite shift (using global shift %.4f)",
            n_nonfinite, n_good if n_good > 0 else 1, global_shift)
    if n_clamped > 0:
        log.info(
            "fitshift: %d/%d chunk shifts clamped to offset range [%d, %d]",
            n_clamped, n_good if n_good > 0 else 1,
            int(offsets[0]), int(offsets[-1]))
    if n_no_signal > 0:
        log.warning(
            "fitshift: %d/%d chunks had no signal (using global shift %.4f)",
            n_no_signal, n_total, global_shift)

    # Compare local shifts to global for spatial variation diagnostic
    good_mask = (status_arr == CHUNK_OK) | (status_arr == CHUNK_BOUNDARY) | (status_arr == CHUNK_CLAMPED)
    if np.any(good_mask):
        local_shifts = shiftarr[good_mask]
        deviation = local_shifts - global_shift
        log.info(
            "fitshift: local vs global — mean delta=%.4f, std=%.4f, "
            "max |delta|=%.4f (%d good chunks)",
            float(np.mean(deviation)), float(np.std(deviation)),
            float(np.max(np.abs(deviation))), int(np.sum(good_mask)))
        if float(np.std(deviation)) < 0.05:
            log.info(
                "fitshift: shifts are spatially uniform — "
                "fitshift_nchunks=1 would give equivalent results")

    #####################################################################
    # Median-filter the shift map and interpolate to full resolution
    #####################################################################

    if shiftarr.shape[0] >= 3:
        shiftarr[1:-1, 1:-1] = signal.medfilt2d(shiftarr, 3)[1:-1, 1:-1]
        shiftarr[0, 1:-1] = signal.medfilt(shiftarr[0], 3)[1:-1]
        shiftarr[-1, 1:-1] = signal.medfilt(shiftarr[-1], 3)[1:-1]
        shiftarr[1:-1, 0] = signal.medfilt(shiftarr[:, 0], 3)[1:-1]
        shiftarr[1:-1, -1] = signal.medfilt(shiftarr[:, -1], 3)[1:-1]

    x = old_div((1. * np.arange(ny)), dx) - 0.5
    x *= x > 0
    x[np.where(x > shiftarr.shape[0] - 1)] = shiftarr.shape[0] - 1
    x, y = np.meshgrid(x, x)

    fullshiftarr = ndimage.map_coordinates(shiftarr, [y, x], order=3)

    psflets = matutils.interpcal(psflets, image.data, mask, fullshiftarr,
                                 maxproc=maxcpus)

    return FitshiftResult(
        psflets=psflets,
        global_shift=float(global_shift),
        shiftarr=shiftarr,
        status_arr=status_arr,
        fullshiftarr=fullshiftarr,
        nchunks=(nchunks_y, nchunks_x),
        dx=dx,
        n_good=n_good,
        n_total=n_total,
        n_no_signal=n_no_signal,
        n_boundary_peak=n_boundary_peak,
        n_nonfinite=n_nonfinite,
        n_clamped=n_clamped,
    )
