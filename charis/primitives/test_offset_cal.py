"""Tests for FitshiftResult QC-diagnostic serialisation and writing."""

import os

import numpy as np
from astropy.io import fits

from charis.primitives.offset_cal import CHUNK_OK, FitshiftResult


def _make_result(nchunks=(2, 2), ny=8, nx=8):
    ncy, ncx = nchunks
    return FitshiftResult(
        psflets=np.zeros((3, ny, nx), dtype=np.float32),
        global_shift=0.25,
        shiftarr=np.full((ncy, ncx), 0.25, dtype=np.float32),
        status_arr=np.full((ncy, ncx), CHUNK_OK, dtype=np.int8),
        fullshiftarr=np.full((ny, nx), 0.25, dtype=np.float32),
        nchunks=nchunks,
        dx=ny // ncy,
        n_good=ncy * ncx,
        n_total=ncy * ncx,
        n_no_signal=0,
        n_boundary_peak=0,
        n_nonfinite=0,
        n_clamped=0,
    )


def test_to_fits_has_shiftmap_chunks_and_status():
    hdul = _make_result().to_fits()
    assert len(hdul) == 3
    assert hdul[1].name == 'CHUNKS'
    assert hdul[2].name == 'STATUS'
    assert hdul[0].header['GSHIFT'] == 0.25


def test_write_diagnostics_writes_file_and_returns_true(tmp_path):
    path = str(tmp_path / 'frame_fitshift_diag.fits')
    assert _make_result().write_diagnostics(path) is True
    assert os.path.exists(path)
    with fits.open(path) as hdul:
        assert hdul[2].name == 'STATUS'


def test_write_diagnostics_creates_missing_parent_dirs(tmp_path):
    path = str(tmp_path / 'does' / 'not' / 'exist' / 'diag.fits')
    assert _make_result().write_diagnostics(path) is True
    assert os.path.exists(path)


def test_write_diagnostics_swallows_write_failure_and_returns_false(tmp_path):
    """A QC-write failure must never propagate: the fit result is already in
    hand, so a bad path is logged and ignored, not raised."""
    blocker = tmp_path / 'blocker'
    blocker.write_text('i am a file, not a directory')
    # makedirs on a path whose parent is a regular file raises NotADirectoryError
    path = str(blocker / 'sub' / 'diag.fits')
    assert _make_result().write_diagnostics(path) is False
