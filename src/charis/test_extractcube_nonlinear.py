"""Tests for the opt-in SPHERE saturation mask (issue #51)."""
import numpy as np

from charis.extractcube import nonlinear_pixel_mask

THRESHOLD = 40000.


def test_threshold_none_flags_nothing():
    data = np.full((10, 10), 1e6)
    mask = nonlinear_pixel_mask(data, None)
    assert mask.shape == data.shape
    assert mask.dtype == bool
    assert not mask.any()


def test_threshold_none_flags_nothing_3d():
    mask = nonlinear_pixel_mask(np.full((3, 10, 10), 1e6), None)
    assert mask.shape == (10, 10)
    assert not mask.any()


def test_threshold_flags_pixels_above_it():
    data = np.full((10, 10), 1000.)
    data[0, 0] = data[3, 4] = data[9, 9] = 50000.
    mask = nonlinear_pixel_mask(data, THRESHOLD)
    assert mask.sum() == 3
    assert mask[0, 0] and mask[3, 4] and mask[9, 9]


def test_threshold_uses_the_brightest_dit():
    data = np.full((4, 10, 10), 1000.)
    data[2, 5, 5] = 45000.
    assert data.mean(axis=0)[5, 5] < THRESHOLD
    mask = nonlinear_pixel_mask(data, THRESHOLD)
    assert mask.sum() == 1
    assert mask[5, 5]
