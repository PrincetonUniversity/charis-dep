"""Tests for the hexagon->square resampling of data, ivar and good-fraction.

These exercise ``resample_ivar_cube`` and ``resample_good_fraction_cube``
against the *real* shipped SPHERE calibration, because the resample's crop
(``[:, 18:-68, 46:-40]``) is hardwired to the 201-lenslet geometry and a
synthetic mini-grid would not survive it.
"""

import json
import os

import numpy as np
import pytest

from charis.image.image_geometry import (
    resample_good_fraction_cube,
    resample_image_cube,
    resample_ivar_cube,
)

CALIBRATION = os.path.join(
    os.path.dirname(__file__), os.pardir, "calibrations", "SPHERE",
    "hexagon_mapping_calibration.json")


@pytest.fixture(scope="module")
def clip_infos():
    with open(CALIBRATION) as fout:
        return json.load(fout)


def test_resample_ivar_reproduces_empirical_variance_of_data_resample(clip_infos):
    """The resampled ivar equals the true inverse variance of the resampled data.

    Push independent unit-variance hexagonal noise through the *data* operator
    and compare its empirical per-pixel variance to ``1 / resample_ivar_cube``
    of a uniform ``ivar = 1`` field.  This ties the ivar propagation to the
    actual data operator -- a mismatched weight, hexagon area, index or crop
    would show up here.
    """
    rng = np.random.RandomState(1)
    n_frames = 400
    noise = rng.randn(n_frames, 201, 201)  # Var(d_h) = 1 = 1 / ivar_h

    empirical_var = resample_image_cube(noise, clip_infos).var(axis=0, ddof=1)

    claimed_ivar = resample_ivar_cube(np.ones((1, 201, 201)), clip_infos)[0]
    interior = claimed_ivar > 0
    ratio = empirical_var[interior] / (1.0 / claimed_ivar[interior])

    assert abs(np.median(ratio) - 1.0) < 0.05
    assert abs(np.mean(ratio) - 1.0) < 0.05


def _pixels_touching_lenslets(clip_infos, masked_lenslets):
    """Pixel-grid mask of squares that draw from any of ``masked_lenslets``.

    Independent geometric oracle for the strict "any bad contributor" rule,
    applying the same reshape/swap/crop as the resample functions.
    """
    masked = set(masked_lenslets)
    number_of_pixels = int(np.sqrt(len(clip_infos)))
    flat = np.zeros(len(clip_infos), dtype=bool)
    for index, clip_info in enumerate(clip_infos):
        if len(clip_info["areas"]) == 0:
            continue
        if masked.intersection(clip_info["hex_indices"]):
            flat[index] = True
    grid = np.swapaxes(flat.reshape(number_of_pixels, number_of_pixels), 0, 1)
    return grid[18:-68, 46:-40]


def test_masked_lenslet_zeros_every_square_it_touches_and_nothing_else(clip_infos):
    """A masked lenslet zeros exactly the output pixels that draw from it.

    ``value_j`` contains ``w_jh * d_h`` from the masked lenslet, so the whole
    square is unusable; a square that draws from no masked lenslet is untouched.
    """
    masked_lenslets = [row * 201 + col
                       for row in range(95, 105) for col in range(95, 105)]
    ivar = np.ones((1, 201, 201))
    for lenslet in masked_lenslets:
        ivar[0, lenslet // 201, lenslet % 201] = 0.0

    base_ivar = resample_ivar_cube(np.ones((1, 201, 201)), clip_infos)[0]
    masked_ivar = resample_ivar_cube(ivar, clip_infos)[0]
    good_fraction = resample_good_fraction_cube(ivar, clip_infos)[0]

    touched = _pixels_touching_lenslets(clip_infos, masked_lenslets)
    untouched = (base_ivar > 0) & ~touched

    assert touched.sum() > 0
    assert np.all(masked_ivar[touched] == 0)
    assert np.all(masked_ivar[untouched] > 0)

    # The soft map agrees: untouched pixels are exactly full, touched ones drop.
    assert np.all(good_fraction[untouched] == 1.0)
    assert np.any(good_fraction[touched] < 1.0)


def test_uniform_field_produces_moire_unlike_the_flux_operator(clip_infos):
    """For a uniform ivar field the corrected map varies (the moire).

    ``sum w**2`` varies square to square (n_eff runs 1-3), so even a perfectly
    uniform input yields a spatially varying ivar -- real structure the wrong
    flux operator (``resample_image_cube`` on ivar) smears into a constant.
    """
    uniform = np.ones((1, 201, 201)) * 5.0

    corrected = resample_ivar_cube(uniform, clip_infos)[0]
    flux_operator = resample_image_cube(uniform, clip_infos)[0]

    interior = corrected > 0
    coefficient_of_variation = corrected[interior].std() / corrected[interior].mean()

    assert coefficient_of_variation > 0.1
    assert not np.allclose(corrected[interior], flux_operator[interior])


def test_resampled_ivar_and_good_fraction_align_with_resampled_data(clip_infos):
    """ivar and good-fraction share the data resample's shape and crop."""
    data_shape = resample_image_cube(np.ones((3, 201, 201)), clip_infos).shape

    assert resample_ivar_cube(np.ones((3, 201, 201)), clip_infos).shape == data_shape
    assert resample_good_fraction_cube(np.ones((3, 201, 201)), clip_infos).shape == data_shape


def test_good_fraction_is_one_when_all_good_and_bounded_when_masked(clip_infos):
    """Good-fraction is exactly 1 on illuminated all-good pixels, and in [0, 1]."""
    all_good = np.ones((1, 201, 201))
    illuminated = resample_image_cube(all_good, clip_infos)[0] > 0

    good_fraction = resample_good_fraction_cube(all_good, clip_infos)[0]
    assert np.allclose(good_fraction[illuminated], 1.0)

    masked = all_good.copy()
    masked[0, 95:105, 95:105] = 0.0
    masked_fraction = resample_good_fraction_cube(masked, clip_infos)[0]

    assert masked_fraction.min() >= 0.0
    assert masked_fraction.max() <= 1.0 + 1e-9
    assert np.any((masked_fraction > 0) & (masked_fraction < 1.0))
