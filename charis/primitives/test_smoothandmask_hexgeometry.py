"""Tests for the one-sided hexagonal bad-lenslet flagging.

The cube geometry is synthetic: a small square grid whose neighbour lists are built by hand, so
these run without the shipped SPHERE calibration.
"""

import numpy as np
import pytest

from charis.image.image import Image
from charis.image.image_geometry import count_finite_hex_cube
from charis.primitives.fit_psflets import (
    _out_of_field_lenslets,
    _smoothandmask_hexgeometry,
    _static_bad_lenslets,
)

SIZE = 9
NLAM = 3
CENTRE = (SIZE // 2) * SIZE + SIZE // 2


def _neighbour_indices(size=SIZE, nneigh=6):
    """Six neighbours for every interior spaxel, fewer at the array border."""
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)][:nneigh]
    indices = []
    for y in range(size):
        for x in range(size):
            neighbours = [(y + dy) * size + (x + dx) for dy, dx in offsets
                          if 0 <= y + dy < size and 0 <= x + dx < size]
            indices.append(np.array(sorted(neighbours), dtype=int))
    return indices


def _cube(ivar_value=1.0, data_value=10.0):
    data = np.full((NLAM, SIZE, SIZE), data_value)
    ivar = np.full((NLAM, SIZE, SIZE), ivar_value)
    return Image(data=data, ivar=ivar)


def _run(cube, **kwargs):
    return _smoothandmask_hexgeometry(cube, _neighbour_indices(), **kwargs)


def _centre_masked(cube):
    return bool(cube.ivar.reshape(NLAM, -1)[0, CENTRE] == 0)


def test_ivar_deficit_is_masked():
    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.2
    # break the exact uniformity so mad_std is nonzero and the floor is not what decides
    cube.ivar[0] += np.linspace(0, 0.05, SIZE * SIZE).reshape(SIZE, SIZE)
    result = _run(cube)
    assert _centre_masked(result)


def test_ivar_excess_is_not_masked():
    """The mirror image of a masked deficit must survive: the criterion is one-sided."""
    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 1.8
    cube.ivar[0] += np.linspace(0, 0.05, SIZE * SIZE).reshape(SIZE, SIZE)
    result = _run(cube)
    assert not _centre_masked(result)


def test_scale_floor_protects_a_near_uniform_neighbourhood():
    """A 1% deficit in a neighbourhood with a collapsed mad_std must not mask."""
    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.99
    assert not _centre_masked(_run(cube))
    assert _centre_masked(_run(_cube_with_small_deficit(), ivar_mad_floor=0.0))


def _cube_with_small_deficit():
    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.99
    return cube


def test_minimum_neighbour_gate():
    """A large deficit is masked with six neighbours and exempt with four."""
    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.1
    cube.ivar[0] += np.linspace(0, 0.05, SIZE * SIZE).reshape(SIZE, SIZE)
    assert _centre_masked(_run(cube, min_neighbours=6))

    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.1
    cube.ivar[0] += np.linspace(0, 0.05, SIZE * SIZE).reshape(SIZE, SIZE)
    assert not _centre_masked(_run(cube, min_neighbours=7))


def test_zero_ivar_input_is_masked_and_data_stays_finite():
    cube = _cube()
    cube.ivar[:, SIZE // 2, SIZE // 2] = 0.0
    result = _run(cube)
    assert _centre_masked(result)
    assert np.all(np.isfinite(result.data))


def test_masked_spaxel_takes_the_neighbour_median_flux():
    cube = _cube()
    cube.data[0, SIZE // 2, SIZE // 2] = 1e6
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.0
    result = _run(cube)
    assert result.data[0, SIZE // 2, SIZE // 2] == pytest.approx(10.0)


def test_masked_spaxel_without_finite_neighbours_keeps_its_value():
    """Writing NaN here would propagate into every square pixel the lenslet touches."""
    cube = _cube()
    cube.ivar[0] = 0.0
    cube.data[0] = np.nan
    cube.data[0, SIZE // 2, SIZE // 2] = 42.0
    result = _run(cube)
    assert np.all(result.ivar[0] == 0)
    assert result.data[0, SIZE // 2, SIZE // 2] == 42.0


def test_out_of_field_zeros_stay_masked():
    cube = _cube()
    cube.data[:, 0, :] = 0.0
    cube.ivar[:, 0, :] = 0.0
    result = _run(cube)
    assert np.all(result.ivar[:, 0, :] == 0)


def test_out_of_field_lenslets_are_not_filled_from_their_neighbours():
    """Filling the rim would enlarge the apparent footprint by one lenslet."""
    cube = _cube()
    cube.data[:, 0, :] = 0.0
    cube.ivar[:, 0, :] = 0.0
    result = _run(cube)
    assert np.all(np.isnan(result.data[:, 0, :]))
    assert np.all(np.isfinite(result.data[:, 1:, :]))


def _out_of_field_top_row():
    """Top row of the grid outside the field, everything else inside."""
    outside = np.zeros((SIZE, SIZE), dtype=bool)
    outside[0, :] = True
    return outside


def _rim_cube():
    """Out-of-field top row whose flux is an exact zero at every wavelength but one."""
    cube = _cube()
    cube.data[:, 0, :] = 0.0
    cube.data[1, 0, :] = 0.3          # crosstalk from the neighbouring microspectra
    cube.ivar[:, 0, :] = 0.0
    return cube


def test_explicit_footprint_masks_the_rim_even_where_it_carries_flux():
    """The rim's crosstalk flux must not make it look in-field at some wavelengths.

    This is the case an inferred footprint gets wrong: an exact zero at one wavelength and
    a small nonzero at the next, so the same lenslet flips between filled and NaN.
    """
    inferred = _run(_rim_cube())
    assert np.isnan(inferred.data[0, 0, 0]), "exact zero is inferred out-of-field"
    assert np.isfinite(inferred.data[1, 0, 0]), "nonzero rim flux is inferred in-field"

    explicit = _run(_rim_cube(), out_of_field=_out_of_field_top_row())
    assert np.all(np.isnan(explicit.data[:, 0, :])), "the whole rim stays out-of-field"
    assert np.all(explicit.ivar[:, 0, :] == 0)


def test_no_nan_survives_inside_the_field():
    """The guarantee downstream relies on: NaN only ever marks the out-of-field border."""
    cube = _cube()
    cube.data[:, 0, :] = 0.0
    cube.ivar[:, 0, :] = 0.0
    cube.data[0, 4, 4] = np.nan                    # isolated defect
    cube.data[1, 2:5, 2:5] = np.nan                # whole neighbourhood gone
    cube.ivar[1, 2:5, 2:5] = 0.0

    result = _run(cube, out_of_field=_out_of_field_top_row())
    in_field = result.data[:, 1:, :]
    assert np.all(np.isfinite(in_field)), "no NaN may remain inside the field"
    assert np.all(np.isnan(result.data[:, 0, :])), "the border keeps its NaN"
    assert result.data[0, 4, 4] == pytest.approx(10.0), "isolated defect is interpolated"
    assert np.all(result.ivar[1, 2:5, 2:5] == 0), "unfillable spaxels still carry ivar 0"


def test_stranded_spaxels_fall_back_to_zero_not_nan():
    """Zero is neutral in the area-weighted resample; NaN would poison every square."""
    cube = _cube()
    cube.data[0] = np.nan
    cube.ivar[0] = 0.0
    result = _run(cube, out_of_field=np.zeros((SIZE, SIZE), dtype=bool))
    assert np.all(result.data[0] == 0.0)
    assert np.all(result.ivar[0] == 0)


def test_out_of_field_lenslets_are_a_subset_of_the_static_bad_ones():
    """Weak but real lenslets stay in the field so their flux can be interpolated."""
    flat = np.array([[0.0, 1.0, 0.5, 0.8, 1.02]])
    assert np.array_equal(_out_of_field_lenslets(flat),
                          np.array([[True, True, False, False, False]]))
    assert np.all(_static_bad_lenslets(flat)[_out_of_field_lenslets(flat)])


def test_idempotent():
    cube = _cube()
    cube.ivar[0, SIZE // 2, SIZE // 2] = 0.2
    cube.ivar[0] += np.linspace(0, 0.05, SIZE * SIZE).reshape(SIZE, SIZE)
    once = _run(cube)
    first_mask = (once.ivar == 0).copy()
    first_data = once.data.copy()
    twice = _run(once)
    assert np.array_equal(twice.ivar == 0, first_mask)
    assert np.allclose(twice.data, first_data, equal_nan=True)


def test_threshold_is_the_only_thing_separating_masked_from_kept():
    """Golden case: a deficit sized to sit either side of the default threshold."""
    scale = 0.02  # floor dominates: scale = 0.02 * neighbour median
    for factor, expected in ((1.0 - 9 * scale, True), (1.0 - 7 * scale, False)):
        cube = _cube()
        cube.ivar[0, SIZE // 2, SIZE // 2] = factor
        assert _centre_masked(_run(cube)) is expected


def test_count_finite_hex_cube_matches_a_naive_count():
    rng = np.random.default_rng(0)
    flat = rng.standard_normal((NLAM, SIZE * SIZE))
    flat[flat < -0.5] = np.nan
    indices = _neighbour_indices()
    counts = count_finite_hex_cube(flat, indices)
    for spaxel, neighbours in enumerate(indices):
        for lam in range(NLAM):
            assert counts[lam, spaxel] == np.isfinite(flat[lam, neighbours]).sum()


def test_static_bad_lenslets_covers_zero_sentinel_and_low_throughput():
    flat = np.array([[0.0, 1.0, 0.5, 0.8, 1.02]])
    assert np.array_equal(_static_bad_lenslets(flat),
                          np.array([[True, True, True, False, False]]))
