"""Tests for sph_ifs_fix_badpix.

These pin the two properties the extraction relies on: that the repair touches only bad
pixels, and that interpolating ``ivar`` before zeroing it is redundant. The third pins the
in-place mutation of the caller's bad-pixel map, which is a live side effect rather than a
desired behaviour -- see
https://github.com/PrincetonUniversity/charis-dep/issues/50.
"""

import numpy as np

from charis.tools.toolbox import sph_ifs_fix_badpix

EXT = 10  # border sph_ifs_fix_badpix clears, and the reach of its spectral search


def _frame(size=40):
    """A frame that varies smoothly along y, as a microspectrum does."""
    y = np.arange(size, dtype=float)
    return np.tile(y[:, None] ** 1.5 + 100.0, (1, size))


def _bpm(size=40, positions=((20, 20),)):
    bpm = np.zeros((size, size), dtype=int)
    for y, x in positions:
        bpm[y, x] = 1
    return bpm


def test_good_pixels_are_never_modified():
    img = _frame()
    bpm = _bpm(positions=((20, 20), (21, 20), (25, 31)))
    out = sph_ifs_fix_badpix(img=img.copy(), bpm=bpm.copy())

    changed = np.nonzero(out != img)
    assert set(zip(*changed)) <= {(20, 20), (21, 20), (25, 31)}


def test_repaired_value_follows_the_local_gradient():
    img = _frame()
    bpm = _bpm(positions=((20, 20),))
    out = sph_ifs_fix_badpix(img=img.copy(), bpm=bpm.copy())

    assert out[20, 20] != img[20, 20]
    assert abs(out[20, 20] - img[20, 20]) < 0.05 * img[20, 20]


def test_interpolating_ivar_before_zeroing_it_is_redundant():
    """The reason the ivar interpolation was removed from the extraction path.

    Repairing ivar and then zeroing the bad pixels gives the same array as zeroing them
    directly, because the repair only ever writes where the mask is set.
    """
    ivar = 1.0 / _frame()
    bpm = _bpm(positions=((20, 20), (21, 20), (25, 31)))
    mask = bpm.astype(bool)

    direct = ivar.copy()
    direct[mask] = 0

    interpolated = sph_ifs_fix_badpix(img=ivar.copy(), bpm=bpm.copy())
    interpolated[mask] = 0

    assert np.array_equal(direct, interpolated)


def test_caller_bad_pixel_map_is_cleared_at_the_border():
    """Characterisation, not endorsement: the function mutates the caller's bpm.

    ``extractcube`` derives ``good_pixel_mask`` from ``bpm`` *after* this call, so the
    cleared border currently decides which pixels get flat-fielded. Changing the function
    to copy its input would therefore change the delivered arrays.
    """
    bpm = _bpm(positions=((2, 2), (20, 20)))
    sph_ifs_fix_badpix(img=_frame(), bpm=bpm)

    assert bpm[2, 2] == 0, "border flag was cleared in the caller's array"
    assert bpm[20, 20] == 1, "interior flags survive"
