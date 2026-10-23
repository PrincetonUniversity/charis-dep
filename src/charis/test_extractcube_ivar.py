"""Tests for the SPHERE variance model (issue #52)."""
import numpy as np

from charis.extractcube import sphere_variance

COUNTS = 1000.
GAIN = 1.8
READNOISE = 6.  # electrons
N_SAMPLES = 200_000


def _draw_counts(rng, shape):
    """Counts from Poisson(c * g) electrons plus Gaussian read noise (in e-)."""
    electrons = rng.poisson(COUNTS * GAIN, size=shape) + rng.normal(0, READNOISE, size=shape)
    return electrons / GAIN


def test_sphere_variance_matches_poisson_in_counts():
    rng = np.random.default_rng(52)
    samples = _draw_counts(rng, N_SAMPLES)
    expected = sphere_variance(COUNTS, GAIN, READNOISE, 1, 0)
    assert np.isclose(expected, samples.var(ddof=1), rtol=0.02)


def test_sphere_ivar_of_a_mean_of_ndit_frames():
    ndit = 16
    rng = np.random.default_rng(53)
    samples = _draw_counts(rng, (N_SAMPLES, ndit)).mean(axis=1)
    expected = sphere_variance(COUNTS, GAIN, READNOISE, ndit, 0)
    assert np.isclose(expected, samples.var(ddof=1), rtol=0.02)
    assert np.isclose(expected, sphere_variance(COUNTS, GAIN, READNOISE, 1, 0) / ndit, rtol=1e-12)


def test_noisefac_term_does_not_average_down():
    diff = (sphere_variance(COUNTS, GAIN, READNOISE, 16, 0.05)
            - sphere_variance(COUNTS, GAIN, READNOISE, 16, 0))
    assert np.isclose(diff, (0.05 * COUNTS)**2, rtol=1e-9)
