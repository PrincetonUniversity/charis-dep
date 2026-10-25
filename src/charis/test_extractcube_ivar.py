"""Tests for the SPHERE variance model (issues #52 and #62)."""
import numpy as np
import pytest

from charis.extractcube import sky_variance, sphere_readnoise, sphere_variance

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


def test_readnoise_is_interpolated_and_held_outside_the_measured_range():
    assert 7.3 < sphere_readnoise(16.) < 7.6
    assert sphere_readnoise(1.) == pytest.approx(8.6)
    assert sphere_readnoise(64.) == pytest.approx(7.3)


def test_sky_variance_of_a_median_sky_frame():
    """The scaled median of ``nsky`` raw sky DITs has about ``norm**2 * sky_variance``.

    pi/2 is the large-N variance factor of the median; for 9 frames the true factor is about
    1.49, so the model is up to ~5% conservative here.
    """
    rng = np.random.default_rng(62)
    sky_counts, nsky, norm = 300., 9, 1.3
    sky_frames = (rng.poisson(sky_counts * GAIN, size=(N_SAMPLES, nsky))
                  + rng.normal(0, READNOISE, size=(N_SAMPLES, nsky))) / GAIN
    measured = (norm * np.median(sky_frames, axis=1)).var(ddof=1)
    expected = norm**2 * sky_variance(sky_counts, GAIN, READNOISE, nsky)
    assert np.isclose(expected, measured, rtol=0.1)
    assert expected > measured


def test_sky_variance_of_one_or_two_frames_is_the_mean():
    single = sphere_variance(COUNTS, GAIN, READNOISE, 1, 0)
    assert sky_variance(COUNTS, GAIN, READNOISE, 1) == pytest.approx(single)
    assert sky_variance(COUNTS, GAIN, READNOISE, 2) == pytest.approx(single / 2)
