"""A serial build must say so at run time when more than one thread is requested."""

import logging

import pytest

from charis.primitives import openmp_check


@pytest.fixture
def log():
    return logging.getLogger("charis.test_openmp_check")


def test_warns_when_threads_are_requested_from_a_serial_build(monkeypatch, caplog, log):
    monkeypatch.setattr(openmp_check, "openmp_enabled", lambda: False)
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert openmp_check.warn_if_serial(4, log) is True
    assert "without OpenMP" in caplog.text
    assert "4 threads" in caplog.text


def test_silent_for_a_single_thread(monkeypatch, caplog, log):
    # spherical runs one process per DIT with maxcpus=1; a serial build costs it nothing.
    monkeypatch.setattr(openmp_check, "openmp_enabled", lambda: False)
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert openmp_check.warn_if_serial(1, log) is False
    assert caplog.text == ""


def test_silent_when_built_with_openmp(monkeypatch, caplog, log):
    monkeypatch.setattr(openmp_check, "openmp_enabled", lambda: True)
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert openmp_check.warn_if_serial(8, log) is False
    assert caplog.text == ""
