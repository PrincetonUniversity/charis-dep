"""Calibration data is resolved from a local cache and downloaded from Zenodo only when missing."""

import hashlib
import io
import tarfile

import pytest
from astropy.io import fits

from charis import calibration_data
from charis.calibration_data import REGISTRY, calibration_root, fetch
from charis.extractcube import _append_calibration_header
from charis.instruments import SPHERE

VERSION = REGISTRY["SPHERE"]["version"]


def _archive(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}


@pytest.fixture
def no_network(monkeypatch):
    def urlopen(*args, **kwargs):
        raise AssertionError("tried to download")
    monkeypatch.setattr(calibration_data, "urlopen", urlopen)


@pytest.fixture
def zenodo(monkeypatch):
    """Serve a tiny SPHERE archive whose md5 matches the registry; returns the requested URLs."""
    content = _archive({"SPHERE/YH/lamsol.dat": b"lamsol\n", "SPHERE/mask.fits": b"mask\n"})
    monkeypatch.setitem(REGISTRY["SPHERE"], "record_id", 1234)
    monkeypatch.setitem(REGISTRY["SPHERE"], "md5", hashlib.md5(content).hexdigest())
    requested = []

    def urlopen(request, timeout):
        requested.append(request.full_url)
        return _Response(content)
    monkeypatch.setattr(calibration_data, "urlopen", urlopen)
    return requested


def test_existing_data_is_used_without_download(tmp_path, monkeypatch, no_network):
    (tmp_path / "env" / VERSION / "SPHERE").mkdir(parents=True)
    (tmp_path / "cache" / "charis" / "calibrations" / VERSION / "SPHERE").mkdir(parents=True)
    monkeypatch.setenv("CHARIS_CALIBRATION_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert calibration_root("SPHERE") == tmp_path / "env" / VERSION


def test_missing_data_without_download_names_the_fetch_command(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv("CHARIS_CALIBRATION_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="charis-calibrations fetch SPHERE") as excinfo:
        calibration_root("SPHERE", download=False)
    assert str(tmp_path / VERSION / "SPHERE") in str(excinfo.value)
    assert VERSION in str(excinfo.value)


def test_fetch_downloads_checks_and_extracts(tmp_path, zenodo):
    instrument_dir = fetch("SPHERE", tmp_path)

    assert instrument_dir == tmp_path / VERSION / "SPHERE"
    assert (instrument_dir / "YH" / "lamsol.dat").read_bytes() == b"lamsol\n"
    assert zenodo == [f"https://zenodo.org/api/records/1234/files/{REGISTRY['SPHERE']['archive']}/content"]
    assert [p.name for p in (tmp_path / VERSION).iterdir()] == ["SPHERE"]


def test_fetch_rejects_a_bad_checksum(tmp_path, monkeypatch, zenodo):
    monkeypatch.setitem(REGISTRY["SPHERE"], "md5", "0" * 32)

    with pytest.raises(OSError, match="md5"):
        fetch("SPHERE", tmp_path)
    assert list((tmp_path / VERSION).iterdir()) == []


def test_fetch_never_replaces_an_existing_tree(tmp_path, zenodo):
    existing = tmp_path / VERSION / "SPHERE"
    existing.mkdir(parents=True)
    (existing / "mine.txt").write_text("keep")

    assert fetch("SPHERE", tmp_path) == existing
    assert [p.name for p in existing.iterdir()] == ["mine.txt"]
    assert [p.name for p in (tmp_path / VERSION).iterdir()] == ["SPHERE"]


def test_cube_header_records_the_calibration_version(calibrations, tmp_path):
    fits.PrimaryHDU(header=fits.Header([("CAL_BAND", "YH")])).writeto(tmp_path / "cal_params.fits")
    header = fits.Header()
    _append_calibration_header(header, str(tmp_path), SPHERE("YH"))
    assert header["CALVERS"] == REGISTRY["SPHERE"]["version"]
    assert header["CALDOI"] == REGISTRY["SPHERE"]["doi"]
    assert header["CAL_BAND"] == "YH"

    header = fits.Header()
    _append_calibration_header(header, str(tmp_path), SPHERE("YH", static_calibdir=str(calibration_root("SPHERE"))))
    assert "CALVERS" not in header
    assert "CALDOI" not in header
