"""Locate the static calibration data, downloading it from Zenodo on first use.

Each charis release pins the exact calibration set it was tested with in
``calibration_registry.json``: one Zenodo version record per instrument. The data lives in
``<root>/<version>/<instrument>/``, where ``<root>`` is ``$CHARIS_CALIBRATION_DIR`` if set,
else ``${XDG_CACHE_HOME:-~/.cache}/charis/calibrations``.
"""

import hashlib
import json
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
from importlib.resources import files
from urllib.request import Request, urlopen

import charis

__all__ = ["REGISTRY", "cache_root", "calibration_root", "fetch"]

log = logging.getLogger(__name__)

REGISTRY = json.loads(files("charis").joinpath("calibration_registry.json").read_text())


def cache_root():
    """The directory that holds one folder per calibration data version."""
    if os.environ.get("CHARIS_CALIBRATION_DIR"):
        return pathlib.Path(os.environ["CHARIS_CALIBRATION_DIR"])
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(cache) / "charis" / "calibrations"


def calibration_root(instrument, *, download=True):
    """Return the directory that contains ``<instrument>/``, fetching the data if it is missing.

    Parameters
    ----------
    instrument : str
        ``'SPHERE'`` or ``'CHARIS'``.
    download : bool, optional
        Download the data from Zenodo if it is not there yet. Default True.

    Returns
    -------
    pathlib.Path
        ``<root>/<version>``, so that ``os.path.join(path, instrument)`` is the instrument's
        calibration directory.

    Raises
    ------
    FileNotFoundError
        If the data is missing and ``download`` is False.
    """
    version = REGISTRY[instrument]["version"]
    root = cache_root()
    instrument_dir = root / version / instrument
    # An existing tree is used as is, without any network access, so read-only shared
    # installations work.
    if instrument_dir.is_dir():
        return instrument_dir.parent
    if not download:
        raise FileNotFoundError(
            f"No {instrument} calibration data version {version} in {instrument_dir}. "
            f"Run `charis-calibrations fetch {instrument}` or set CHARIS_CALIBRATION_DIR.")
    return fetch(instrument, root).parent


def fetch(instrument, root):
    """Download and unpack the calibration data of one instrument into ``root/<version>/``.

    Parameters
    ----------
    instrument : str
        ``'SPHERE'`` or ``'CHARIS'``.
    root : pathlib.Path
        The calibration root, see :func:`cache_root`.

    Returns
    -------
    pathlib.Path
        ``root/<version>/<instrument>``. If another process created it first, its tree is kept.
    """
    entry = REGISTRY[instrument]
    version_dir = pathlib.Path(root) / entry["version"]
    instrument_dir = version_dir / instrument
    version_dir.mkdir(parents=True, exist_ok=True)
    # The partial directory sits next to its destination, so the final rename stays on one
    # filesystem and is atomic.
    partial = pathlib.Path(tempfile.mkdtemp(prefix=f".{instrument}.partial-", dir=version_dir))
    try:
        url = f"https://zenodo.org/api/records/{entry['record_id']}/files/{entry['archive']}/content"
        archive = partial / entry["archive"]
        request = Request(url, headers={"User-Agent": f"charis/{charis.__version__}"})
        md5 = hashlib.md5()
        with urlopen(request, timeout=120) as response, open(archive, "wb") as fout:
            size = int(response.headers.get("Content-Length", 0))
            log.info(f"Downloading {instrument} calibration data {entry['version']} "
                     f"({size / 1e6:.0f} MB) from {url}")
            for chunk in iter(lambda: response.read(1 << 20), b""):
                md5.update(chunk)
                fout.write(chunk)
        if md5.hexdigest() != entry["md5"]:
            raise OSError(f"md5 mismatch for {url}: got {md5.hexdigest()}, expected {entry['md5']}")
        with tarfile.open(archive) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(partial, filter="data")
            else:
                tar.extractall(partial)
        try:
            os.replace(partial / instrument, instrument_dir)
        except OSError:
            if not instrument_dir.is_dir():
                raise
            log.info(f"Keeping the existing {instrument_dir}")
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return instrument_dir
