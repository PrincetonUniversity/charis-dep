import os
import pathlib

import pytest

import charis
from charis.calibration_data import REGISTRY


def _checkout_calibrations():
    # Only found when the package is imported from a git checkout (editable install).
    for parent in pathlib.Path(charis.__file__).resolve().parents:
        if (parent / ".git").exists():
            calibrations = parent / "calibrations"
            return calibrations if calibrations.is_dir() else None
    return None


@pytest.fixture
def calibrations(tmp_path, monkeypatch):
    """The calibration root (``CHARIS_CALIBRATION_DIR``) holding the real calibration data.

    Uses an existing ``CHARIS_CALIBRATION_DIR``, or else links the checkout's ``calibrations/``
    into a temporary one, so no test ever downloads.
    """
    if os.environ.get("CHARIS_CALIBRATION_DIR"):
        return pathlib.Path(os.environ["CHARIS_CALIBRATION_DIR"])
    source = _checkout_calibrations()
    if source is None:
        pytest.skip("no calibration data: set CHARIS_CALIBRATION_DIR or run from a git checkout")
    for instrument, entry in REGISTRY.items():
        version_dir = tmp_path / entry["version"]
        version_dir.mkdir(exist_ok=True)
        (version_dir / instrument).symlink_to(source / instrument)
    monkeypatch.setenv("CHARIS_CALIBRATION_DIR", str(tmp_path))
    return tmp_path
