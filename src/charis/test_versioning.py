"""The package version comes from the git tags (setuptools_scm), not a hard-coded string."""

import importlib.metadata
import pathlib
import shutil
import subprocess

import pytest

import charis


def test_version_matches_metadata():
    assert charis.__version__ == importlib.metadata.version("charis")
    assert charis.__version__ != "0.0.0"


def _checkout_root():
    # Only meaningful when the package is imported from a git checkout (editable install).
    for parent in pathlib.Path(charis.__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def test_version_is_derived_from_git_tags():
    root = _checkout_root()
    if root is None or shutil.which("git") is None:
        pytest.skip("not imported from a git checkout")
    on_tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"], cwd=root, capture_output=True).returncode == 0
    if on_tag:
        assert ".dev" not in charis.__version__
    else:
        assert ".dev" in charis.__version__, f"{charis.__version__} looks hard-coded: HEAD is past the last tag"
