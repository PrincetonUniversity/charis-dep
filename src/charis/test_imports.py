"""Every charis module imports with only the declared dependencies installed."""

import importlib
import importlib.metadata
import os
import pkgutil

import pytest

import charis

# The per-directory setup.py files are legacy standalone build scripts (they import
# distutils and call setup() at import time); they are not part of the package API.
MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(charis.__path__, "charis.")
    if not m.name.rsplit(".", 1)[-1].startswith("test_") and m.name.rsplit(".", 1)[-1] != "setup"
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    # Most modules are imported by some other test, but a module nothing else
    # imports can depend on a package that only an optional extra provides.
    importlib.import_module(name)


def test_compiled_extensions_load():
    from charis.primitives import matutils
    from charis.utr import fitramp

    assert callable(matutils.lstsq)
    assert callable(fitramp.fit_ramp)


def test_calibrations_resolve_from_an_installed_package():
    from charis.instruments import SPHERE

    mode_dir = SPHERE("YH").calibration_path_mode
    assert os.path.isfile(os.path.join(mode_dir, "lamsol.dat"))


@pytest.mark.parametrize("script", ["buildcal", "extractcube", "hexplot"])
def test_console_scripts_point_into_the_package(script):
    # A top-level "scripts" package would be installed into users' site-packages.
    (entry_point,) = [ep for ep in importlib.metadata.entry_points(group="console_scripts") if ep.name == script]
    assert entry_point.value == f"charis.scripts.{script}:main"
    assert callable(entry_point.load())
