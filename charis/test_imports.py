"""Every charis module imports with only the declared dependencies installed."""

import importlib
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
