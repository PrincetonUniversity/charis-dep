"""The Cython extensions are built with OpenMP, and the parallel solver is correct."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import charis
from charis.primitives import matutils

HOMEBREW_PREFIXES = (Path("/opt/homebrew"), Path("/usr/local"))


@pytest.mark.skipif(
    os.environ.get("CHARIS_REQUIRE_OPENMP") != "1",
    reason="only asserted when the build was required to use OpenMP (CHARIS_REQUIRE_OPENMP=1)",
)
def test_openmp_is_enabled():
    assert charis.openmp_enabled()


def _lstsq_problem(ncut=2000, npix=80, ncoef=20, seed=12345):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(ncut, npix, ncoef))
    b = rng.normal(size=(ncut, npix))
    indx = np.arange(ncut, dtype=np.int64)
    size = np.full(ncut, npix, dtype=np.int64)
    return A, b, indx, size


def test_parallel_lstsq_matches_serial():
    A, b, indx, size = _lstsq_problem()
    ncut = A.shape[0]
    # lstsq modifies A and b in place, so every call gets fresh copies.
    # maxproc is passed explicitly: the parallel region overrides OMP_NUM_THREADS.
    serial = matutils.lstsq(A.copy(), b.copy(), indx, size, ncut, returncov=0, maxproc=1)
    parallel = matutils.lstsq(A.copy(), b.copy(), indx, size, ncut, returncov=0, maxproc=4)

    assert np.array_equal(serial, parallel)

    reference = np.array([np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(50)])
    np.testing.assert_allclose(serial[:50], reference, rtol=0, atol=1e-12)


def _loaded_libomps():
    libc = ctypes.CDLL(None)
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    names = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
    return [name for name in names if Path(name).name.startswith("libomp")]


# Same listing in a fresh interpreter that dlopens only the extension file, so the file's
# own rpath decides which libomp dyld finds. In the test process numpy or scipy may already
# have loaded a libomp that dyld reuses for the extension, which would hide a stale rpath.
_ISOLATED_LOAD = """
import ctypes, sys
ctypes.CDLL(sys.argv[1])
libc = ctypes.CDLL(None)
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
libc._dyld_get_image_name.restype = ctypes.c_char_p
for i in range(libc._dyld_image_count()):
    name = libc._dyld_get_image_name(i).decode()
    if name.rsplit("/", 1)[-1].startswith("libomp"):
        print(name)
"""


def _isolated_libomps(extension):
    result = subprocess.run(
        [sys.executable, "-I", "-c", _ISOLATED_LOAD, extension], capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def _assert_from_this_environment(libomps, where):
    # setup.py also searches sys.base_prefix: a venv made from a conda or pixi Python links
    # the base environment's libomp.
    prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    for path in libomps:
        lib = Path(path).resolve()
        in_homebrew = ".pixi" not in lib.parts and any(lib.is_relative_to(root) for root in HOMEBREW_PREFIXES)
        assert any(lib.is_relative_to(prefix) for prefix in prefixes) or in_homebrew, (
            f"{where}: libomp {lib} is neither under sys.prefix or sys.base_prefix ({sorted(map(str, prefixes))}) "
            "nor in a Homebrew prefix"
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="dyld image list is macOS only")
def test_openmp_runtime_belongs_to_this_environment():
    in_process = _loaded_libomps()
    _assert_from_this_environment(in_process, "test process")
    assert len(set(map(os.path.realpath, in_process))) <= 1, f"two OpenMP runtimes loaded: {in_process}"

    _assert_from_this_environment(_isolated_libomps(matutils.__file__), f"{Path(matutils.__file__).name} alone")
