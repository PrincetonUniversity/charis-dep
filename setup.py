#!/usr/bin/env python
"""Build the Cython extensions, with OpenMP where available.

Build logic only; package metadata lives in pyproject.toml.

Environment variables:
    CHARIS_NO_OPENMP=1       build serially on purpose (no OpenMP lookup).
    CHARIS_REQUIRE_OPENMP=1  fail the build if OpenMP is not found or does not compile,
                             instead of falling back to a serial build.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from Cython.Distutils import build_ext as cython_build_ext
from setuptools import Extension, setup
from setuptools.errors import CompileError, LinkError

README_SECTION = 'README.md, section "OpenMP"'


def env_flag(name):
    return os.environ.get(name, "").strip() == "1"


class OpenMPNotFound(RuntimeError):
    pass


def macos_libomp_roots():
    """Candidate libomp prefixes on macOS, in lookup order."""
    # The building interpreter's prefix covers pixi and conda envs. Under an isolated
    # build (pip, uv) sys.prefix can be a temporary build env, so also try the base.
    roots = [Path(sys.prefix), Path(sys.base_prefix)]
    brew = shutil.which("brew")
    if brew:
        try:
            result = subprocess.run([brew, "--prefix", "libomp"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            roots.append(Path(result.stdout.strip()))
    roots += [Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")]
    return list(dict.fromkeys(roots))


def openmp_flags():
    """Return (compile_args, link_args) for OpenMP, or raise OpenMPNotFound."""
    system = platform.system()
    if system == "Darwin":
        roots = macos_libomp_roots()
        for root in roots:
            inc, lib = root / "include", root / "lib"
            if (inc / "omp.h").is_file() and (lib / "libomp.dylib").is_file():
                return (
                    ["-Xpreprocessor", "-fopenmp", f"-I{inc}"],
                    [f"-L{lib}", "-lomp", f"-Wl,-rpath,{lib}"],
                )
        searched = "\n".join(f"      {root}/include/omp.h + {root}/lib/libomp.dylib" for root in roots)
        raise OpenMPNotFound(f"libomp was not found. Searched:\n{searched}")
    if system == "Linux":
        return ["-fopenmp"], ["-fopenmp"]
    raise OpenMPNotFound(f"no OpenMP configuration is known for platform {system!r}")


def warn_serial(cause):
    bar = "=" * 78
    message = (
        f"\n{bar}\n"
        "WARNING: charis is being built WITHOUT OpenMP.\n"
        "The extensions will run single-threaded, whatever maxproc is set to.\n"
        f"Cause: {cause}\n"
        f"How to enable OpenMP: see {README_SECTION}.\n"
        "Set CHARIS_REQUIRE_OPENMP=1 to make a missing OpenMP an error instead.\n"
        f"{bar}\n"
    )
    print(message, file=sys.stderr, flush=True)


class build_ext(cython_build_ext):
    """Build with OpenMP, falling back to a serial build unless OpenMP is required."""

    def build_extensions(self):
        # Always recompile and relink. setuptools otherwise skips an extension whose binary is
        # newer than its sources, which would keep a binary built with other OpenMP settings,
        # or (for the shared in-tree editable build) linked against another environment's libomp.
        self.force = True
        require = env_flag("CHARIS_REQUIRE_OPENMP")
        if env_flag("CHARIS_NO_OPENMP"):
            if require:
                raise RuntimeError("CHARIS_NO_OPENMP=1 and CHARIS_REQUIRE_OPENMP=1 are mutually exclusive.")
            warn_serial("CHARIS_NO_OPENMP=1 is set (serial build requested).")
            super().build_extensions()
            return

        try:
            compile_args, link_args = openmp_flags()
        except OpenMPNotFound as exc:
            if require:
                raise RuntimeError(f"CHARIS_REQUIRE_OPENMP=1 but OpenMP is unavailable: {exc}") from exc
            warn_serial(exc)
            super().build_extensions()
            return

        original = [(list(ext.extra_compile_args or []), list(ext.extra_link_args or [])) for ext in self.extensions]
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_args
            ext.extra_link_args = list(ext.extra_link_args or []) + link_args
        try:
            super().build_extensions()
            return
        except (CompileError, LinkError) as exc:
            if require:
                raise
            cause = f"the OpenMP build failed ({type(exc).__name__}: {exc})"

        for ext, (ext_compile_args, ext_link_args) in zip(self.extensions, original):
            ext.extra_compile_args, ext.extra_link_args = ext_compile_args, ext_link_args
        warn_serial(cause)
        super().build_extensions()


setup(
    ext_modules=[
        Extension("charis.primitives.matutils", ["charis/primitives/matutils.pyx"]),
        Extension("charis.utr.fitramp", ["charis/utr/fitramp.pyx"]),
    ],
    cmdclass={"build_ext": build_ext},
)
