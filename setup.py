#!/usr/bin/env python
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Distutils import build_ext as cython_build_ext

def find_libomp():
    """
    Searches common Homebrew lib paths for omp.h and libomp.dylib.
    Returns (include_dir, lib_dir) or (None, None) if not found.
    """
    homebrew = Path("/opt/homebrew")
    for header in homebrew.rglob("omp.h"):
        inc = header.parent
        lib = None
        for dylib in homebrew.rglob("libomp.dylib"):
            lib = dylib.parent
            break
        if lib:
            return str(inc), str(lib)
    return None, None

is_macos = platform.system() == "Darwin"
omp_include, omp_lib = find_libomp()

def get_extensions(with_openmp=True):
    if is_macos and with_openmp and omp_include and omp_lib:
        compile_args = ["-Xpreprocessor", "-fopenmp", f"-I{omp_include}"]
        link_args = ["-Xpreprocessor", "-fopenmp", f"-L{omp_lib}"]
        libs = ["omp"]
    else:
        compile_args = []
        link_args = []
        libs = []

    return [
        Extension(
            "charis.primitives.matutils",
            ["charis/primitives/matutils.pyx"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            libraries=libs,
        ),
        Extension(
            "charis.utr.fitramp",
            ["charis/utr/fitramp.pyx"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            libraries=libs,
        ),
    ]

try:
    setup(
        ext_modules=get_extensions(with_openmp=True),
        cmdclass={"build_ext": cython_build_ext},
    )
except Exception as e:
    print("⚠️ OpenMP build failed or not supported; falling back without OpenMP.")
    print(f"  Reason: {e}")
    setup(
        ext_modules=get_extensions(with_openmp=False),
        cmdclass={"build_ext": cython_build_ext},
    )
