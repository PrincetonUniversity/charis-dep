from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("charis")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"

from . import buildcalibrations, extractcube, image, instruments, parallel, primitives, tools, utr
from .primitives.matutils import openmp_enabled

__all__ = [
    'instruments',
    'buildcalibrations',
    'extractcube',
    'image',
    'parallel',
    'primitives',
    'tools',
    'utr',
    'openmp_enabled',
]