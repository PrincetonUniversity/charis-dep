"""Run-time notice for builds without OpenMP.

pip and uv hide the build-time warning unless run with ``-v``, so a serial build would
otherwise only show up as slow extractions.
"""

from .matutils import openmp_enabled


def warn_if_serial(nthreads, log):
    """Log a warning if ``nthreads > 1`` was requested but the extensions are serial.

    Returns True if it warned.
    """
    if nthreads is None or nthreads <= 1 or openmp_enabled():
        return False
    log.warning(
        "charis was built without OpenMP, so the %d threads requested run serially. "
        "See 'OpenMP' in the README to rebuild with OpenMP; charis.openmp_enabled() reports the build.",
        nthreads)
    return True
