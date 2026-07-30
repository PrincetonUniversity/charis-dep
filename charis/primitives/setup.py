from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

ext_module = Extension("matutils",
                       ['matutils.pyx'],
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'],
                       )

setup(
    name='matutils',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
)
