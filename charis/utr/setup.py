from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

ext_module = Extension("fitramp",
                       ['fitramp.pyx'],
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'],
                       )

setup(
    name='fitramp',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
)
