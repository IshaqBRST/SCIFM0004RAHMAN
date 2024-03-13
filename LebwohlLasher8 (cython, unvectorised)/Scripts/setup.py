

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("script.pyx"),
    include_dirs=[numpy.get_include()]  # This is required for compiling with NumPy
)
