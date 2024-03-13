from setuptools import setup
from Cython.Build import cythonize

setup(
    name='energy_functions',
    ext_modules=cythonize("all_energy.pyx"),
)
