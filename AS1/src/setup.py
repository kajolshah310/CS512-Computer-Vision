# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:44:56 2022

@author: Anshu Limbasiya
"""
import numpy
import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
      name = 'My RGB Loop',
      ext_modules = cythonize("cy1.pyx"),
      include_dirs=[numpy.get_include()],
      )