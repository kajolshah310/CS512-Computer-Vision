#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:24:25 2022

@author: kajol
"""

import numpy
import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
      name = 'My line detection  function',
      ext_modules = cythonize("cy3.pyx"),
      include_dirs=[numpy.get_include()],
      )
