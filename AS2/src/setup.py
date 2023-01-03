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
      name = 'My smoothing  function',
      ext_modules = cythonize("cy2.pyx"),
      include_dirs=[numpy.get_include()],
      )