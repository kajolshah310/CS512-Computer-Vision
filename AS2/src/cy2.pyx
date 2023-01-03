#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:44:31 2022

@author: kajol
"""
import numpy as np2
cimport numpy as np
def cysmooth(np.ndarray imageS, np.ndarray kernelS, int n):
    cdef np.ndarray z
#    cdef np.float64 l
    cdef int i, j, s0, s1, f0, f1, r, c, l
    s0 = imageS.shape[0]
    s1 = imageS.shape[1]
    f0 = kernelS.shape[0]
    f1 = kernelS.shape[1]
    r = s0+f0-1
    c = s1+f1-1
    z = np2.zeros((r, c))
    for i in range(s0):
        for j in range(s1):
            z[i+np2.int((f0-1)/2), j+np2.int((f1-1)/2)] = imageS[i,j]
    
    for i in range(s0):
        for j in range(s1):
            k = z[i:i+f0, j:j+f1]
            l = np2.int(np2.sum(k*kernelS))
            imageS[i,j] = l
    return imageS
