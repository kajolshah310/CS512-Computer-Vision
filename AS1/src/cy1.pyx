# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:37:13 2022

@author: Anshu Limbasiya
"""
cimport numpy as np

def rgbtogray3(np.ndarray arr):
    cdef int i, j, b, r, g
    cdef float new_value
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            b = arr[i, j, 0]
            g = arr[i, j, 1]
            r = arr[i, j, 2]
            grayimage = b * 0.114 + g * 0.587 + r * 0.299
            arr[i, j] = grayimage
    return arr