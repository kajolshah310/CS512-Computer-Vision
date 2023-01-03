#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:22:19 2022

@author: kajol
"""

import numpy as np2
cimport numpy as np

#checking accumulator values with threshold\(
def accu_check(int dbins, int tbins, np.ndarray accum, int thresh):
    cdef list lines = []
    cdef int y, x
    for y in range(0, dbins): #d
        for x in range(0, tbins): #theta
            if accum[y, x] > thresh:
                lines.append((y, x, accum[y, x]))
    return lines

def peaks_lines(int dbins, int tbins, np.ndarray accum, int thresh, np.ndarray dpeaks):
    cdef list lines2 = []    
    for y in range(0, dbins): #d
        for x in range(0, tbins): #theta
            if dpeaks[y, x] and accum[y, x] > thresh:
                lines2.append((y, x, accum[y,x]))
    return lines2
            
def final_lines(int dbins, int tbins, np.ndarray accum, list lines2):
    cdef list final_lines = []
    cdef int d_nbh, t_nbh, k, l
    for line in lines2:
        if accum[line[0], line[1]] == line[2]:
#            if tbins - line[1] <= 4
             final_lines.append(line)
             d_nbh = 10
             t_nbh = 10
             for k in range(-d_nbh, d_nbh):
                 for l in range(-t_nbh, t_nbh):
                     if line[1] + l >= tbins:
                         try:
                             accum[dbins/2 - (line[0] + k), (line[1] + l) % tbins] = 0
                         except:
                             pass
                     else:
                         try:
                             accum[line[0] + k, line[1] + l] = 0
                         except:
                             pass
                         
    return final_lines