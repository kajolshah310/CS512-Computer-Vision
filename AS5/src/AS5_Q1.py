#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 05:46:46 2022

@author: kajol
"""

import cv2
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readimage(filename):
    image = cv2.imread(filename)
    return image

def rgbtogray(image):
    gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gimage

def findcorners(image):
    f_corns, corners = cv2.findChessboardCorners(image, (7, 7), None)
    return f_corns, corners

def drawcorners(image, gimage, f_corns, corners):
    ps_2d = []
    ps_3d = []
    p_obj = np.zeros((7 * 7, 3), np.float32)
    p_obj[:, :2]  = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    if f_corns == True:
        print(f_corns)
        ps_3d.append(p_obj)
        
        corn_coords = cv2.cornerSubPix(gimage, corners, (11, 11), (-1, -1), crit)
        
        ps_2d.append(corn_coords)
        
        cv2.drawChessboardCorners(image, (7, 7), corn_coords, f_corns)
        
    return ps_2d, ps_3d
        
def writepoints(ps_2d, ps_3d):
    with open("/home/kajol/Desktop/world_points.txt", "w") as f:
        for wp in ps_3d[0]:
            f.write(str(wp[0]) + "  " + str(wp[1]) + "  " + str(wp[2]) + "\n");
    f.close()  
    
    with open("/home/kajol/Desktop/image_points.txt", "w") as f:
        for ip in ps_2d[0]:
            f.write(str(ip[0][0]) + "  " + str(ip[0][1]) + "\n");
            #f.write(str(ip[0]) + "\t" + str(ip[1]) + "\n");
    f.close()


filename = "/home/kajol/Desktop/chess.jpg"
#filename = "/home/kajol/Desktop/chess.jpg"
image = readimage(filename)
image = cv2.resize(image,(700, 700))
gimage = rgbtogray(image)
found_corners, corners = findcorners(gimage)
ps_2d, ps_3d = drawcorners(image, gimage, found_corners, corners)
writepoints(ps_2d, ps_3d)
cv2.imshow('corners_image', image)
cv2.waitKey()
cv2.destroyAllWindows()

