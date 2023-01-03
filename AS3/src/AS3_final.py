#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 05:41:56 2022
Assignement 3

Kajol Tanesh Shah
A20496724
Spring 2022 CS512 Computer Vision

@author: kajol
"""

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def rgbtogray(image):
    #converting to grayscale
    gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gimage

def edge_detection(image):
    #smoothing the image
    smooth_image = cv2.GaussianBlur(image, (3,3), 0)
    #edge detection along both x and y axis
    edge_detect = cv2.Sobel(smooth_image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    return edge_detect

def canny_edge(image):
#    smooth_image = cv2.GaussianBlur(image, (3,3), 0)
    canny_detect = cv2.Canny(image, 80, 300)
    return canny_detect

def peak_detection(image):
    #neighbourhood
    nbh = generate_binary_structure(2, 2)
    
    #local max filter applied
    l_max = maximum_filter(image, footprint=nbh)==image  
    #background masking to get peaks
    bg = (image==0)
    e_bg = binary_erosion(bg, structure=nbh, border_value=1)
    #xor operation to get peaks
    peaks = l_max ^ e_bg
    
    return peaks
    
def hough_transform(image, hough_plot):
    print("image shape ", image.shape[:2])
    
    hplot4 = hough_plot.add_subplot(1, 5, 4)
    hplot4.set_facecolor((0, 0, 0))
    #calculate width and height of image
    rows, cols = image.shape
    #calculate length of image diagonal for distance d using pythagoras theorem r^2 = x^2 + y^2
    img_diag= int(math.ceil(math.sqrt(rows**2 + cols**2)))
    #distance d and theta t bins
    dbins = int(2 * img_diag)
    tbins = int(180.0) + 1
    #quantize distance d
    ds = np.linspace(-img_diag, img_diag, 2 * img_diag)
    #quantize angle theta
    thetas = np.arange(0, 180, 1)
    #converting degree to radian
#    theta = np.linspace(0, 180, 1)
    
    #initialize accumulator - 2D array representing Hough parametric space with dimensions as d and theta
    accum = np.zeros((dbins, tbins), dtype=int)
    #find theta and d for edge pixels and increment the accumulator    
    for i in range(0, cols):
        for j in range(0, rows):
            #check if image point is an edge point
            if image[j, i] != 0:
                hsx = []
                hsy = []
#                for tind in range(theta_num):
                for tind in np.arange(0.0, 180.0, 1):
                    theta = math.radians(tind)
                    d = i * math.cos(theta) + j * math.sin(theta)
                    #
                    tbin = tind
                    dbin = d + dbins / 2
                    accum[int(dbin), int(tbin)] +=1
                    tind = int(tind)
                    hsx.append(tind)
                    hsy.append(d)

                hplot4.plot(hsx, hsy, color = "white", alpha=0.01)
    display_line(image, accum, thetas, ds)
                    
                 
#    plt.show()
    return accum, hplot4
                

def hough_trans_peaks(accum, thresh, maxlines):    
    dbins, tbins = accum.shape
    lines = []
    
    def to_d(dbin):
        return (dbin - (dbins / 2))
    
    def to_t(tbin):
        return (tbin)
    
    #checking accumulator values with threshold
    for y in range(0, dbins): #d
        for x in range(0, tbins): #theta
            if accum[y, x] > thresh:
                lines.append((y, x, accum[y, x]))
                
    dpeaks = peak_detection(accum)
    lines2 = []    
    for y in range(0, dbins): #d
        for x in range(0, tbins): #theta
            if dpeaks[y, x] and accum[y, x] > thresh:
                lines2.append((y, x, accum[y,x]))
                
    lines.sort(key=lambda x: x[2], reverse = True)
    lines2.sort(key=lambda x: x[2], reverse = True)
    
    final_lines=[]
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
                         
    k_lines = list(map(lambda x: (to_d(x[0]), to_t(x[1])), final_lines))
    if len(k_lines) > maxlines:
        k_lines = k_lines[0:maxlines]
        
    return k_lines

def display_line(image, accum, thetas, ds):
    #extent=[horizontal_min,horizontal_max,vertical_min,vertical_max]
    plt.imshow(accum, cmap = 'gray', extent = [thetas[-1], thetas[0], ds[-1], ds[0]])
#    plt.show()

def draw_lines(image, lines, hplot1):
    r, c = image.shape
    for d, theta in lines:
        t_r = math.radians(theta)
        start = (0,0)
        end = (0,0)
        
        if theta ==0:
            start = (int(d), 0)
            end = (int(d), r)
        elif theta == 90:
            
            start = (0, int(d))
            end = (c, int(d))
        else:
            m = 0
            m = 1 / math.tan(t_r)
            b = (-d) / math.sin(t_r)
            start = (0, -int(b))
            end = (int(c), -int(m * c + b))
            
        thickness = int(r/250) + 1
        line_type = 8
        color = (255, 0, 0)
        image = cv2.line(image, start, end, color, thickness, line_type)
        hplot1.plot(theta, d, marker = 'o', color = "red")
    return image
            
def main():
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) < 2:
        filename = "/home/kajol/Desktop/b1.jpeg"
    
    image = cv2.imread(filename)
    image2 = image
    #plots
    hough_plot = plt.figure(figsize=(20,20))
    hplot1 = hough_plot.add_subplot(1, 5, 1)
    hplot1.imshow(image)
        
    #converting image to grayscale
    image = rgbtogray(image)
    print(image.shape)    
    hplot2 = hough_plot.add_subplot(1, 5, 2)
    hplot2.imshow(image)
    #canny edge detection
    image = canny_edge(image)
    hplot3 = hough_plot.add_subplot(1, 5, 3)
    hplot3.imshow(image)

    thresh = 40 #threshold
    num_lines = 5 #k number of lines to be detected
    accum, hplot4 = hough_transform(image, hough_plot)
    lines = hough_trans_peaks(accum, thresh, num_lines)
    img_lines = draw_lines(image, lines, hplot4)
    hplot5 = hough_plot.add_subplot(1, 5, 5)
    hplot5.imshow(img_lines)

    

main()