#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:19:58 2022
Assignment 2

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
import cy2

    
def rgbtogray1(image):
    gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gimage

#function to calculate x-derivative
def xderivative(sm3):
    xder = cv2.Sobel(sm3, cv2.CV_64F, 1, 0, ksize = 5)
    cv2.normalize(xder, xder, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    print(xder)
    return xder

#function to calculate y-derivative
def yderivative(sm3):
    yder = cv2.Sobel(sm3, cv2.CV_64F, 0, 1, ksize = 5)
    cv2.normalize(yder, yder, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return yder

def pdescription():
    print("'s': smooth the image using opencv")
    print("'S': smooth the image using your own implementation in cython")
    print("'D': Downsample the image")
    print("'U': Upsample the image")    
    print("'x': x-derivative of the image")
    print("'m': magnitude of the image")
    print("'p': image gradient vectors")
    print("'c': detection of corners using cornerharris")
    print("'C': detection of corners without using cornerharris")
    print("'h': display a short description of the program and the keys")

def main():
    global image
    global image2
    global imagec
    global filename
    global imageC
    global key_c
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) < 2:
        filename = "/home/kajol/Desktop/hp.jpeg"
    image = cv2.imread(filename)
    image2 = image
    image = rgbtogray1(image)
    print(image.shape)    
    key_c = 0
    print("Press 'h' to see program description and keys, 'e' for exit")
    key = input()
    while(key != 'e'):
        if key == 's':
#smoothing using filter2D
            def sliderHandler1(n):   
                imagen = cv2.imread(filename)
                images = rgbtogray1(imagen)
                kernel = np.ones((n,n),np.float32)/(n*n)    #make a smoothing filter
                sm = cv2.filter2D(images, -1, kernel)    #convolve
                cv2.imshow(winName, sm)
            winName = 'Smoothing'
            cv2.imshow(winName, image)
            cv2.createTrackbar('s',winName, 0, 255, sliderHandler1) #slider bar [0, 255]
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#smoothing using own logic and cython  

        elif key == 'S':
            def sliderHandlerS(n):
            
                image2 = cv2.imread(filename)
                imageS = rgbtogray1(image2)
                kernelS = np.ones((n,n),np.float32)/(n*n)
                conv_image = cy2.cysmooth(imageS, kernelS, n)
                cv2.imshow(winNameS, conv_image)
            winNameS = 'Smoothing using cython'
#            print(type(image))
            cv2.imshow(winNameS, image)
            cv2.createTrackbar('s',winNameS, 0, 255, sliderHandlerS) #slider bar [0, 255]
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                        
        elif key == 'D':
#downsampling image by a factor of 2
            n=5
            kernel = np.ones((n,n),np.float32)/(n*n)    #make a smoothing filter
            imageD = cv2.filter2D(image, -1, kernel)  
            print("Size of image before downsampling is :", image.shape)
            w = int(image.shape[1]/2)
            h = int(image.shape[0]/2)
            dims = (w, h)
            imageD = cv2.resize(imageD, dims)
#            imageD = cv2.pyrDown(image)
            print("Size of image after downsampling is :", imageD.shape)
            cv2.imshow('downsample',imageD)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
#upsampling image by a factor of 2
        elif key == 'U':
            n=5
            print("Size of image before upsampling is :", image.shape)
            w2 = int(image.shape[1]*2)
            h2 = int(image.shape[0]*2)
            dims2 = (w2, h2)
            imageU = cv2.resize(image, dims2)
            kernel = np.ones((n,n),np.float32)/(n*n)    #make a smoothing filter
            imageU = cv2.filter2D(imageU, -1, kernel)  
#            imageD = cv2.pyrDown(image)
            print("Size of image after upsampling is :", imageU.shape)
            cv2.imshow('upsample',imageU)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
#computing x-derivative
        elif key == 'x':
#            n=5
#            kernel = np.ones((n,n),np.float32)/(n*n)    #make a smoothing filter
#            sm3 = cv2.filter2D(image, -1, kernel)  
            xder = xderivative(image)
            cv2.imshow('xderivative', xder)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#computing magnitude of image gradient
        elif key == 'm':
            n=5
            image3 = cv2.imread(filename)
            imagem = rgbtogray1(image2)  
#            kernel = np.ones((n,n),np.float32)/(n*n)    #make a smoothing filter
#            sm3 = cv2.filter2D(imagem, -1, kernel)  
            xder = xderivative(imagem)
            yder = yderivative(imagem)
            grad_mag = np.sqrt((xder ** 2 ) + (yder ** 2))
            grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
            (figure, axis) = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
            axis.imshow(grad_mag, cmap="jet")
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            plt.tight_layout()
            plt.show()
            print("magnitude", grad_mag)
#plot image gradient vectors using line segments
        elif key == 'p':
            def sliderHandlerp(N):
                image2 = cv2.imread(filename)
                imagep = rgbtogray1(image2)   
                n=5
#                kernel = np.ones((n,n),np.float32)/(n*n)    #make a smoothing filter
#                sm3 = cv2.filter2D(image, -1, kernel)  
                xder = xderivative(imagep)
                yder = yderivative(imagep)
                for i in range(0, imagep.shape[0], N):
                    for j in range(0, imagep.shape[1], N):
                        theta = math.atan2(yder[i, j], xder[i, j])
                        i_grad = int(i + N * math.cos(theta))
                        j_grad = int(j + N * math.sin(theta))
                        for k in range(0,5):
                            im = k+1
                        cv2.arrowedLine(imagep, (j,i), (j_grad, i_grad), (0,0,255))
                cv2.imshow(winNamep, imagep)
            winNamep = 'gradient vectors'
            cv2.imshow(winNamep, image)
            cv2.createTrackbar('c',winNamep, 0, 100, sliderHandlerp) #slider bar [0, 255]
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#detect corners using cornerharris
        elif key == 'c':
            def sliderHandler1(n2):    
                global imagec
                n2 = n2*0.01
                imagec = np.float32(image3)    #make a smoothing filter
                dst = cv2.cornerHarris(imagec, 2, 3, n2) 
                dst = cv2.dilate(dst, None)
                imagec=cv2.merge((image3, image3, image3))
                threshold = 0.01*dst.max()
                imagec[dst>threshold]=[0,0,255]
                cv2.imshow(winName2,imagec)
            
            winName2 = 'corner detection'
            image3 = rgbtogray1(image2)            
            cv2.imshow(winName2, image3)
            cv2.createTrackbar('c',winName2, 0, 100, sliderHandler1) #slider bar [0, 255]
            cv2.waitKey(0)
            cv2.destroyAllWindows()

#implementing corner detection without using cornerharris            
        elif key == 'C':
            def sliderHandlerC(nC):
                global imagef
#                global imageg
                image2 = cv2.imread(filename)
                imageg = rgbtogray1(image2)
                imagef = np.float32(imageg)
                corn = cv2.goodFeaturesToTrack(imagef, nC, 0.01, 10)
                corn = np.int0(corn)
                imagef=cv2.merge((imageg, imageg, imageg))
                for c in corn:
                    x, y = c.ravel()
                    cv2.circle(imagef, (x, y), 3, (0,0,255), -1)
                cv2.imshow(winName3, imagef)
            winName3 = 'Corner detection'
            cv2.imshow(winName3, image)
            cv2.namedWindow(winName3, cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar('corner points',winName3, 0, 255, sliderHandlerC) #slider bar [0, 255]
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif key == 'h':
            pdescription()
        else:
            if key == 27 :
                cv2.destroyAllWindows()
            break
        key = input()

main()

