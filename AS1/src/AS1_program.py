# -*- coding: utf-8 -*-
"""
Assignment 1

Kajol Tanesh Shah
A20496724
Spring 2022 CS512 Computer Vision


"""

import cv2
import numpy as np
#import pandas as pd
import sys
from PIL import Image
import matplotlib.pyplot as plt
#from cy1 import rgbtogray3

global image
global filename
global key_c
key_c = 0

#
#cv2.imshow("friends", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#return image

#function to reload image
def reloadImage():
#    rimage = readImage()
    rimage = cv2.imread(filename)
    return rimage

#function to save image
def writeImage(image):
    cv2.imwrite("updated_img.png",image)

#function to convert rgb to gray using function
def rgbtogray1(image):
    gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gimage

#function to convert rgb to gray without function
def rgbtogray2(image):
#    red = image[:,:,0] 
#    green = image[:,:,1]
#    blue = image[:,:,2]
    grayimage = np.dot(image[...,:3],[0.299, 0.587, 0.114])
#    grayimage1 = 0.299 * red + 0.587 * green + 0.114 * blue
    return grayimage

#def rgbtogray3(image):
#    arr = np.array(image)
    #print(arr)
#    for i in range(len(arr)):
#        for j in range(len(arr[i])):
#            b = arr[i, j, 0]
#            g = arr[i, j, 1]
#            r = arr[i, j, 2]
#            grayimage = b * 0.114 + g * 0.587 + r * 0.299
#            arr[i, j] = grayimage
#    Image.fromarray(arr)
#    pixels = image.load()
#    x,y,z = image.shape
#    for y 
def cyclecolor(image):
    b, g, r = cv2.split(image)
    if key_c == 0:
        cv2.imshow("Blue", b)
        key_c = key_c + 1
    elif key_c == 1:
        cv2.imshow("Green", g)
        key_c = key_c + 1
    else:
        cv2.imshow("Red", r)
        key_c = 0
    
#def sliderHandler2(theta):
#    rows = image.shape[0]
#    cols = image.shape[1]
#    M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, 1)
#    dst = cv2.warpAffine(image, M, (cols, rows))
#    cv2.imshow(winName, dst)
    
def rotation1(image):
    def sliderHandler1(theta):
        rows = image.shape[0]
        cols = image.shape[1]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, 1)
        dst = cv2.warpAffine(image, M, (cols, rows))
        cv2.imshow(winName, dst)
    image = rgbtogray1(image)
    cv2.namedWindow("RotationWindow")
    winName = "RotationWindow"
#    cv2.imshow(winName, gimage)
    cv2.createTrackbar('r', "RotationWindow", 0, 360, sliderHandler1)   
    cv2.imshow("RotationWindow", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
def pdescription():
    print("'i': reload the image")
    print("'w': save the image")
    print("'g': convert the image to grayscale using opencv fn")
    print("'G': convert the image to grayscale using user-defined fn")    
    print("'T': convert the image to grayscale using loops")
    print("'c': cycle through the color channels of the image")
    print("'r': convert image to grayscale and rotate it using a trackbar")
    print("'h': display a short description of the program and the keys")

 
   
#image1 = reloadImage()
#gimage = rgbtogray1(image)
#rotation1(image)
#cv2.imshow("grayscale", gimage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#pdescription()

def main():
    global image
    global filename
    global key_c
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) < 2:
#        filename = "C:/Users/Anshu Limbasiya/Downloads/fr.jpg"
        filename = "C:/Users/Anshu Limbasiya/Downloads/rgb01.png"
    image = cv2.imread(filename)
    print(image.shape)    
    key_c = 0
    print("Press 'h' to see program description and keys, 'e' for exit")
    key = input()
    while(key != 'e'):
#        print("Enter a key to process image")
 #       key = cv2.waitKey(10)
#        print(key)
#        if key == ord('i'):
        if key == 'i':
            print("in key i")
            image = reloadImage()
        elif key == 'w':
            writeImage(image)
        elif key == 'g':
            rimage = rgbtogray1(image)
            cv2.imshow("Grayscale",rimage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif key == 'G':
            gimage = reloadImage()
#           grayi = lambda image : np.dot(image[... , :3] , [0.299 , 0.587, 0.114]) 
#           grayi = grayi(image)
            grayimage = np.dot(gimage[...,:3],[0.299, 0.587, 0.114])
#            rimage2 = np.dot(image[...,:3],[0.299, 0.587, 0.114])
#            rimage2 = rgbtogray2(image)
#            cv2.imshow("Grayscale2",grayimage)
            plt.imshow(grayimage, cmap=plt.get_cmap("gray"))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif key == 'T':
            arr = np.array(image)
            #print(arr)
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    b = arr[i, j, 0]
                    g = arr[i, j, 1]
                    r = arr[i, j, 2]
                    grayimage = b * 0.114 + g * 0.587 + r * 0.299
                    arr[i, j] = grayimage
            Image.fromarray(arr).show()
#            arr = cy1.rgbtogray3(arr)
#            rgbtogray3(image)
        elif key == 'c':
            b, g, r = cv2.split(image)
            if key_c == 0:
                cv2.imshow("Blue", b)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                key_c = key_c + 1
            elif key_c == 1:
                cv2.imshow("Green", g)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
#                key_c = key_c + 1
                key_c = key_c + 1
            else:
                cv2.imshow("Red", r)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
#                key_c = key_c + 1
                key_c = 0
#            cyclecolor(image)
#           colorcycle(image)
        elif key == 'r':
            rotation1(image)
        elif key == 'h':
            pdescription()
#        elif key == 't':
#            arr = np.array(image)
#            arr = rgbtogray3(arr)
#            Image.fromarray(arr).show()
        else: 
            if key == 27 :
                cv2.destroyAllWindows()
            break
        key = input()
    
#    cv2.imshow("modififed",image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#if __name__ ==' __main__':
main()