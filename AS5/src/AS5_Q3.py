#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 20:59:32 2022

@author: kajol
"""

import cv2
import numpy as np
import pandas as pd
import random
import math


def read_points(filename):
    points = pd.read_csv(filename, skipinitialspace= "True", delimiter=" ", header = None)
    return points

def config_file():
    cfilename = "/home/kajol/Desktop/RANSAC.config"
    with open(cfilename, 'r') as wc:
        prob = float(wc.readline().split()[0])
        kmax = int(wc.readline().split()[0])
        nmin = int(wc.readline().split()[0])
        nmax = int(wc.readline().split()[0])
        w = float(wc.readline().split()[0])
    return prob, nmin, nmax, kmax, w

def convto3dh(ps_3d):
    ps_3d = np.array(ps_3d)
    print("jjj", ps_3d.shape)
    ps_3dh = cv2.convertPointsToHomogeneous(ps_3d)
    shape0 = ps_3dh.shape[0]
    shape2 = ps_3dh.shape[2]
    ps_3dh = ps_3dh.reshape(shape0, shape2)
    return ps_3dh
    
def calculate_svd(pmatrix):
    U, D, V = np.linalg.svd(pmatrix)
    min_val = np.argmin(D)
    pmatrix_svd = V[min_val]
    print(pmatrix_svd.shape)
    return pmatrix_svd

def construct_pmatrix(ps_3d, ps_2d, no_of_points):
    
    pmatrix = []
    for j in range(no_of_points):
        xi, yi =  ps_2d.iloc[j]
        xw, yw, zw = ps_3d.iloc[j]
        irow = []
        jrow = []
        list3dp = [xw, yw, zw, 1]
        listzeros = [0, 0, 0 , 0]
        listpi = [-1 * xi *xw, -1 * xi * yw, -1 * xi * zw, -1 * xi ]
        listpj = [-1 * yi *xw, -1 * yi * yw, -1 * yi * zw, -1 * yi ]
        irow = list3dp + listzeros + listpi
        jrow = listzeros + list3dp + listpj
        pmatrix.append(irow)
        pmatrix.append(jrow)        
        # irow.append(xw)
        # irow.append(yw)
        # irow.append(zw)
        # irow.append(1)
        # for i in range(4):
        #     irow.append(0)
        #     jrow.append(0)
        # irow.append(xw)
        # irow.append(yw)
        # irow.append(zw)
        # irow.append(1)
    pmatrix = np.array(pmatrix)
    #print(pmatrix)
    #print(pmatrix.shape)
    return pmatrix

def construct_rmatrix(ps_3d, ps_2d, no_of_points):
    
    pmatrix = []
    for j in range(no_of_points):
        xi, yi =  ps_2d[j]
        xw, yw, zw = ps_3d[j]
        irow = []
        jrow = []
        list3dp = [xw, yw, zw, 1]
        listzeros = [0, 0, 0 , 0]
        listpi = [-1 * xi *xw, -1 * xi * yw, -1 * xi * zw, -1 * xi ]
        listpj = [-1 * yi *xw, -1 * yi * yw, -1 * yi * zw, -1 * yi ]
        irow = list3dp + listzeros + listpi
        jrow = listzeros + list3dp + listpj
        pmatrix.append(irow)
        pmatrix.append(jrow)        
        # irow.append(xw)
        # irow.append(yw)
        # irow.append(zw)
        # irow.append(1)
        # for i in range(4):
        #     irow.append(0)
        #     jrow.append(0)
        # irow.append(xw)
        # irow.append(yw)
        # irow.append(zw)
        # irow.append(1)
    pmatrix = np.array(pmatrix)
    #print(pmatrix)
    #print(pmatrix.shape)
    return pmatrix


def error_distance(pmatrix_svd, no_of_points, ps_3d, ps_2d):
    #print(pmatrix_svd[0])
    #print(pmatrix_svd[1])
    #print(pmatrix_svd[2])
    ps_3dh = convto3dh(ps_3d)
    

    m1 = pmatrix_svd[0]
    m2 = pmatrix_svd[1]
    m3 = pmatrix_svd[2]
    #tpe = 0
    dist = []
    for j in range(no_of_points):
        pxu = np.dot(m1, ps_3dh[j])
        pxl = np.dot(m3, ps_3dh[j])
        px = pxu/pxl
        pyu = np.dot(m2, ps_3dh[j])
        pyl = np.dot(m3, ps_3dh[j])
        py = pyu/pyl
        #print(type(ps_2d))
        #ps_2dl = ps_2d.values
        ipx = ps_2d.iloc[j][0]
        #ipx = ps_2d[j][0]
        #ipy = ps_2d[j][1]
        ipy = ps_2d.iloc[j][1]
        proj_er = (ipx - px) ** 2 + (ipy - py) ** 2
        sqrtpe = np.sqrt(proj_er)
        dist.append(sqrtpe)
    #mean square error
    #mse = tpe/no_of_points
    #print("Mean Square error: ", mse)
    return dist
    
def implement_ransac(dist, pmatrix_svd, no_of_points, ps_3d, ps_2d):
    prob, nmin, nmax, kmax, w = config_file()
    k = kmax
    n2d = random.randint(nmin, nmax)
    new_matrix = []
    md = np.median(dist)
    tt = md * 1.5
    max_inlrs = 0
    c = 0
    for i in range(kmax):
        if c < k and c < kmax:
            inlrs = []
            ind = np.random.choice(no_of_points, n2d)
            rs_3d, rs_2d = np.array(ps_3d)[ind], np.array(ps_2d)[ind]
            rno_of_points = len(rs_3d)
            
            rmatrix = construct_rmatrix(rs_3d, rs_2d, rno_of_points)
            rmatrix_svd = calculate_svd(rmatrix)
            rmatrix_svd = rmatrix_svd.reshape(3, 4)
            print(rmatrix_svd.shape)
            
            ran_dist = error_distance(rmatrix_svd, rno_of_points, rs_3d, rs_2d)
            for j, d in enumerate(ran_dist):
                if d < tt:
                    inlrs.append(d)
                if len(inlrs) > max_inlrs:
                    max_inlrs = len(inlrs)
                    in_3d, in_2d = np.array(ps_3d)[inlrs], np.array(ps_2d)[inlrs]
                    in_no_of_points = len(in_3d)
                    new_matrix = construct_pmatrix(in_3d, in_2d, in_no_of_points)
                if w != 0:
                    leninlrs = float(len(inlrs))
                    nps_2d = np.array(ps_2d)
                    lenips = float(len(nps_2d))
                    w = leninlrs/lenips
                    ku = float(math.log(1-prob))
                    kl = np.absolute(math.log(1-(w ** n2d)))
                    k = ku/kl
    return new_matrix, max_inlrs


def camera_parameters(pmatrix_svd):
    print("Camera parameters are:- ")
    b = pmatrix_svd[:3, 3]
    a1 = pmatrix_svd[0, :3]
    a2 = pmatrix_svd[1, :3]
    a3 = pmatrix_svd[2, :3]    
    a3_inv = 1/(np.sqrt(np.sum(np.square(a3))))
    if b[2] >= 0:
        signbz = 1
    else:
        signbz = -1
    p = signbz * a3_inv
    print("Rho: ",p)
    
    a1dota3 = np.dot(a1, a3)
    a2dota3 = np.dot(a2, a3)
    a2dota2 = np.dot(a2, a2)
    a1dota1 = np.dot(a1, a1)
    p2 = p ** 2
    u0 = p2 * a1dota3
    print("Intrinsic parameters: ")
    print("u0: ", u0)
    v0 = p2 * a2dota3
    print("v0: ", v0)
    a1crossa3 = np.cross(a1, a3)
    a2crossa3 = np.cross(a2, a3)
    av = np.sqrt((p2 * a2dota2)-(v0 ** 2))
    print("Alpha v: ", av)
    p4 = p ** 4
    s = (1/av) * p4 * np.dot(a1crossa3, a2crossa3)
    print("s: ", s)
    s2 = s ** 2
    u02 = u0 ** 2
    au = np.sqrt((p2 * a1dota1) - s2 - u02)
    print("Alpha u: ", au)
    K1 = []
    K2 = []
    K3 = []
    K1.append(au)
    K1.append(s)
    K1.append(u0)
    K2.append(0)
    K2.append(av)
    K2.append(v0)
    K3.append(0)
    K3.append(0)
    K3.append(1)
    K_matrix = []
    K_matrix.append(K1)
    K_matrix.append(K2)
    K_matrix.append(K3)
    K_matrix = np.array(K_matrix)
    print("K*: ",K_matrix)
    #print(type(K_matrix))
    K_inv = np.linalg.inv(K_matrix)
    e = signbz 
    print("Epsilon: ", e)
    Kmulb = np.matmul(K_inv, b)
    print("Extrinsic parameters: ")
    T_matrix = e * p * Kmulb
    
    
    print("T*:", T_matrix)
    r3 = e * p * a3
    print("r3: ", r3)
    r1 = (1/av) * p2 * a2crossa3
    print("r1: ", r1)
    r2 = np.cross(r3, r1)
    print("r2: ", r2)
    r1T = np.transpose(r1)
    r2T = np.transpose(r2)
    r3T = np.transpose(r3)
    
    R_matrix = []
    R_matrix.append(r1T)
    R_matrix.append(r2T)
    R_matrix.append(r3T)
    R_matrix = np.array(R_matrix)
    print("R*: ", R_matrix)
    

    
filename_3d = "/home/kajol/Desktop/given_points/ncc-worldPt.txt"
filename_2d = "/home/kajol/Desktop/given_points/ncc-imagePt.txt"
ps_3d = read_points(filename_3d)
#ps_3dh = convto3dh(ps_3d)
#shape0 = ps_3dh.shape[0]
#shape2 = ps_3dh.shape[2]
#ps_3dh = ps_3dh.reshape(shape0, shape2)

ps_2d = read_points(filename_2d)
no_of_points = len(ps_3d)
pmatrix = construct_pmatrix(ps_3d, ps_2d, no_of_points)
pmatrix_svd = calculate_svd(pmatrix)
pmatrix_svd = pmatrix_svd.reshape(3, 4)
print(pmatrix_svd.shape)
dist = error_distance(pmatrix_svd, no_of_points, ps_3d, ps_2d)
new_matrix, max_inlrs = implement_ransac(dist, pmatrix_svd, no_of_points, ps_3d, ps_2d)
new_matrix_svd = calculate_svd(new_matrix)
new_matrix_svd = new_matrix_svd.reshape(3, 4)
camera_parameters(new_matrix_svd)