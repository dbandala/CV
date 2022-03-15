#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def getNextSlope(ima,rad_max,coordPrev,coord):
    vect_prev = coord-coordPrev
    yc = coord[0]
    xc = coord[1]
    for rad in range(int(rad_max),1,-1):
        for i in range(yc-rad,yc+rad+1):
            if i<0 or i>=len(ima):
                continue
            j2 = round(xc-np.sqrt(rad**2-(yc-i)**2))
            coord2 = np.array([i,j2])
            if ima[i][j2]>0:
                vect = coord2-coord
                theta = np.arctan2(vect[0],vect[1])-np.arctan2(vect_prev[0],vect_prev[1])
                if theta>np.pi:
                    theta -= 2*np.pi
                m = theta/np.pi
                if np.abs(m)<1-1/rad:
                    return coord2,-m
        for i in range(yc+rad,yc-rad-1,-1):
            if i<0 or i>=len(ima):
                continue
            j1 = round(xc+np.sqrt(rad**2-(yc-i)**2))
            coord1 = np.array([i,j1])
            if ima[i][j1]>0:
                vect = coord1-coord
                theta = np.arctan2(vect[0],vect[1])-np.arctan2(vect_prev[0],vect_prev[1])
                if theta>np.pi:
                    theta -= 2*np.pi
                m = theta/np.pi
                if np.abs(m)<1-1/rad:
                    return coord1,m
    return None,None

def slopeChainCodeWithGraphics(ima,rad,segments):
    chain=[]
    points=[]
    for i in range(len(ima)):
        for j in range(len(ima[0])):
            if ima[i][j]>0:
                coord = np.array([i,j])
                points.append(list(coord))
                coord_prev = np.array([i,j+rad])
                new_coord,new_slope = getNextSlope(ima,rad,coord_prev,coord)
                #print(coord,new_coord,new_slope)
                if new_slope:
                    coord_prev = coord
                    coord = new_coord
                while not (new_slope is None):
                    points.append(list(coord))
                    new_coord,new_slope = getNextSlope(ima,rad,coord_prev,coord)
                    #print(coord,new_coord,new_slope)
                    if not (new_slope is None):
                        coord_prev = np.array(coord)
                        coord = np.array(new_coord)
                        chain.append(new_slope)
                    if len(chain)>segments:
                        break
                pt = np.array(points).T
                fig,axs = plt.subplots(2,figsize=(16,16))
                axs[0].imshow(ima,'gray')
                dib = axs[0].scatter(pt[1],pt[0])
                axs[1].plot(chain)
                axs[1].grid()
                plt.show()
                return chain
                
def slopeChainCode(ima,rad,segments):
    chain=[]
    for i in range(len(ima)):
        for j in range(len(ima[0])):
            if ima[i][j]>0:
                coord = np.array([i,j])
                coord_prev = np.array([i,j+rad])
                new_coord,new_slope = getNextSlope(ima,rad,coord_prev,coord)
                if new_slope:
                    coord_prev = coord
                    coord = new_coord
                while not (new_slope is None):
                    new_coord,new_slope = getNextSlope(ima,rad,coord_prev,coord)
                    if not (new_slope is None):
                        coord_prev = np.array(coord)
                        coord = np.array(new_coord)
                        chain.append(new_slope)
                    if len(chain)>segments:
                        break
                return chain

def discreteTortuosity(ssc):
    return np.sum(np.abs(ssc))

def getInitialPixel(ima):
    for i in range(len(ima)):
        for j in range(len(ima[0])):
            if ima[i][j]>0:
                return i,j

def slopeChainCodeUnitRadio(ima):
    code = []
    points = []
    ones = 0
    threes = 0
    i,j = getInitialPixel(ima)
    ant = [i,j]
    act = [i+1,j] # counterclockwise tracking
    break_loop = 0
    while break_loop<20000:
        break_loop += 1
        if act[:] in points:
            break
        pix = ima[act[0]-1:act[0]+1,act[1]-1:act[1]+1]
        pix_sum = np.sum(pix)
        points.append(act[:])
        if pix_sum==1:
            ones += 1
            act, new_elmnt = walk_corner1(pix,act)
            code.append(new_elmnt)
        elif pix_sum==2:
            if pix[0][0]==pix[0][1]:
                if pix[0][0]==1:
                    act[1] +=1
                    code.append(0)
                else:
                    act[1] -=1
                    code.append(1)
            elif pix[0][0]==0:
                act[0] +=1
                code.append(-0.67)
            else:
                act[0] -=1
                code.append(0.67)
        else:
            threes +=1
            act,new_elmnt = walk_corner3(pix,act)
            code.append(new_elmnt)
        if i%2==0:
            ant[0] = act[0]
            ant[1] = act[1]
    return np.array(code)

def walk_corner1(pix,act):
    if pix[0][0]>0:
        return [act[0]-1,act[1]], 0.16
    elif pix[0][1]>0:
        return [act[0],act[1]+1], -0.16
    elif pix[1][0]>0:
        return [act[0],act[1]-1], 0.84
    else:
        return [act[0]+1,act[1]], -0.84

def walk_corner3(pix,act):
    if pix[0][0]==0:
        return [act[0],act[1]-1], -0.67
    elif pix[0][1]==0:
        return [act[0]-1,act[1]], 0.67
    elif pix[1][0]==0:
        return [act[0]+1,act[1]], -0.33
    elif pix[1][1]==0:
        return [act[0],act[1]+1], 0.33
    else:
        return [act[0]+1,act[1]+1], -0.33