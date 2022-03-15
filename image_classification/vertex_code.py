#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def getInitialPixel(ima):
    for i in range(len(ima)):
        for j in range(len(ima[0])):
            if ima[i][j]>0:
                return i,j

def vertexChainCodeWithGraphics(ima,max_iterations=100000):
    ones = 0
    threes = 0
    cad = []
    points = []
    i,j = getInitialPixel(ima)
    points.append([i,j])
    act = [i+1,j] # counterclockwise tracking
    break_loop = 0
    edges_img = np.zeros((ima.shape))
    #while ones!=4+threes and break_loop<max_iterations: # iterate till closed loop
    while break_loop<max_iterations:
        break_loop += 1
        if act[:] in points:
            break
        pix = ima[act[0]-1:act[0]+1,act[1]-1:act[1]+1]
        pix_sum = np.sum(pix)
        cad.append(int(pix_sum))
        points.append(act[:])
        edges_img[act[0],act[1]] = 1
        if pix_sum==1:
            ones += 1
            act = walk_corner1(pix,act)
        elif pix_sum==2:
            if pix[0][0]==pix[0][1]:
                if pix[0][0]==1:
                    act[1] +=1
                else:
                    act[1] -=1
            elif pix[0][0]==0:
                act[0] +=1
            else:
                act[0] -=1
        else:
            threes +=1
            act = walk_corner3(pix,act)
    pt = np.array(points).T
    fig = plt.figure(figsize=(10,6))
    plt.imshow(ima,'gray')
    dib = plt.scatter(pt[1],pt[0],2,c=np.arange(len(pt[0])),cmap='jet')
    plt.colorbar(dib)
    plt.show()
    return np.array(cad),edges_img
    
def vertexChainCode(ima,max_iterations=100000):
    ones = 0
    threes = 0
    cad = []
    i,j = getInitialPixel(ima)
    act = [i+1,j] # counterclockwise tracking
    break_loop = 0
    edges_img = np.zeros((ima.shape))
    while ones!=4+threes and break_loop<max_iterations: # iterate till closed loop
        break_loop += 1
        pix = ima[act[0]-1:act[0]+1,act[1]-1:act[1]+1]
        pix_sum = np.sum(pix)
        cad.append(int(pix_sum))
        edges_img[act[0],act[1]] = 1
        if pix_sum==1:
            ones += 1
            act = walk_corner1(pix,act)
        elif pix_sum==2:
            if pix[0][0]==pix[0][1]:
                if pix[0][0]==1:
                    act[1] +=1
                else:
                    act[1] -=1
            elif pix[0][0]==0:
                act[0] +=1
            else:
                act[0] -=1
        else:
            threes +=1
            act = walk_corner3(pix,act)
    return np.array(cad),edges_img
    
def walk_corner1(pix,act):
    if pix[0][0]>0:
        return [act[0]-1,act[1]]
    elif pix[0][1]>0:
        return [act[0],act[1]+1]
    elif pix[1][0]>0:
        return [act[0],act[1]-1]
    else:
        return [act[0]+1,act[1]]
        
def walk_corner3(pix,act):
    if pix[0][0]==0:
        return [act[0],act[1]-1]
    elif pix[0][1]==0:
        return [act[0]-1,act[1]]
    elif pix[1][0]==0:
        return [act[0]+1,act[1]]
    elif pix[1][1]==0:
        return [act[0],act[1]+1]
    else:
        return [act[0]+1,act[1]+1]


# In[23]:


#vcode = vertexChainCode(contour)


