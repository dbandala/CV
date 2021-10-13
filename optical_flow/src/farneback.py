import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import flow_vis
import os


path = '../data/alley_1/*.png'

imgF = sorted(glob.glob(path))
detach_dir = '.'
if 'farneback_results' not in os.listdir(detach_dir):
    os.mkdir('farneback_results')

for i in range(len(imgF)-1):
    act = cv2.imread (imgF[i], cv2.IMREAD_GRAYSCALE)
    sig = cv2.imread(imgF[i+1],cv2.IMREAD_GRAYSCALE)
    
    flow = cv2.calcOpticalFlowFarneback(act, sig, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    flow_mag_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    cv2.imshow('Flow estimated with Farnerback',flow_mag_color)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
   # cv2.imshow('Input sequence',cv2.imread(imgF[i]))
   # j = cv2.waitKey(30) & 0xff
   # if j == 27:
   #     break
    cv2.imwrite(f'./farneback_results/optical_farne_{i}.png', flow_mag_color)

