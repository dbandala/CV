""" The code heavily based on opencv (https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html). """

import os
import cv2 as cv
import numpy as np
from utils import getfiles


# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300,
                      qualityLevel = 0.2,
                      minDistance = 2,
                      blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Variable for color to draw optical flow track
color = (0, 255, 0)


def lk_from_image(path):
    flist = getfiles(path)
    detach_dir = '.'
    if 'lk_results' not in os.listdir(detach_dir):
        os.mkdir('lk_results')

    first_frame = cv.imread(flist[0])
    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)

    for i in range(len(flist) - 1):
        im1 = cv.imread(flist[i])
        im2 = cv.imread(flist[i+1])
        gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
        prev = cv.goodFeaturesToTrack(gray1, mask = None, **feature_params)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv.calcOpticalFlowPyrLK(gray1, gray2, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]
        # Draws the optical flow tracks
        for new, old in zip(good_new, good_old):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            im2 = cv.circle(im2, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv.add(im2, mask)
        # Updates previous frame
        gray1 = gray2.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        cv.imshow('Flow estimated with LK',output)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        cv.imwrite(f'./lk_results/optical_lk_{i}.png', output)


path = '../data/alley_1'
lk_from_image(path)


