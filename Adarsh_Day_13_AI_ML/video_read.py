# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:25:54 2024

@author: STUDENT
"""
# using opencv read the video file and run it frame
# by frame
# in python environment

import cv2
import numpy as np

# copy the path of the video
path = "D:\\PROJECT_MOHD_SAHIL\\DAY 13\\vid_mpeg4.mp4"
# path1 = "D:\\PROJECT_MOHD_SAHIL\\DAY 13\\Street - 3617.mp4"


# create video reader object
vid = cv2.VideoCapture(path)

print(vid)
print(vid.isOpened())
frame_count = 0

while(vid.isOpened()):
    val,frame = vid.read()
    frame_count += 1
    
    # if frame is captured
    if(val):
        gray_im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_im = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        cv2.imshow('GRAY', gray_im)
        cv2.imshow('Original', frame)
        cv2.imshow('HSV', hsv_im)
        cv2.imshow('YCRCB', ycrcb_im)
        if(cv2.waitKey(1)==ord('q')):
            break

vid.release()
cv2.destroyAllWindows()
print(frame_count)
