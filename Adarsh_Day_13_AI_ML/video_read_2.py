# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:52:45 2024

@author: STUDENT
"""


# background substraction on a video

import cv2
import numpy as np
import imutils


# copy the path of the video
path = "D:\\PROJECT_MOHD_SAHIL\\DAY 13\\VIRAT_S_050201_05_000890_000944.mp4"
# path1 = "D:\\PROJECT_MOHD_SAHIL\\DAY 13\\Street - 3617.mp4"

# background substractor
# back_gd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 

#back_subs = cv2.createBackgroundSubtractorMOG2(varThreshold=500,detectShadows=False)

back_subs = cv2.createBackgroundSubtractorKNN(dist2Threshold=750,detectShadows=False) 


# create video reader object
vid = cv2.VideoCapture(path)

print(vid)
print(vid.isOpened())
frame_count = 0

f=0

while(f<=500):
    f+=1
    val,frame = vid.read()
    frame_count += 1
    if(val):
    
        # video background substraction
        mask_im = back_subs.apply(frame)
        
        # find contours
        cnts = cv2.findContours(mask_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = imutils.grab_contours(cnts)
        
        #big object detection
        for c in final_contours:
            area = cv2.contourArea(c)
            if(area>250):
                print(area)
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # c have every info about contour
                cv2.drawContours(frame, [c], -1, (0,0,255))
                cv2.circle(frame, (cx,cy), 4 ,(0,255,0))
                
                
        cv2.imshow('Mask', mask_im)
        cv2.imshow('Original', frame)
        
    
        
    if(cv2.waitKey(1)==ord('q')):
            break

vid.release()
cv2.destroyAllWindows()
print(frame_count)
