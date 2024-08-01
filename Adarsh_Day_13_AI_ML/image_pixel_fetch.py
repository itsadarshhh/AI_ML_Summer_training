# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:22:20 2024

@author: STUDENT
"""

# background substraction on a video

import cv2
import numpy as np
import imutils


# copy the path of the video
path = "D:\\PROJECT_MOHD_SAHIL\\DAY 13\\VIRAT_S_050201_05_000890_000944.mp4"


def mouseHSV (event,x,y,flag,param):
    if(event==cv2.EVENT_FLAG_LBUTTON):
        colorH = frame[x,y,0]
        colorS = frame[x,y,1]
        colorV = frame[x,y,2]
        print('BGR values : ',colorH,colorS,colorV)
        print('corr : ',x,y)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouseHSV)

# create video reader object
vid = cv2.VideoCapture(path)

print(vid)
print(vid.isOpened())
frame_count = 0
val,frame=vid.read()
image_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow('frame', image_hsv)
        
while(1):    
        
    if(cv2.waitKey(1)==ord('q')):
            break

vid.release()
cv2.destroyAllWindows()
print(frame_count)
