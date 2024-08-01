


import cv2
import numpy as np
import imutils

lowHSV = np.array([0,98,123])
highHSV =np.array([71,219,255]) 
# copy the path of the video
#path = 'C:\\Users\\pc\\vid_mpeg4.mp4'
path = "D:\\PROJECT_MOHD_SAHIL\\DAY 13\\VIRAT_S_050201_05_000890_000944.mp4"

# create the video reader object
vid = cv2.VideoCapture(path)
print(vid)
print(vid.isOpened())
frame_counter=0


while(vid.isOpened()):
    val,frame=vid.read()
    image_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_counter +=1
    # if the frame is captured
    if(val):
        mask_image = cv2.inRange(image_hsv,(lowHSV),(highHSV))
        
        # find contours
        cnts = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = imutils.grab_contours(cnts)
        #big object detection
        for c in final_contours:
            area = cv2.contourArea(c)
            if(area>950 and area<976):
                print(area)
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # c have every info about contour
                cv2.drawContours(frame, [c], -1, (255,0,0))
                cv2.circle(frame, (cx,cy), 4 ,(255,0,0))
    
    
        cv2.imshow('Frame',frame)
        #cv2.imshow('mask_image',mask_image)
    if(cv2.waitKey(1)==ord('q')):
        break
    

    
vid.release() # close the object
cv2.destroyAllWindows()
        
print(frame_counter)    
