import numpy as np
import cv2
from collections import deque
def nothing(x):
    pass
cv2.namedWindow('Threshold')
cap=cv2.VideoCapture(0)
mask=None
count=0
h, s, v=100, 100, 100
cv2.createTrackbar('Hue       ', 'Threshold',0,179,nothing)
cv2.createTrackbar('Saturation', 'Threshold',0,255,nothing)
cv2.createTrackbar('Value     ', 'Threshold',0,255,nothing)
while(True):
    ret, frame = cap.read() 
    if ret==True:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        frame=cv2.flip(frame,1) 
        hsv=cv2.flip(hsv,1) 
        h=cv2.getTrackbarPos('Hue       ','Threshold')
        s=cv2.getTrackbarPos('Saturation','Threshold')
        v=cv2.getTrackbarPos('Value     ','Threshold')  
        lower=np.array([h-10,s,v], dtype=np.uint8)
        upper=np.array([h+10,255,255], dtype=np.uint8)
        mask=cv2.inRange(hsv, lower, upper)     
        mask=cv2.bitwise_and(hsv,hsv,mask=mask)     
        cv2.imshow("Threshold", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break
cv2.destroyAllWindows()

cv2.namedWindow('Frame')
cv2.createTrackbar('Thickness', 'Frame', 0, 20, nothing)
cv2.createTrackbar('Red      ', 'Frame',0,255,nothing)
cv2.createTrackbar('Green    ', 'Frame',0,255,nothing)
cv2.createTrackbar('Blue     ', 'Frame',0,255,nothing)
its=deque()
pts=deque()
thickness=deque()
red=deque()
green=deque()
blue=deque()
l=0
while(True):
    ret, frame=cap.read()
    if ret==True:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        frame=cv2.flip(frame,1)
        hsv=cv2.flip(hsv,1) 
        Thickness=cv2.getTrackbarPos('Thickness','Frame')   
        Red=cv2.getTrackbarPos('Red      ','Frame') 
        Green=cv2.getTrackbarPos('Green    ','Frame')   
        Blue=cv2.getTrackbarPos('Blue     ','Frame')    
        lower=np.array([h-10,s,v], dtype=np.uint8)
        upper=np.array([h+10,255,255], dtype=np.uint8)
        mask=cv2.inRange(hsv, lower, upper)
        mask=cv2.erode(mask, None, iterations=2)
        mask=cv2.dilate(mask, None, iterations=2)   
        cnts=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts)>0:
            c=max(cnts, key=cv2.contourArea)
            ((x, y), radius)=cv2.minEnclosingCircle(c)
            M=cv2.moments(c)
            center=(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            if l>=20:
                pts.appendleft(center)
                thickness.appendleft(Thickness)
                l=0
                its.appendleft(20)
                red.appendleft(Red) 
                green.appendleft(Green) 
                blue.appendleft(Blue)   
            its.appendleft(0)                   
            pts.appendleft(center)  
            red.appendleft(Red) 
            green.appendleft(Green) 
            blue.appendleft(Blue)   
            thickness.appendleft(Thickness) 
        else:
            l=l+1
        for i in xrange(1, len(pts)):
            if its[i-1]<20:
                cv2.line(frame, pts[i - 1], pts[i], (blue[i-1], green[i-1], red[i-1]), thickness[i-1])
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord("c"):
                its.clear()
                pts.clear()
                red.clear()
                green.clear()
                blue.clear()
                thickness.clear()
        if key==ord("s"):
            count=count+1
            cv2.imwrite("Frame%d.jpg" % count, frame) 
        if key==ord("q"):
            break
    else:
        break
pts.clear()
its.clear()
cap.release()
cv2.destroyAllWindows()
