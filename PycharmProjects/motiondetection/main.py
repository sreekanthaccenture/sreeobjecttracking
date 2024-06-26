import cv2
import numpy as np
import matplotlib.pyplot as plt
x1 = None
x2 = None
y1 = None
y2 = None
def concolor(img,frame):
    x,y,w,h = cv2.boundingRect(img)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    return img
videopath = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\Applications\intruder_2.mp4"
vcap = cv2.VideoCapture(videopath)
#outpath = 'int_detect_1.avi'
#out = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc(*'MP4V'),vcap.get(cv2.CAP_PROP_FPS),(int(vcap.get(3))*2,int(vcap.get(4))*2))
bacsub = cv2.createBackgroundSubtractorKNN(history=200)
while True:
    ret,frame = vcap.read()
    if frame is None:
        break
    framecopy = frame.copy()
    fgmask = bacsub.apply(frame)
    fgmaskerode = cv2.erode(fgmask, (5, 5),iterations=3)
    contours,_ = cv2.findContours(fgmaskerode,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours,key = cv2.contourArea, reverse=True)
    for i in range(min(3,len(contours_sorted))):
        xc,yc,wc,hc = cv2.boundingRect(contours_sorted[i])
        if i == 0:
            x1 = xc
            y1 = yc
            x2 = xc+wc
            y2 = yc+hc
        else:
            x1 = min(xc,x1)
            y1 = min(y1,yc)
            x2 = min(x2,xc+wc)
            y2 = min(y2,yc+hc)
    cv2.rectangle(framecopy,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('screen', framecopy)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
vcap.release()
cv2.destroyWindow()


