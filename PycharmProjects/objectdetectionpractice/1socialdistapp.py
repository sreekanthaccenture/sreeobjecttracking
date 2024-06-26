import cv2
import numpy as np
import matplotlib.pyplot as plt
configfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\Applications\MobileNetSSD_deploy.prototxt"
modelfile  = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\Applications\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(configfile,modelfile)
videopath = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\Applications\input.mp4"
cap = cv2.VideoCapture(videopath)
def detections(image,net):
    blob = cv2.dnn.blobFromImage(image,0.007843,(300,300),(127.5,127.5,127.5))
    net.setInput(blob)
    return net.forward()
def drawtext(image,objects):
    height,width = image.shape[0:2]
    for object in objects[0][0]:
        confidence = float(object[2])
        if int(object[1]) == 15 and confidence > 0.7:
            x = int(object[3]*width)
            y = int(object[4]*height)
            w = int(object[5]*width) -x
            h = int(object[6]*height) - y
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),4)
    return image
while True:
    ret,frame = cap.read()
    if not ret:
        break
    persons = detections(frame,net)
    image = drawtext(frame,persons)
    cv2.imshow("screen", image)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
cap.release()