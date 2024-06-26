import cv2
import numpy as np
import matplotlib.pyplot as plt
modelfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_frozen_inference_graph.pb"
configfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\coco_class_labels.txt"
imagefile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\fruit-vegetable.jpg"
image = cv2.imread(imagefile,cv2.IMREAD_COLOR)
net = cv2.dnn.readNetFromTensorflow(modelfile,configfile)
classlist = []
with open(classfile) as f:
    classlist = f.read().split('\n')

def detect(img,net):
    blob = cv2.dnn.blobFromImage(img,1.0,(300,300),(0,0,0),True)
    net.setInput(blob)
    objects = net.forward()
    return objects
def drawboxes(img,objects,classlist):
    height,width = img.shape[0:2]
    for object in objects[0][0]:
        classid = int(object[1])
        score =  float(object[2])
        if score > 0.4:
            name = classlist[classid]
            x = int(object[3]*width)
            y = int(object[4]*height)
            w = int(object[5]*width - x)
            h = int(object[6]*height - y)
            img = drawtext(img,(x,y,w,h,name,score))
    return img
def drawtext(img,cord):
    text = f'{cord[4]}:{round(cord[5],2)}'
    cv2.rectangle(img,(cord[0],cord[1]),(cord[0]+cord[2],cord[1]+cord[3]),(255,0,0),1)
    (width,height),banner = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.7,1)
    cv2.rectangle(img,(cord[0],cord[1]),(cord[0]+int(width),cord[1]+int(height)+int(banner)),(0,0,0),-1)
    cv2.putText(img,text,(cord[0],cord[1]+int(height)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
    return img
objects = detect(image,net)
mimage = drawboxes(image,objects,classlist)
plt.figure(figsize=(15,10))
plt.imshow(mimage[:,:,::-1])
plt.show()



