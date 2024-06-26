import cv2
import matplotlib.pyplot as plt
import numpy as np
frame = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\street.jpg",cv2.IMREAD_COLOR)
classesFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\coco.names"
modelWeights = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\yolov5s.onnx"
classlist = []
with open(classesFile) as f:
    classlist = f.read().split('\n')
def detectobjects(image,net):
    blob = cv2.dnn.blobFromImage(image,1/255,(640,640),(0,0,0),True,False)
    net.setInput(blob)
    outputlayers = net.getUnconnectedOutLayersNames()
    objects = net.forward(outputlayers)
    return objects
def drawtext(image,objects,classlist):
    height,width = image.shape[0:2]
    xfactor = width/640
    yfactor = height/640
    details = []
    for object in objects[0][0]:
        confidence = float(object[4])
        if confidence > 0.45:
            score = object[5:]
            scoreindex = np.argmax(score)
            if score[scoreindex] > 0.5:
                cx = object[0]
                cy = object[1]
                w = object[2]
                h = object[3]
                x = int((cx - w / 2) * xfactor)
                y = int((cy - h / 2) * yfactor)
                w = int(w * xfactor)
                h = int(h * yfactor)
                name = classlist[scoreindex]
                details.append(([x, y, w, h], name, confidence))
    indices = cv2.dnn.NMSBoxes([i[0] for i in details],[j[2] for j in details],0.45,0.45)
    for index in indices:
        boxcord = details[index][0]
        boxname = details[index][1]
        boxconf = details[index][2]
        image = banner(image,(boxcord,boxname,boxconf))
    return image

def banner(image,boxdetails):
    name = boxdetails[1]
    confidence = boxdetails[2]
    x =  boxdetails[0][0]
    y =  boxdetails[0][1]
    w =  boxdetails[0][2]
    h =  boxdetails[0][3]
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),4)
    label = f'{name}:{round(confidence,2)}'
    (width,height),banner = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,1,2)
    cv2.rectangle(image,(x,y),(x+width,y+height+banner),(0,0,0),-1)
    cv2.putText(image,label,(x,y+height),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    return image

net = cv2.dnn.readNet(modelWeights)
objects = detectobjects(frame,net)
frame = drawtext(frame,objects,classlist)
plt.figure(figsize=(15,10))
plt.imshow(frame[:,:,::-1])
plt.show()
