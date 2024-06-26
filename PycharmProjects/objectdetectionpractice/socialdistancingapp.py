import cv2
import numpy as np
import matplotlib.pyplot as plt
modelfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_frozen_inference_graph.pb"
configfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classfile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\coco_class_labels.txt"
videopath = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\Applications\input.mp4"
cap = cv2.VideoCapture(videopath)
net = cv2.dnn.readNetFromTensorflow(modelfile,configfile)
classlist = []
with open(classfile) as f:
    classlist = f.read().split('\n')
def detectperson(image,net):
    persons = []
    height,width = image.shape[0:2]
    blob = cv2.dnn.blobFromImage(image,1.0,(300,300),(0,0,0),True)
    net.setInput(blob)
    objects = net.forward()
    for object in objects[0][0]:
        classid = int(object[1])
        score =  float(object[2])
        if classid == 1 and score >0.4:
            x = int(object[3] * width)
            y = int(object[4] * height)
            w = int(object[5] * width) - x
            h = int(object[6] * height) - y
            cx = x + int(w/2)
            cy = y + int(h/2)
            persons.append((x,y,w,h,cx,cy))
    return persons
def vio(persons):
    violations = set()
    if len(persons) >=2:
        for i in range(len(persons)):
            for j in range(i+1,len(persons)):
                person1 = persons[i]
                person2 = persons[j]
                refdist = int(1.2*(min(person1[2],person2[2])))
                ecudist = pow((pow((person2[4] - person1[4]),2) + pow((person2[5] - person1[5]),2)),0.5)
                if ecudist < refdist:
                    violations.add(i)
                    violations.add(j)
    return violations
while True:
    ret,image = cap.read()
    if not ret:
        break
    persons = detectperson(image, net)
    violations = vio(persons)
    for index,person in enumerate(persons):
        color = (0,0,255) if index in violations else (0,255,0)
        textt = "NOT SAFE"if index in violations else "SAFE"
        cv2.rectangle(image,(person[0],person[1]),(person[0]+person[2],person[1]+person[3]),color,2)
        cv2.putText(image,textt,(person[0],person[1]-1),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2 )

    cv2.imshow("screen", image)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
cap.release()