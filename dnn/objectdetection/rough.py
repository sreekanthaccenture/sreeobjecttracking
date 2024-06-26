import numpy as np
import cv2
import matplotlib.pyplot as plt


modelFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_frozen_inference_graph.pb"
configFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\coco_class_labels.txt"

with open(classFile) as fp:
    labels = fp.read().split('\n')
k = labels[int(5.70000000e+01)]
print(k)

frame = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\street.jpg")
print(frame)