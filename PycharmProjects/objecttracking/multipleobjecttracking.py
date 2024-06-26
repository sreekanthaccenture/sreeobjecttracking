import cv2
from random import randint
import numpy as np
import matplotlib.pyplot as plt
videoPath = r"C:\Users\sreekanth.maramreddy\Desktop\ds\week9-python\data\videos\cycle.mp4"
cap = cv2.VideoCapture(videoPath)
success, frame = cap.read()
colors = []
for i in range(3):
    colors.append((randint(64, 255), randint(64, 255),randint(64, 255)))
bboxes = [(471, 250, 66, 159), (349, 232, 69, 102)]
multi_tracker = cv2.MultiTracker_create()
for bbox in bboxes:
    multi_tracker.add(cv2.TrackerKCF_create(), frame, bbox)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    success, boxes = multi_tracker.update(frame)
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 4, cv2.LINE_AA)
    if count % 10 == 0:
        plt.imshow(frame[:, :, ::-1])
        plt.show()
    count += 1
    if count > 50:
        break
cap.release()





