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
trackers = []
for bbox in bboxes:
    tracker = cv2.TrackerKCF_create()  # You can use other trackers like TrackerMIL, TrackerCSRT, etc.
    tracker.init(frame, bbox)
    trackers.append(tracker)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    if count % 10 == 0:
        plt.imshow(frame[:, :, ::-1])
        plt.show()
    count += 1
    if count > 50:
        break
cap.release()





