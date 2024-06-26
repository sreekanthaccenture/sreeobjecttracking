import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\Applications\intruder_2.mp4")
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)
ksize = (5,5)
frame_start = 0
frame_count = 5
max_contors = 3
x1 = None
y1 = None
x2 = None
y2 = None
while True:
    ok,frame = cap.read()
    if not ok:
        break
    else:
        z = frame.copy()
    frame_start = frame_start+1
    fg_mask = bg_sub.apply(frame)
    if frame_start > frame_count:
        fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
        blurred = cv2.GaussianBlur(fg_mask_erode, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in range(min(max_contors, len(contours_sorted))):
                x, y, w, h = cv2.boundingRect(contours_sorted[cnt])
                if cnt == 0:
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                else:
                    x1 = min(x1, x)
                    y1 = min(y1, y)
                    x2 = max(x + w, x2)
                    y2 = max(y + h, y2)
            cv2.rectangle(z, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.imshow("out", z)
    cv2.waitKey(3)
cap.release()