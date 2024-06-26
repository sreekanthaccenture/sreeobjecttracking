import cv2
import numpy as np

k = 0
video_path = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\Applications\intruder_2.mp4"
cap = cv2.VideoCapture(video_path)
bg_subtractor = cv2.createBackgroundSubtractorKNN()
x1 = 0
x2 = 0
y1 = 0
y2 = 0
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#output_video_path = 'output_video8.avi'
#out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    fg_mask = bg_subtractor.apply(frame)
    fg_mask_erode = cv2.erode(fg_mask,np.ones((5,5),np.uint8))
    blurred = cv2.GaussianBlur(fg_mask_erode, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    #cv2.imshow('Motion Detection', edges)
    #_, thresh = cv2.threshold(fg_mask_erode, 50, 160, cv2.THRESH_BINARY)
    #cv2.imshow('Motion Detection', thresh)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours,key = cv2.contourArea,reverse=True)
    for i in range(min(2,len(contours_sorted))):
        xc, yc, wc, hc = cv2.boundingRect(contours_sorted[i])
        if i == 0:
            x1 = xc
            y1 = yc
            x2 = xc + wc
            y2 = yc + hc
        else:
            x1 = min(x1, xc)
            y1 = min(y1, yc)
            x2 = max(x2, xc + wc)
            y2 = max(y2, yc + hc)
    cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)

    #out.write(frame)

    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
