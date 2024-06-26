import cv2
import numpy as np
video_path = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\Applications\intruder_2.mp4"
cap = cv2.VideoCapture(video_path)
bg_subtractor = cv2.createBackgroundSubtractorKNN()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output_video11.avi'
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fg_mask = bg_subtractor.apply(frame)
    fg_mask_erode = cv2.erode(fg_mask, np.ones((5, 5), np.uint8),iterations=3)
    dilated_image = cv2.dilate(fg_mask_erode, np.ones((5, 5), np.uint8), iterations=3)
    out.write(fg_mask_erode)
    cv2.imshow('sc',fg_mask_erode)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
