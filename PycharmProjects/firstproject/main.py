import cv2

video_cap = cv2.VideoCapture(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module04-video-processing-and-analysis\race_car.mp4")
win_name = 'Video Preview'
cv2.namedWindow(win_name)
while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
video_cap.release()
cv2.destroyWindow(win_name)