import sys
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_input_file_name = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module16-object-tracking\race_car.mp4"
tracker = cv2.legacy.TrackerBoosting_create()
#tracker = cv2.legacy.TrackerCSRT_create()
video_cap = cv2.VideoCapture(video_input_file_name)
ok, frame = video_cap.read()
if video_cap.isOpened():
    width  = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(video_cap.get(cv2.CAP_PROP_FPS))
else:
    print('Could not open video')
    sys.exit()
bbox = (820, 510, 420, 180)
frame_copy = frame.copy()
cv2.rectangle(frame_copy,(int(bbox[0]),int(bbox[1])),(int(bbox[0])+int(bbox[2]),int(bbox[1])+int(bbox[3])),(0,255,0),2,1)
tracker.init(frame, bbox)
'''plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(20, 8))
im = ax.imshow(frame)'''
window_width = 800
window_height = 600

# Create a named window with the WINDOW_NORMAL flag
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", window_width, window_height)
#frames = []
while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    ok, bbox = tracker.update(frame)
    if ok:
        cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[0])+int(bbox[2]),int(bbox[1])+int(bbox[3])), (0, 255, 255),2,1)
        #frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''im.set_array(frame)
    plt.draw()
    plt.pause(0.001)'''
video_cap.release()
cv2.destroyWindow()
'''clip = ImageSequenceClip(frames, fps)
clip = clip.resize(height=600)
clip.ipython_display()'''
'''plt.ioff()  # Turn off interactive mode
plt.show()'''