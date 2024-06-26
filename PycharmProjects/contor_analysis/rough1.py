import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r"C:\Users\sreekanth.maramreddy\Desktop\ds\week9-python\data\videos\video.mp4")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(n_frames)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(w,h,fps)
out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w*2, h))
SMOOTHING_RADIUS=50
_, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
transforms = np.zeros((n_frames-1, 3), np.float32)
print(prev.shape)
print(transforms)
for i in range(n_frames-2):
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
  print(f'prev_pts:{prev_pts}')
  success, curr = cap.read()
  if not success:
    break
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
  print(f'curr_pts:{curr_pts}')
  print(f'status:{status}')
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]
  m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
  print(f'm:{m}')
  dx = m[0][0,2]
  dy = m[0][1,2]
  da = np.arctan2(m[0][1,0], m[0][0,0])
  transforms[i] = [dx,dy,da]
  prev_gray = curr_gray
  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
  if i ==2:
      break

print(transforms)