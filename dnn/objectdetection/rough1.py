import numpy as np
import cv2
#import matplotlib.pyplot as plt
# r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\coco.names"
'''
classesFile = 
classes = None
with open(classesFile, 'rt') as f:
  classes = f.read().rstrip('\n').split('\n')
print(classes)
print(len(classes))
'''
def dist(A,B):
  p1 = np.sum(A ** 2, axis=1)[:, np.newaxis]
  p2 = np.sum(B ** 2, axis=1)
  p3 = -2 * np.dot(A, B.T)
  z = np.round(np.sqrt(p1 + p2 + p3), 2)
  return z
def distmine(a,b):
  matt = np.zeros((3,3),dtype=float)
  for i in range(len(a)):
    for j in range(len(a)):
      if i == j:
        continue
      dist = pow(pow(b[j][0] - a[i][0],2) + pow(b[j][1] - a[i][1],2),0.5)
      matt[i][j] = dist
      s = np.round(matt, 2)
  return  s

a = np.array([(1,2),(3,4),(5,6)])
b = np.array([(1,2),(3,4),(5,6)])
k = dist(a,b)
g = distmine(a,b)
print(k)
print(g)



