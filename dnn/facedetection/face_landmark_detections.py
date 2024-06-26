import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

img1 = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\visuals\c0-m14-NB-img1.jpg", cv2.IMREAD_COLOR)
def detect(frame, net, scale, mean, in_width, in_height):
    h = frame.shape[0]
    w = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            p = x2-x1
            q = y2-y1
            face_roi = (x1, y1, p, q)
            faces.append(face_roi)

    return np.array(faces).astype(int)

mean = [104, 117, 123]
scale = 1.0
in_width = 300
in_height = 300
detection_threshold = 0.7
net = cv2.dnn.readNetFromCaffe(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\deploy.prototxt",r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\res10_300x300_ssd_iter_140000.caffemodel")
op1 = detect(img1, net, scale, mean, in_width, in_height)
alpha = img1.copy()
beta = img1.copy()
landmarkDetector = cv2.face.createFacemarkLBF()
model = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\lbfmodel.yaml"
landmarkDetector.loadModel(model)
for i in op1:
    cv2.rectangle(alpha, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
retval, landmarksList = landmarkDetector.fit(img1, op1)
for landmarks in landmarksList:
    cv2.face.drawFacemarks(beta, landmarks, (0, 255, 0))
plt.imshow(beta[...,::-1])
plt.show()