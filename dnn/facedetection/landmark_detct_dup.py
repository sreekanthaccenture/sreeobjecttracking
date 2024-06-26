import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
net = cv2.dnn.readNetFromCaffe(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\deploy.prototxt",r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\res10_300x300_ssd_iter_140000.caffemodel")


def detect_faces(image, detection_threshold=0.70):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    faces = []
    img_h = image.shape[0]
    img_w = image.shape[1]
    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left = detection[3] * img_w
            top = detection[4] * img_h
            right = detection[5] * img_w
            bottom = detection[6] * img_h
            face_w = right - left
            face_h = bottom - top
            face_roi = (left, top, face_w, face_h)
            faces.append(face_roi)
    return np.array(faces).astype(int)

landmarkDetector = cv2.face.createFacemarkLBF()
model = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\lbfmodel.yaml"
landmarkDetector.loadModel(model)

image_filename = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\visuals\c0-m14-NB-img1.jpg"
img = cv2.imread(image_filename)
img_display_faces = img.copy()
img_display_marks = img.copy()
faces = detect_faces(img)
if len(faces) > 0:
    for face in faces:
        cv2.rectangle(img_display_faces, face, (0 ,255 ,0), 3)
    retval, landmarksList = landmarkDetector.fit(img, faces)
    for landmarks in landmarksList:
        cv2.face.drawFacemarks(img_display_marks, landmarks, (0, 255, 0))
    fig = plt.figure(figsize=(20 ,10))
    plt.subplot(121); plt.imshow(img_display_faces[... ,::-1]); plt.axis('off');
    plt.subplot(122); plt.imshow(img_display_marks[... ,::-1]); plt.axis('off');
    plt.show()
else:
    print('No faces detected in image.')