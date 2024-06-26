import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\visuals\c0-m14-NB-img1.jpg", cv2.IMREAD_COLOR)

def detect(frame, net, scale, mean, in_width, in_height):
    h = frame.shape[0]
    w = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    print(detections)
    # Process each detection.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:

            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            # Annotate the video frame with the detection results.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = 'Confidence: %.4f' % confidence
            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))
    return frame

mean = [104, 117, 123]
scale = 1.0
in_width = 300
in_height = 300

detection_threshold = 0.5

font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

net = cv2.dnn.readNetFromCaffe(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\deploy.prototxt",r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\res10_300x300_ssd_iter_140000.caffemodel")
op1 = detect(img1, net, scale, mean, in_width, in_height)

plt.figure(figsize = [15,10])
plt.imshow(op1[:,:,::-1])
plt.title("Image 2")
