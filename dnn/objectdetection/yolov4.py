import numpy as np
import cv2
import sys
from os import path
import matplotlib.pyplot as plt

INPUT_WIDTH = 640             # Width of network's input image, larger is slower but more accurate
INPUT_HEIGHT = 640            # Height of network's input image, larger is slower but more accurate
SCORE_THRESHOLD = 0.5         # Class score threshold, accepts only if score is above the threshold.
NMS_THRESHOLD = 0.45          # Non-maximum suppression threshold, higher values result in duplicate boxes per object
CONFIDENCE_THRESHOLD = 0.45

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.5
THICKNESS = 4

def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    print(output_layers)
    outputs = net.forward(output_layers)
    print(outputs)
    # print(outputs[0].shape)
    return outputs

def post_process(input_image, outputs):
    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    print(image_height, image_width)
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    print(x_factor, y_factor)
    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        print(row)
        confidence = row[4]
        print(confidence)
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            print(f'classes_scores: {classes_scores}')
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
            print(class_id)
            print(classes_scores[class_id])
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                print(confidences)
                print(class_ids)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                print(cx, cy, w, h)
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                print(left,top,width,height)
                box = np.array([left, top, width, height])
                print(box)
                boxes.append(box)
                print(boxes)
                break
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print(indices)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), (255,178,50), 4*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, (0,255,255), THICKNESS, cv2.LINE_AA)

def put_efficiency(input_img, net):
  t, _ = net.getPerfProfile()
  print(t)
  print(cv2.getTickFrequency())
  label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
  print(label)
  cv2.putText(input_img, label, (20, 80), FONT_FACE, FONT_SCALE, (0,0,255), THICKNESS, cv2.LINE_AA)

frame = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\street.jpg")
classesFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\coco.names"
classes = None
with open(classesFile, 'rt') as f:
  classes = f.read().rstrip('\n').split('\n')

modelWeights = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection-YOLOv5\yolov5s.onnx"
net = cv2.dnn.readNet(modelWeights)
detections = pre_process(frame, net)
img = post_process(frame.copy(), detections)
plt.imshow(img[...,::-1])
plt.show()
put_efficiency(img, net)
plt.imshow(img[...,::-1])
plt.show()
