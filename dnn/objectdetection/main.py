import numpy as np
import cv2
import matplotlib.pyplot as plt

modelFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_frozen_inference_graph.pb"
configFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\coco_class_labels.txt"

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
with open(classFile) as fp:
    labels = fp.read().split('\n')
def detect_objects(net, img):
    dim = 300
    mean = (0, 0, 0)
    blob = cv2.dnn.blobFromImage(img, 1.0, (dim, dim), mean, True)
    net.setInput(blob)
    objects = net.forward()
    return objects

food_img = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module15-object-detection\fruit-vegetable.jpg")
food_objects = detect_objects(net, food_img)


def draw_text(im, text, x, y):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    textSize = cv2.getTextSize(text, fontface, font_scale, thickness)
    dim = textSize[0]
    baseline = textSize[1]

    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED);
    cv2.putText(im, text, (x, y + dim[1]), fontface, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def draw_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]

    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        if score > threshold:
            draw_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return mp_img

result = draw_objects(food_img.copy(), food_objects, 0.4)
plt.figure(figsize=(30, 10)); plt.imshow(result); plt.show();