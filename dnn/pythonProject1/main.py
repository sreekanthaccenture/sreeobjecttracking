import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

with open(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module13-introduction-to-deep-learning-with-opencv\input\classification_classes_ILSVRC2012.txt", 'r') as f:
    image_net_names = f.read().split('\n')
    class_names = image_net_names[:-1]

config_file = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module13-introduction-to-deep-learning-with-opencv\models\DenseNet_121.prototxt"
model_file = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module13-introduction-to-deep-learning-with-opencv\models\DenseNet_121.caffemodel"
model = cv2.dnn.readNet(model=model_file, config=config_file, framework='Caffe')

tiger_img = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module13-introduction-to-deep-learning-with-opencv\input\image1.jpg")
print(tiger_img.shape)
print(tiger_img.mean())
'''plt.figure(figsize=[10, 10])
plt.imshow(tiger_img[:, :, ::-1])
plt.show()'''
blob = cv2.dnn.blobFromImage(image=tiger_img, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123), swapRB=False, crop=False)
model.setInput(blob)
outputs = model.forward()
final_outputs = outputs[0]
final_outputs = final_outputs.reshape(1000, 1)
label_id = np.argmax(final_outputs)
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
print(probs[:10])
print("Max probability:", np.max(probs))
final_prob = np.max(probs) * 100.0
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}%"
plt.imshow(tiger_img[:, :, ::-1])
plt.title(out_text)
plt.xticks([]), plt.yticks([])
plt.show()
