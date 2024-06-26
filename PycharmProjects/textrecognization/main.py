import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module19-text-detecttion-and-ocr\visuals\dutch_signboard.jpg")
inputSize = (320, 320)
textDetectorEAST = cv2.dnn_TextDetectionModel_EAST(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module19-text-detecttion-and-ocr\resources\frozen_east_text_detection.pb")
conf_thresh = 0.8
nms_thresh = 0.4
textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)
imEAST = image.copy()
boxesEAST, confsEAST = textDetectorEAST.detect(image)
print(boxesEAST[0])
cv2.polylines(imEAST, boxesEAST, isClosed=True, color=(255, 0, 255), thickness=4)
plt.figure(figsize=(10,8))
plt.imshow(imEAST[:, :, ::-1]); plt.title('Bounding boxes for EAST')
plt.show()



