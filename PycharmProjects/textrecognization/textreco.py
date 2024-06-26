import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
image = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module19-text-detecttion-and-ocr\visuals\dutch_signboard.jpg")
vocabulary =[]
with open(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module19-text-detecttion-and-ocr\resources\alphabet_94.txt") as f:
    for l in f:
        vocabulary.append(l.strip())
    f.close()
textDetector = cv2.dnn_TextDetectionModel_DB(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module19-text-detecttion-and-ocr\resources\DB_TD500_resnet50.onnx")
inputSize = (640, 640)
binThresh = 0.3
polyThresh = 0.5
mean = (122.67891434, 116.66876762, 104.00698793)
textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
textDetector.setInputParams(1.0/255, inputSize, mean, True)
textRecognizer = cv2.dnn_TextRecognitionModel(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module19-text-detecttion-and-ocr\resources\crnn_cs.onnx")
textRecognizer.setDecodeType("CTC-greedy")
textRecognizer.setVocabulary(vocabulary)
textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)
boxes, confs = textDetector.detect(image)
cv2.polylines(image, boxes, True, (255, 0, 255), 4)
def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([[0, outputSize[1] - 1],[0, 0],[outputSize[0] - 1, 0],[outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result
textData=[]
outputCanvas = np.full(image.shape[:3], 255, dtype=np.uint8)
outputCanvas1 = np.full(image.shape[:3], 255, dtype=np.uint8)
outputCanvas2 = np.full(image.shape[:3], 255, dtype=np.uint8)
print("Recognized Text:")
for box in boxes:
    croppedRoi = fourPointsTransform(image, box)
    recResult = textRecognizer.recognize(croppedRoi)
    boxHeight = int((abs((box[0, 1] - box[1, 1]))))
    fontScale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, boxHeight - 30, 1)
    placement = (int(box[0, 0]), int(box[0, 1]))
    cv2.putText(outputCanvas, recResult, placement,cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 1, 5)
    textData.append(recResult)
for box in boxes:
    croppedRoi = fourPointsTransform(image, box)
    recResult = textRecognizer.recognize(croppedRoi)
    boxHeight = int((abs((box[0, 1] - box[1, 1]))))
    fontScale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, boxHeight - 10, 1)
    placement = (int(box[0, 0]), int(box[0, 1]))
    cv2.putText(outputCanvas1, recResult, placement,cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 1, 5)
    textData.append(recResult)
for box in boxes:
    croppedRoi = fourPointsTransform(image, box)
    recResult = textRecognizer.recognize(croppedRoi)
    boxHeight = int((abs((box[0, 1] - box[1, 1]))))
    fontScale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, boxHeight, 1)
    placement = (int(box[0, 0]), int(box[0, 1]))
    cv2.putText(outputCanvas2, recResult, placement,cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 1, 5)
    textData.append(recResult)

#textData = ' '.join(textData)
#print(textData);
top = np.hstack([image,outputCanvas])
bot = np.hstack([outputCanvas1,outputCanvas2])
tot = np.vstack([top,bot])
etot = cv2.resize(tot,None,fx= 0.4,fy = 0.4)
plt.imshow(etot[:, :, ::-1])
plt.show()