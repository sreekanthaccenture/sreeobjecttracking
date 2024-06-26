import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\Applications\boy.jpg", cv2.IMREAD_COLOR)
modelFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\model\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(prototxt=configFile, caffeModel=modelFile)
def blur(face, factor=3):
    h, w = face.shape[:2]

    if factor < 1: factor = 1  # Maximum blurring
    if factor > 5: factor = 5  # Minimal blurring

    # Kernel size.
    w_k = int(w / factor)
    h_k = int(h / factor)

    # Insure kernel is an odd number.
    if w_k % 2 == 0: w_k += 1
    if h_k % 2 == 0: h_k += 1

    blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred


def face_blur_rect(image, net, factor=3, detection_threshold=0.9):
    img = image.copy()

    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])

    # Pass the blob to the DNN model.
    net.setInput(blob)

    # Retrieve detections from the DNN model.
    detections = net.forward()

    (h, w) = img.shape[:2]

    # Process the detetcions.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract the face ROI.
            face = img[y1:y2, x1:x2]

            face = blur(face, factor=factor)

            # Replace the detected face with the blurred one.
            img[y1:y2, x1:x2] = face

    return img
img1_rect = face_blur_rect(img1, net, factor=2.5)

fig = plt.figure(figsize=(15,10))
plt.subplot(1,2,1); plt.axis('off'); plt.imshow(img1[:,:,::-1]);      plt.title('Original')
plt.subplot(1,2,2); plt.axis('off'); plt.imshow(img1_rect[:,:,::-1]); plt.title('Rectangular Blur');
plt.show()
'''
def face_blur_ellipse(image, net, factor=3, detect_threshold=0.90, write_mask=False):
    
    img = image.copy()
    img_blur = img.copy()
    
    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
    
    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300,300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detect_threshold:

            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # The face is defined by the bounding rectangle from the detection.
            face = img[int(y1):int(y2), int(x1):int(x2), :]
           
            # Blur the rectangular area defined by the bounding box.
            face = blur(face, factor=factor)

            # Copy the `blurred_face` to the blurred image.
            img_blur[int(y1):int(y2), int(x1):int(x2), :] = face
            
            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1)/2, y1 + (y2 - y1)/2)
            e_size   = (x2 - x1, y2 - y1)
            e_angle  = 0.0
            
            # Create an elliptical mask. 
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), 
                                                      (255, 255, 255), -1, cv2.LINE_AA)  
            # Apply the elliptical mask
            np.putmask(img, elliptical_mask, img_blur)
            
    if write_mask:
        cv2.imwrite('elliptical_mask.jpg', elliptical_mask)

    return img
'''
'''
img1_ellipse = face_blur_ellipse(img1, net, factor=2.5, write_mask=True)

mask = cv2.imread('elliptical_mask.jpg')
fig = plt.figure(figsize=(20,10))
plt.subplot(1,3,1); plt.axis('off'); plt.imshow(img1[:,:,::-1]);         plt.title('Original')
plt.subplot(1,3,2); plt.axis('off'); plt.imshow(mask);                   plt.title('Elliptical Mask')
plt.subplot(1,3,3); plt.axis('off'); plt.imshow(img1_ellipse[:,:,::-1]); plt.title('Elliptical Blur');
'''