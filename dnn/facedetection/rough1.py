import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module14-face-detection\visuals\c0-m14-NB-img1.jpg", cv2.IMREAD_COLOR)
'''x1 = 210
y1 = 54
x2 = 325
y2 = 241'''
'''cv2.rectangle(img1, (210, 54), (325, 214), (0, 255, 0), 2)
plt.imshow(img1[:,:,::-1])
plt.show()'''
'''elliptical_mask = np.zeros(img1.shape, dtype=img1.dtype)
e_center = (x1 + (x2 - x1)/2, y1 + (y2 - y1)/2)
e_size   = (x2 - x1, y2 - y1)
e_angle  = 0.0
elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle),(255, 255, 255), -1, cv2.LINE_AA)
plt.imshow(elliptical_mask)
plt.show()'''


def pixelate(roi, pixels=16):
    # Size of region to pixelate.
    roi_h, roi_w = roi.shape[:2]

    if roi_h > pixels and roi_w > pixels:
        # Resize input ROI to the (small) pixelated size.
        roi_small = cv2.resize(roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)

        # Now enlarge the pixelated ROI to fill the size of the original ROI.
        roi_pixelated = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_pixelated = roi

    return roi_pixelated
roi = img1[54:241,210:325]
z = pixelate(roi)
plt.imshow(z)
plt.show()
