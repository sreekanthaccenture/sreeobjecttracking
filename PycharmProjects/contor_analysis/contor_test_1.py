import cv2
import matplotlib.pyplot as plt
import numpy as np
def draw(img):
    plt.imshow(img)
    plt.imshow
def plott(img1,img2):
    h,w,v = img1.shape
    images = np.zeros(img1.shape, np.uint8)
    smller_frame = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    smller_ans = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    images[0:int(h // 2), 0:int(w // 2)] = smller_frame
    images[0:int(h // 2), int(w // 2):w] = smller_ans
    cv2.imshow('total', images)
image = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\shapes.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours1,hierarchy1 = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour_image1 = np.copy(image)
contour_image2 = np.copy(image)
cv2.drawContours(contour_image1, contours, -1, (0, 255, 0), 2)
cv2.drawContours(contour_image2, contours, -1, (0, 255, 0), 2)
plott(contour_image1,contour_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
