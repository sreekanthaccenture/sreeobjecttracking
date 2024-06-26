import cv2
import matplotlib.pyplot as plt
import numpy as np
def draw(img):
    plt.imshow(img)
    plt.show()
image = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\shapes.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = np.copy(image)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_image,(x,y), (x+w,y+h),(255,0,0),4)
#cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
draw(contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
