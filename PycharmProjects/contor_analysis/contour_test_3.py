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
contours_sorted = sorted(contours,key=cv2.contourArea,reverse=True)
contour_image = np.copy(image)
#x1 = None
#y1 = None
#x2 = None
#y2 = None
for cnt in range(2):
    x,y,w,h = cv2.boundingRect(contours_sorted[cnt])
    if cnt == 0:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
    else:
        x1 = min(x1,x)
        y1 = min(y1,y)
        x2 = max(x+w,x2)
        y2 = max(y+h,y2)
cv2.rectangle(contour_image,(x1,y1), (x2,y2),(255,0,0),4)
#cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
draw(contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
