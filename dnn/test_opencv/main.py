import cv2
import matplotlib.pyplot as plt
import numpy as np

image = np.ones((400, 600, 3), dtype=np.uint8) * 255 # (height,width)
height,width,channels = image.shape
#cv2.line(image, (10, 10), (200, 10), (255, 0, 0), 5) #(x,y)
#cv2.rectangle(image, (50, 50), (150, 200), (0, 255, 0), 3)
#cv2.circle(image, (100, 100), 50, (0, 0, 255), -1)
#cv2.ellipse(image, (100, 125), (75, 50), 90, 0, 360, (255, 255, 0), 2)
#cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
#cv2.putText(image, "sree", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
(text_width, text_height), baseline = cv2.getTextSize("ram", cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2)
print(baseline)
cv2.rectangle(image, (int(width/2), int(height/2)), (int(width/2)+text_width+5, int(height/2)+text_height+baseline), (0, 0, 0), -1)
cv2.putText(image, "sree", (int(width/2), int(height/2)+text_height+int(baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
plt.imshow(image)
plt.show()
