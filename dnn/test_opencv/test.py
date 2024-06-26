import cv2
import matplotlib.pyplot as plt
import numpy as np
img1_path = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data\driving_license\3.jpg"
img2_path = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data\driving_license\10.jpg"
img3_path = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data\driving_license\29.jpg"
img4_path = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data\driving_license\46.jpg"
screen = np.ones((600, 600, 3), dtype=np.uint8) * 255 # (height,width)
c = 0
def text(image,text):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2)
    cv2.rectangle(image, (0, 0),(0 + text_width + 5, 0 + text_height + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (0, 0 + text_height + int(baseline / 2)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return
def screens(screen,image):
    global c
    sc_height,sc_width, channel = screen.shape
    image = cv2.resize(image,(int(sc_width/2),int(sc_height/2)))
    if c == 0:
        screen[0:int(600/2),0:int(600/2)] = image
        c = c + 1
        return
    if c == 1:
        screen[int(600/2):600,0:int(600/2)] = image
        c = c + 1
        return
    if c == 2:
        screen[0:int(600/2),int(600/2):600] = image
        c = c + 1
        return
    if c == 3:
        screen[int(600/2):600,int(600/2):600] = image
        c = c + 1
img1 = cv2.imread(img1_path,cv2.IMREAD_COLOR)
text(img1,"alpha")
screens(screen,img1)
plt.imshow(screen)
plt.show()

