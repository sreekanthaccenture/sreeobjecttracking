import cv2
import matplotlib.pyplot as plt
import numpy as np


def headline(frame,name):
    height,width,channel = frame.shape
    top_coordinates = (0,0)
    bot_coordinates = (width,int(height*0.2))
    print(top_coordinates,bot_coordinates)
    cv2.rectangle(frame, top_coordinates, bot_coordinates, (0, 0, 0), thickness=-1,lineType=cv2.LINE_AA)
    cv2.putText(frame, name, (0,int(bot_coordinates[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    #plt.figure(figsize=(10,6))
    #plt.imshow(frame)
    #plt.show()
    return frame

x = cv2.imread(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module01-getting-started-with-images\Eagle_in_Flight.jpg",cv2.IMREAD_COLOR)
z_left = headline(x,"sreekanth")
z_right = headline(x,"rama")
#com_pic = np.hstack([z_left,z_right])
com_pic = np.hstack([z_left,z_right])
height,width,x = com_pic.shape
print(height,width)
horrline = cv2.line(com_pic,(int(width/2),0), (int(width/2),height),(255,0,0),3,cv2.LINE_AA)
plt.imshow(horrline)
plt.show()
#cv2.putText(x,"hello",(45,300),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,lineType=cv2.LINE_AA)
#cv2.rectangle(x,(300,150),(480,420),(0,255,0),thickness=-1,lineType=cv2.LINE_AA)
#plt.figure(figsize=(10,6))
#plt.imshow(z)
#plt.show()
#fig,ax = plt.subplots(1,2,figsize=(10,6))
#ax[0].imshow(x);ax[0].set_title('hello')
#ax[1].imshow(y)
#plt.show()