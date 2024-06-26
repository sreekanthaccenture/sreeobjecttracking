import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.ones((300,300,3),dtype=np.uint8)*255
'''cv2.rectangle(img,(0,0),(300,50),(0,0,0),-1)
cv2.putText(img,"sreekanth",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
plt.figure(figsize=(20,8))
plt.imshow(img[:,:,::-1])
plt.show()'''
'''k = f'confidence:{int(45.9898)}'
print(k)'''
'''k = cv2.dnn.blobFromImage(img,1.0,(300,300),(1,2,3),False,False)'''
'''k = 0.78765
l = "carrot"
m = f'{round(k,2)}:{l}'
print(m)'''
a = (4,5)
b = (1,1)
d = pow((pow((b[0] - a[0]),2) + pow((b[1] - a[1]),2)),0.5)
print(d)