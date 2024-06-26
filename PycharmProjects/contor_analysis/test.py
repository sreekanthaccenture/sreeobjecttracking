import cv2
import matplotlib.pyplot as plt
import numpy as np

class framedetails():
    def __init__(self,obj):
        self.framew = int(obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameh = int(obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framefps = int(obj.get(cv2.CAP_PROP_FPS))
        self.fourc = cv2.VideoWriter_fourcc(*'mp4v')
        self.filename =  'new_video_out.mp4'

def banner(frame,name):
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

def views(frame,ans,videocap):
    w = int(videocap.get(3))
    h = int(videocap.get(4))
    images = np.zeros(frame.shape,np.uint8)
    smller_frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)
    smller_ans = cv2.resize(ans, (0, 0), fx=0.5, fy=0.5)
    images[0:int(h//2),0:int(w//2)] = smller_frame
    images[int(h // 2):h, int(w // 2):w] = smller_ans
    cv2.imshow('total', images)


videocap = cv2.VideoCapture(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module04-video-processing-and-analysis\race_car.mp4")
if(videocap.isOpened() == False):
    print("error opening file")

c = framedetails(videocap)
out_mp4 = cv2.VideoWriter(c.filename,c.fourc,int(c.framefps/3),(c.framew,c.frameh))

framecount = 0
while True:
    ok,frame = videocap.read()
    cv2.imshow('frame', frame)
    if not ok:
        break
    framecount = framecount +1
    text = 'FRAME' + str(int(framecount)) + str(c.framefps)
    ans = banner(frame,text)
    cv2.imshow('ans', ans)
    out_mp4.write(ans)
    views(frame,ans,videocap)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
videocap.release()
out_mp4.release()


