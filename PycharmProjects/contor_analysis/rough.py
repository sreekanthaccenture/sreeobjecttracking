import cv2
import matplotlib.pyplot as plt
import numpy as np

def headline(frame,name):
    height,width,channel = frame.shape
    top_coordinates = (0,0)
    bot_coordinates = (width,int(height*0.1))
    print(top_coordinates,bot_coordinates)
    cv2.rectangle(frame, top_coordinates, bot_coordinates, (0, 0, 0), thickness=-1,lineType=cv2.LINE_AA)
    cv2.putText(frame, name, (0,int(bot_coordinates[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

def views(frame,ans,videocap):
    w = int(videocap.get(3))
    h = int(videocap.get(4))
    images = np.zeros(frame.shape,np.uint8)
    smller_frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)
    smller_ans = cv2.resize(ans, (0, 0), fx=0.5, fy=0.5)
    images[0:int(h//2),0:int(w//2)] = smller_frame
    images[int(h // 2):h, int(w // 2):w] = smller_ans
    cv2.imshow('total', images)


class framedetails():
    def __init__(self,obj):
        self.framew = int(obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameh = int(obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framefps = int(obj.get(cv2.CAP_PROP_FPS))
        self.fourc = cv2.VideoWriter_fourcc(*'mp4v')
        self.filename =  'intrudervt.mp4'

videocap = cv2.VideoCapture(r"C:\Users\sreekanth.maramreddy\Desktop\ds\module04-video-processing-and-analysis\motion_test_original.mp4")
if(videocap.isOpened() == False):
    print("error opening file")

c = framedetails(videocap)
out_mp4 = cv2.VideoWriter(c.filename,c.fourc,int(c.framefps),(c.framew,c.frameh))

framecount = 0
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)
ksize = (5,5)
while True:
    ok,frame = videocap.read()
    cv2.imshow('output', frame)
    if not ok:
        break
    else:
        y = frame.copy()
    fg_mask = bg_sub.apply(frame)
    cv2.imshow('output', fg_mask)
    fg_mask_erode = cv2.erode(fg_mask,np.ones(ksize,np.uint8))
    cv2.imshow('output', fg_mask_erode)
    motion_area = cv2.findNonZero(fg_mask_erode)
    x,y,w,h = cv2.boundingRect(motion_area)
    if motion_area is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=6, lineType=cv2.LINE_AA)
    cv2.imshow('output', frame)
    fg_mask_erode_color = cv2.cvtColor(fg_mask_erode, cv2.COLOR_GRAY2BGR)
    cv2.imshow('output', fg_mask_erode_color)
    headline(fg_mask_erode_color,"foreground mask")
    cv2.imshow('output', fg_mask_erode_color)
    #g = views(frame,fg_mask_erode_color,videocap)
    smller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    smller_ans = cv2.resize(fg_mask_erode_color, (0, 0), fx=0.5, fy=0.5)
    com_pic = np.hstack([smller_ans,smller_frame])
    cv2.imshow('sample', com_pic)
    #height, width, x = com_pic.shape
    #horrline = cv2.line(com_pic, (int(width / 2), 0), (int(width / 2), height), (255, 0, 0), 3, cv2.LINE_AA)
    #cv2.imshow('output', horrline)
    #out_mp4.write(g)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

videocap.release()
out_mp4.release()