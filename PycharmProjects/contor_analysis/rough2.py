import numpy as np
'''def iou(box1,box2):
    x1,y1,x2,y2 = box1
    x3,y3,x4,y4 = box2
    xinter1 = max(x1,x3)
    yinter1 = max(y1,y3)
    xinter2 = min(x2,x4)
    yinter2 = min(y2,y4)
    width = max(0,xinter2-xinter1)
    height = max(0,yinter2-yinter1)
    interarea = width*height
    if interarea == 0:
        return 0
    box1area = abs((x2-x1)*(y2-y1))
    box2area = abs((x4-x3)*(y4-y3))
    totalarea = box1area+box2area - interarea
    return interarea/totalarea
def nms(boxes,scores,th):
    indices = np.argsort(scores)[::-1]
    print(indices)
    output_indices = []
    while len(indices)>0:
        currentindice = indices[0]
        print(currentindice)
        output_indices.append(currentindice)
        print( output_indices)
        indices = indices[1:]
        print(indices)
        indices = [i for i in indices if iou(boxes[currentindice],boxes[i]) < th]
        print(indices)
    return output_indices
boxes = np.array([[50, 50, 100, 100], [55, 55, 90, 90], [60, 60, 110, 110],[300, 300, 100, 100], [310, 310, 90, 90], [320, 320, 110, 110]])
scores = np.array([0.9, 0.85, 0.8, 0.95, 0.9, 0.7])
iou_threshold = 0.5
print(nms(boxes,scores,iou_threshold))'''
k = [[1,23],[1,23],[1,23]]
print(len(k))