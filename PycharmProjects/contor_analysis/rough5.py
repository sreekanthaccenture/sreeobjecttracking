import torch

def intersection_over_union(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    x_inter_min = torch.max(x1_min, x2_min)
    y_inter_min = torch.max(y1_min, y2_min)
    x_inter_max = torch.min(x1_max, x2_max)
    y_inter_max = torch.min(y1_max, y2_max)
    inter_width = torch.max(torch.tensor(0.0), x_inter_max - x_inter_min)
    inter_height = torch.max(torch.tensor(0.0), y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else torch.tensor(0.0)
    return iou


def calculate_precision_recall(detections,gboxcount,thr):
    detections = sorted(detections,key = lambda x:x[0],reverse=True)
    tp= 0
    fp = 0
    precisionrecalllist = []
    for detection in detections:
        if detection[1] >= thr:
            tp += 1
        else:
            fp += 1
        precision = tp/(tp+fp)
        recall = tp/gboxcount
        precisionrecalllist.append((precision,recall))
    return precisionrecalllist

def calculate_ap(precision_recall):

    precisions = [pr[0] for pr in precision_recall]
    recalls = [pr[1] for pr in precision_recall]

    # Interpolated precision
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Compute AP
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i - 1])

    return ap


def map(gboxes,pboxes,pscore,thr):
    detections = []
    for i in range(len(pboxes)):
        best = 0.0
        for j in range(len(gboxes)):
            iouu = intersection_over_union(pboxes[i],gboxes[j]).item()
            best = max(best,iouu)
        detections.append((pscore[i],best))
    precision_recall = calculate_precision_recall(detections, len(gboxes), thr)
    ap = calculate_ap(precision_recall)
    return ap

ground_truth_dboxes = torch.tensor([[8, 12, 352, 498], [10, 15, 450, 500]], dtype=torch.float)
predicted_dboxes = torch.tensor([
    [1.000000, 116.384613, 353.000000, 116.384613], [1.000000, 1.000000, 353.000000, 500.000000],
    [9.000000, 14.000000, 452.500000, 500.500000], [1.000000, 154.846161, 353.000000, 154.846161],
    [196.776474, 231.769226, 196.776474, 231.769226], [1.000000, 116.384613, 353.000000, 116.384613],
    [1.000000, 1.000000, 353.000000, 500.000000], [1.000000, 231.769226, 353.000000, 231.769226],
    [45.000000, 235.230774, 175.000000, 361.230774], [44.500000, 234.230774, 176.000000, 362.230774]
], dtype=torch.float)
predicted_dbox_score = torch.tensor(
    [0.180000, 0.985000, 0.98000, 0.200000, 0.090000, 0.170000, 0.570000, 0.210000, 0.855000, 0.850000],
    dtype=torch.float)
iou_threshold = 0.5
x = map(ground_truth_dboxes,predicted_dboxes,predicted_dbox_score,iou_threshold)
print(f"AP: {x:.2f}")