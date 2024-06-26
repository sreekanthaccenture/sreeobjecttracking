import torch


def intersection_over_union(box1, box2):

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    x_inter_min = torch.max(x1_min, x2_min)
    y_inter_min = torch.max(y1_min, y2_min)
    x_inter_max = torch.min(x1_max, x2_max)
    y_inter_max = torch.min(y1_max, y2_max)

    # Calculate the area of the intersection rectangle
    inter_width = torch.max(torch.tensor(0.0), x_inter_max - x_inter_min)
    inter_height = torch.max(torch.tensor(0.0), y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the area of the union
    union_area = box1_area + box2_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area if union_area != 0 else torch.tensor(0.0)

    return iou


def calculate_precision_recall(detections, ground_truth_count, iou_threshold):

    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    precision_recall = []

    for detection in detections:
        if detection[1] >= iou_threshold:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / ground_truth_count
        precision_recall.append((precision, recall))

    return precision_recall


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


def calculate_mAP(ground_truth_boxes, predicted_boxes, predicted_scores, iou_threshold):

    ground_truth_count = ground_truth_boxes.size(0)
    detections = []

    for i in range(predicted_boxes.size(0)):
        pred_box = predicted_boxes[i]
        score = predicted_scores[i]
        best_iou = 0.0

        for j in range(ground_truth_boxes.size(0)):
            gt_box = ground_truth_boxes[j]
            iou = intersection_over_union(pred_box, gt_box).item()
            best_iou = max(best_iou, iou)

        detections.append((score, best_iou))

    precision_recall = calculate_precision_recall(detections, ground_truth_count, iou_threshold)
    ap = calculate_ap(precision_recall)
    return ap


# Data
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

# Calculate mAP
ap = calculate_mAP(ground_truth_dboxes, predicted_dboxes, predicted_dbox_score, iou_threshold)
print(f"AP: {ap:.2f}")
