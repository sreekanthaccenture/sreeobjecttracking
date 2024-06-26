import numpy as np

def iou(box1, box2):
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    if intersection == 0:
        return 0.0

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union

def non_maximum_suppression(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(indices) > 0:
        current = indices[0]
        selected_indices.append(current)
        indices = indices[1:]
        indices = [i for i in indices if iou(boxes[current], boxes[i]) < iou_threshold]

    return selected_indices

# Example bounding boxes and scores
boxes = np.array([[50, 50, 150, 150], [55, 55, 145, 145], [60, 60, 170, 170],
                  [300, 300, 400, 400], [310, 310, 400, 400], [320, 320, 430, 430]])
scores = np.array([0.9, 0.85, 0.8, 0.95, 0.9, 0.7])

# Apply NMS
iou_threshold = 0.5
selected_indices = non_maximum_suppression(boxes, scores, iou_threshold)

# Output final bounding boxes
final_boxes = boxes[selected_indices]
print(final_boxes)
