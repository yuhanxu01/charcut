import numpy as np
from scipy.optimize import linear_sum_assignment

def mask_iou_np(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / union

def hungarian_match(pred_masks, gt_masks, max_cost=0.99):
    P, G = len(pred_masks), len(gt_masks)
    if P == 0 or G == 0:
        return []
    cost = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            cost[i, j] = 1.0 - mask_iou_np(pred_masks[i], gt_masks[j])
    r, c = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(r, c):
        if cost[i, j] < max_cost:
            pairs.append((i, j))
    return pairs
