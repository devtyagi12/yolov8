"""Bounding-box metrics: IoU variants and mAP computation."""

import numpy as np
import torch


def box_iou(box1, box2, eps=1e-7):
    """IoU between every pair of boxes. ``box1`` (N,4), ``box2`` (M,4) xyxy -> (N,M)."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    area1 = (a2 - a1).prod(2)
    area2 = (b2 - b1).prod(2)
    return inter / (area1 + area2 - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """IoU (and CIoU/DIoU/GIoU) between aligned boxes. Both broadcastable to (..., 4)."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = (4 / np.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def compute_ap(recall, precision):
    """Compute average precision given monotone recall / precision arrays."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2.0 renamed trapz
    ap = trapezoid(np.interp(x, mrec, mpre), x)  # 101-point interpolation (COCO)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute the average precision per class.

    Args:
        tp (np.ndarray): (n, 10) boolean true-positive matrix across IoU thresholds.
        conf (np.ndarray): (n,) predicted confidences.
        pred_cls (np.ndarray): (n,) predicted classes.
        target_cls (np.ndarray): (m,) ground-truth classes.
    Returns:
        dict with per-class precision/recall/ap and summary mAP values.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    ap = np.zeros((nc, tp.shape[1]))
    p = np.zeros(nc)
    r = np.zeros(nc)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # labels
        n_p = i.sum()  # predictions
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])
        r[ci] = recall[-1, 0]
        p[ci] = precision[-1, 0]

    return {
        "ap": ap,
        "ap50": ap[:, 0] if ap.size else ap,
        "map50": ap[:, 0].mean() if ap.size else 0.0,
        "map": ap.mean() if ap.size else 0.0,
        "mp": p.mean() if nc else 0.0,
        "mr": r.mean() if nc else 0.0,
        "classes": unique_classes,
    }


def match_predictions(pred_classes, true_classes, iou, iou_thresholds):
    """Match predictions to ground truth across IoU thresholds -> (n_pred, n_thr) bool tp."""
    correct = np.zeros((pred_classes.shape[0], iou_thresholds.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes[None, :]
    iou = iou * correct_class  # zero out wrong-class IoUs
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iou_thresholds):
        matches = np.nonzero(iou >= threshold)  # (gt_idx, pred_idx)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                ious = iou[matches[:, 0], matches[:, 1]]
                matches = matches[ious.argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct
