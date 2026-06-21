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


def smooth(y, f=0.05):
    """Box-filter smoothing of a 1D array (used to pick a stable max-F1 confidence)."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute per-class average precision plus P/R/F1 curves over a confidence sweep.

    Args:
        tp (np.ndarray): (n, niou) boolean true-positive matrix across IoU thresholds.
        conf (np.ndarray): (n,) predicted confidences.
        pred_cls (np.ndarray): (n,) predicted classes.
        target_cls (np.ndarray): (m,) ground-truth classes.
    Returns:
        dict with summary mAP, per-class P/R/F1/AP arrays, and curve data for plotting.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    px = np.linspace(0, 1, 1000)  # confidence sweep for curves
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    prec_values = []
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l, n_p = nt[ci], int(i.sum())
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc)
        # interpolate P/R against the confidence sweep (negative conf -> ascending)
        r_curve[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(px, mrec, mpre))
    prec_values = np.array(prec_values) if prec_values else np.zeros((0, 1000))

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    # pick the confidence that maximises mean F1
    idx = smooth(f1_curve.mean(0), 0.1).argmax() if nc else 0
    p, r, f1 = p_curve[:, idx], r_curve[:, idx], f1_curve[:, idx]

    return {
        "ap": ap,
        "ap50": ap[:, 0] if ap.size else ap,
        "ap_class": ap.mean(1) if ap.size else ap,  # per-class mAP@0.5:0.95
        "map50": ap[:, 0].mean() if ap.size else 0.0,
        "map": ap.mean() if ap.size else 0.0,
        "p": p,
        "r": r,
        "f1": f1,
        "mp": p.mean() if nc else 0.0,
        "mr": r.mean() if nc else 0.0,
        "mf1": f1.mean() if nc else 0.0,
        "nt": nt,
        "classes": unique_classes.astype(int),
        "conf_at_max_f1": float(px[idx]) if nc else 0.0,
        "curves": {"px": px, "p": p_curve, "r": r_curve, "f1": f1_curve, "prec": prec_values},
    }


class ConfusionMatrix:
    """Detection confusion matrix (``nc + 1`` rows/cols; the extra index is background)."""

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1))  # rows: predicted, cols: ground truth

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """Accumulate one image. ``detections`` (n,6) xyxy,conf,cls; ``gt_bboxes`` (m,4) xyxy."""
        if detections is not None and len(detections):
            detections = detections[detections[:, 4] > self.conf]
        gt_cls = gt_cls.astype(int)
        if detections is None or len(detections) == 0:
            for gc in gt_cls:
                self.matrix[self.nc, gc] += 1  # all GT become false negatives (background pred)
            return
        det_cls = detections[:, 5].astype(int)
        if len(gt_bboxes) == 0:
            for dc in det_cls:
                self.matrix[dc, self.nc] += 1  # background false positives
            return

        iou = box_iou(torch.as_tensor(gt_bboxes), torch.as_tensor(detections[:, :4])).numpy()
        x = np.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = np.stack(x, 1)
            v = iou[x[0], x[1]]
            matches = matches[v.argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 2), int)
        matched_gt = matches[:, 0].astype(int)
        matched_det = matches[:, 1].astype(int)
        m_lookup = {int(g): int(d) for g, d in matches[:, :2].astype(int)} if len(matches) else {}

        for gi, gc in enumerate(gt_cls):
            if gi in matched_gt:
                self.matrix[det_cls[m_lookup[gi]], gc] += 1  # TP (or class confusion)
            else:
                self.matrix[self.nc, gc] += 1  # missed GT
        for di, dc in enumerate(det_cls):
            if di not in matched_det:
                self.matrix[dc, self.nc] += 1  # background FP

    def tp_fp(self):
        tp = self.matrix.diagonal()[:-1]
        fp = self.matrix[:-1].sum(1) - tp
        return tp, fp



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
