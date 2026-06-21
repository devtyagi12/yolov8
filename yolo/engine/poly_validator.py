"""Validation for the polygon model. The reported metric is the bounding-box F1-score."""

import torch

from ..utils import LOGGER
from ..utils.metrics import box_iou
from ..utils.ops import non_max_suppression, xywh2xyxy


class PolygonValidator:
    """Compute bounding-box precision / recall / F1 over a polygon dataloader."""

    def __init__(self, model, dataloader, device="cpu", conf=0.25, iou=0.5, max_det=300):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.conf = conf
        self.iou = iou  # IoU threshold for a true positive
        self.max_det = max_det
        self.nc = model.nc

    @torch.no_grad()
    def __call__(self):
        tp = fp = fn = 0
        for batch in self.dataloader:
            imgs = batch["img"].to(self.device)
            _, _, h, w = imgs.shape
            out = self.model(imgs)
            y = out[0] if isinstance(out, tuple) else out
            preds = non_max_suppression(y[:, : 4 + self.nc], self.conf, 0.45, max_det=self.max_det, nc=self.nc)

            for si, pred in enumerate(preds):
                idx = batch["batch_idx"] == si
                gt_cls = batch["cls"][idx].squeeze(-1).to(self.device)
                gt_box = batch["bboxes"][idx].to(self.device)
                ng = gt_box.shape[0]
                npr = pred.shape[0]
                if npr == 0:
                    fn += ng
                    continue
                if ng == 0:
                    fp += npr
                    continue
                tbox = xywh2xyxy(gt_box) * torch.tensor([w, h, w, h], device=self.device)
                ious = box_iou(tbox, pred[:, :4])  # (ng, npr)
                matched_gt, matched_pr = set(), set()
                order = ious.flatten().argsort(descending=True)
                for flat in order.tolist():
                    gi, pi = divmod(flat, npr)
                    if ious[gi, pi] < self.iou:
                        break
                    if gi in matched_gt or pi in matched_pr:
                        continue
                    if int(gt_cls[gi]) != int(pred[pi, 5]):
                        continue
                    matched_gt.add(gi)
                    matched_pr.add(pi)
                tp += len(matched_pr)
                fp += npr - len(matched_pr)
                fn += ng - len(matched_gt)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        LOGGER.info(f"   bbox P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
