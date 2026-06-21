"""Validation for the polygon model. The reported metric is the bounding-box F1-score."""

import torch

from ..utils import LOGGER
from ..utils.metrics import box_iou
from ..utils.ops import non_max_suppression, xywh2xyxy


class PolygonValidator:
    """Compute bounding-box precision / recall / F1 over a polygon dataloader."""

    def __init__(self, model, dataloader, device="cpu", conf=0.25, iou=0.5, max_det=300,
                 names=None, num_angles=None, plot_samples=False, sample_prefix=None):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.conf = conf
        self.iou = iou  # IoU threshold for a true positive
        self.max_det = max_det
        self.nc = model.nc
        self.names = names or getattr(model, "names", None) or {i: str(i) for i in range(self.nc)}
        self.num_angles = num_angles or getattr(model, "num_angles", 24)
        self.plot_samples = plot_samples
        self.sample_prefix = sample_prefix

    @torch.no_grad()
    def __call__(self):
        tp = fp = fn = 0
        for bi, batch in enumerate(self.dataloader):
            imgs = batch["img"].to(self.device)
            _, _, h, w = imgs.shape
            out = self.model(imgs)
            y = out[0] if isinstance(out, tuple) else out
            preds = non_max_suppression(y[:, : 4 + self.nc], self.conf, 0.45, max_det=self.max_det, nc=self.nc)

            if bi == 0 and self.plot_samples and self.sample_prefix:
                self._plot_samples(batch, imgs, preds)

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

    def _plot_samples(self, batch, imgs, preds):
        """Save GT (bbox + star polygon) and prediction (bbox) grids for the first val batch."""
        import cv2
        import numpy as np
        from pathlib import Path

        from ..utils.plotting import draw_poly_star, image_grid, plot_detections

        n = min(9, imgs.shape[0])
        gt_tiles, pred_tiles = [], []
        for si in range(n):
            arr = (imgs[si].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")[:, :, ::-1]
            arr = np.ascontiguousarray(arr)
            idx = batch["batch_idx"] == si
            gt_tiles.append(draw_poly_star(arr.copy(), batch["cls"][idx].numpy().reshape(-1),
                                           batch["bboxes"][idx].numpy(), batch["poly"][idx].numpy(),
                                           self.num_angles, self.names))
            det = preds[si]
            pb = det.cpu().numpy() if det is not None and len(det) else None
            pred_tiles.append(plot_detections(arr.copy(), pb, names=self.names))
        Path(self.sample_prefix).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{self.sample_prefix}_labels.jpg", image_grid(gt_tiles, cols=3))
        cv2.imwrite(f"{self.sample_prefix}_pred.jpg", image_grid(pred_tiles, cols=3))
