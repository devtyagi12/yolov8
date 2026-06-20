"""Detection validator: computes mAP over a dataloader."""

import numpy as np
import torch

from ..utils import LOGGER
from ..utils.metrics import ap_per_class, box_iou, match_predictions
from ..utils.ops import non_max_suppression, scale_boxes, xywh2xyxy


class DetectionValidator:
    """Run detection validation and compute COCO-style mAP@0.5 and mAP@0.5:0.95."""

    def __init__(self, model, dataloader, device="cpu", conf=0.001, iou=0.7, max_det=300):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.nc = model.nc
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)  # IoU thresholds for mAP

    @torch.no_grad()
    def __call__(self):
        stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        for batch in self.dataloader:
            imgs = batch["img"].to(self.device)
            nb, _, h, w = imgs.shape
            preds = self.model(imgs)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            preds = non_max_suppression(preds, self.conf, self.iou, max_det=self.max_det, nc=self.nc)

            for si, pred in enumerate(preds):
                idx = batch["batch_idx"] == si
                cls = batch["cls"][idx].squeeze(-1)
                bbox = batch["bboxes"][idx]  # normalised xywh
                npr = pred.shape[0]

                if npr == 0:
                    if cls.shape[0]:
                        stats["tp"].append(torch.zeros(0, self.iouv.numel(), dtype=torch.bool))
                        stats["conf"].append(torch.zeros(0))
                        stats["pred_cls"].append(torch.zeros(0))
                        stats["target_cls"].append(cls)
                    continue

                if cls.shape[0]:
                    tbox = xywh2xyxy(bbox) * torch.tensor([w, h, w, h], device=self.device)
                    iou = box_iou(tbox.to(self.device), pred[:, :4])
                    correct = match_predictions(pred[:, 5].cpu().numpy(), cls.cpu().numpy(), iou, self.iouv.cpu().numpy())
                    correct = torch.from_numpy(correct)
                else:
                    correct = torch.zeros(npr, self.iouv.numel(), dtype=torch.bool)

                stats["tp"].append(correct)
                stats["conf"].append(pred[:, 4].cpu())
                stats["pred_cls"].append(pred[:, 5].cpu())
                stats["target_cls"].append(cls.cpu())

        return self._summarize(stats)

    def _summarize(self, stats):
        if not stats["tp"]:
            LOGGER.info("No predictions/targets accumulated during validation.")
            return {"map50": 0.0, "map": 0.0, "mp": 0.0, "mr": 0.0}
        tp = torch.cat(stats["tp"]).numpy()
        conf = torch.cat(stats["conf"]).numpy()
        pred_cls = torch.cat(stats["pred_cls"]).numpy()
        target_cls = torch.cat(stats["target_cls"]).numpy()
        if target_cls.shape[0] == 0:
            LOGGER.info("No ground-truth labels found; mAP undefined.")
            return {"map50": 0.0, "map": 0.0, "mp": 0.0, "mr": 0.0}

        res = ap_per_class(tp, conf, pred_cls, target_cls)
        LOGGER.info(
            f"   P={res['mp']:.3f}  R={res['mr']:.3f}  mAP50={res['map50']:.3f}  mAP50-95={res['map']:.3f}"
        )
        return {"map50": float(res["map50"]), "map": float(res["map"]), "mp": float(res["mp"]), "mr": float(res["mr"])}
