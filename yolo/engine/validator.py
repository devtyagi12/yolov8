"""Detection validator: COCO-style mAP, per-class table, confusion matrix, curves, speed."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..utils import LOGGER
from ..utils.metrics import ConfusionMatrix, ap_per_class, box_iou, match_predictions
from ..utils.ops import non_max_suppression, xywh2xyxy
from ..utils.plotting import plot_confusion_matrix, plot_mc_curve, plot_pr_curve


class DetectionValidator:
    """Run detection validation and report rich metrics (optionally saving plots/JSON)."""

    def __init__(self, model, dataloader, device="cpu", conf=0.001, iou=0.7, max_det=300,
                 names=None, save_dir=None, plots=False, save_json=False, plot_samples=False,
                 sample_prefix=None, sample_conf=0.25):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.nc = model.nc
        self.names = names or getattr(model, "names", None) or {i: str(i) for i in range(self.nc)}
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)  # IoU thresholds for mAP
        self.save_dir = Path(save_dir) if save_dir else None
        self.plots = plots
        self.save_json = save_json
        self.plot_samples = plot_samples            # save GT vs prediction images for the first batch
        self.sample_prefix = sample_prefix          # path prefix for "<prefix>_labels.jpg" / "<prefix>_pred.jpg"
        self.sample_conf = sample_conf              # confidence threshold for the prediction image
        self.confusion = ConfusionMatrix(self.nc, conf=0.25, iou_thres=0.45)
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    @torch.no_grad()
    def __call__(self):
        stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        n_imgs = 0
        for bi, batch in enumerate(self.dataloader):
            t0 = time.perf_counter()
            imgs = batch["img"].to(self.device)
            nb, _, h, w = imgs.shape
            n_imgs += nb
            t1 = time.perf_counter()
            out = self.model(imgs)
            out = out[0] if isinstance(out, (list, tuple)) else out
            t2 = time.perf_counter()
            preds = non_max_suppression(out, self.conf, self.iou, max_det=self.max_det, nc=self.nc)
            t3 = time.perf_counter()
            self.speed["preprocess"] += (t1 - t0) * 1e3
            self.speed["inference"] += (t2 - t1) * 1e3
            self.speed["postprocess"] += (t3 - t2) * 1e3

            if bi == 0 and self.plot_samples and self.sample_prefix:
                self._plot_samples(batch, imgs, preds)

            for si, pred in enumerate(preds):
                idx = batch["batch_idx"] == si
                cls = batch["cls"][idx].squeeze(-1)
                bbox = batch["bboxes"][idx]
                npr = pred.shape[0]
                tbox = (
                    xywh2xyxy(bbox) * torch.tensor([w, h, w, h], device=self.device)
                    if cls.shape[0] else torch.zeros((0, 4), device=self.device)
                )
                self.confusion.process_batch(
                    pred.cpu().numpy() if npr else None, tbox.cpu().numpy(), cls.cpu().numpy()
                )

                if npr == 0:
                    if cls.shape[0]:
                        stats["tp"].append(torch.zeros(0, self.iouv.numel(), dtype=torch.bool))
                        stats["conf"].append(torch.zeros(0))
                        stats["pred_cls"].append(torch.zeros(0))
                        stats["target_cls"].append(cls)
                    continue

                if cls.shape[0]:
                    iou = box_iou(tbox.to(self.device), pred[:, :4])
                    correct = torch.from_numpy(
                        match_predictions(pred[:, 5].cpu().numpy(), cls.cpu().numpy(), iou, self.iouv.cpu().numpy())
                    )
                else:
                    correct = torch.zeros(npr, self.iouv.numel(), dtype=torch.bool)

                stats["tp"].append(correct)
                stats["conf"].append(pred[:, 4].cpu())
                stats["pred_cls"].append(pred[:, 5].cpu())
                stats["target_cls"].append(cls.cpu())

        return self._summarize(stats, n_imgs)

    def _summarize(self, stats, n_imgs):
        for k in self.speed:
            self.speed[k] = self.speed[k] / max(n_imgs, 1)

        if not stats["tp"] or torch.cat(stats["target_cls"]).numel() == 0:
            LOGGER.info("No predictions/targets accumulated during validation.")
            return {"map50": 0.0, "map": 0.0, "mp": 0.0, "mr": 0.0, "mf1": 0.0, "speed": self.speed}

        tp = torch.cat(stats["tp"]).numpy()
        conf = torch.cat(stats["conf"]).numpy()
        pred_cls = torch.cat(stats["pred_cls"]).numpy()
        target_cls = torch.cat(stats["target_cls"]).numpy()
        res = ap_per_class(tp, conf, pred_cls, target_cls)

        # Overall + per-class table.
        LOGGER.info(f"{'Class':>16}{'Images':>8}{'Labels':>8}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>9}{'mAP50-95':>10}")
        LOGGER.info(f"{'all':>16}{n_imgs:>8}{int(res['nt'].sum()):>8}"
                    f"{res['mp']:>8.3f}{res['mr']:>8.3f}{res['mf1']:>8.3f}{res['map50']:>9.3f}{res['map']:>10.3f}")
        per_class = {}
        for i, c in enumerate(res["classes"]):
            name = self.names.get(int(c), str(int(c)))
            LOGGER.info(f"{name:>16}{n_imgs:>8}{int(res['nt'][i]):>8}"
                        f"{res['p'][i]:>8.3f}{res['r'][i]:>8.3f}{res['f1'][i]:>8.3f}"
                        f"{res['ap50'][i]:>9.3f}{res['ap_class'][i]:>10.3f}")
            per_class[name] = {"p": float(res["p"][i]), "r": float(res["r"][i]), "f1": float(res["f1"][i]),
                               "map50": float(res["ap50"][i]), "map": float(res["ap_class"][i])}
        LOGGER.info(f"Speed (per image): {self.speed['preprocess']:.2f}ms pre, "
                    f"{self.speed['inference']:.2f}ms inference, {self.speed['postprocess']:.2f}ms post")

        if self.plots and self.save_dir is not None:
            self._save_plots(res)
        if self.save_json and self.save_dir is not None:
            self._save_json({"map50": float(res["map50"]), "map": float(res["map"]),
                             "mp": float(res["mp"]), "mr": float(res["mr"]), "per_class": per_class})

        return {
            "map50": float(res["map50"]), "map": float(res["map"]),
            "mp": float(res["mp"]), "mr": float(res["mr"]), "mf1": float(res["mf1"]),
            "per_class": per_class, "speed": self.speed, "confusion_matrix": self.confusion.matrix,
        }

    def _plot_samples(self, batch, imgs, preds):
        """Save side-by-side ground-truth and prediction grids for the first val batch."""
        import cv2

        from ..utils.plotting import draw_normalized, image_grid, plot_detections

        n = min(9, imgs.shape[0])
        gt_tiles, pred_tiles = [], []
        for si in range(n):
            arr = (imgs[si].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")[:, :, ::-1]
            arr = np.ascontiguousarray(arr)
            idx = batch["batch_idx"] == si
            gt_tiles.append(draw_normalized(arr.copy(), batch["cls"][idx].numpy().reshape(-1),
                                            batch["bboxes"][idx].numpy(), self.names))
            det = preds[si]
            pb = None
            if det is not None and len(det):
                det = det[det[:, 4] > self.sample_conf]  # confident predictions only
                pb = det.cpu().numpy() if len(det) else None
            pred_tiles.append(plot_detections(arr.copy(), pb, names=self.names))
        Path(self.sample_prefix).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{self.sample_prefix}_labels.jpg", image_grid(gt_tiles, cols=3))
        cv2.imwrite(f"{self.sample_prefix}_pred.jpg", image_grid(pred_tiles, cols=3))

    def _save_plots(self, res):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        c = res["curves"]
        plot_pr_curve(c["px"], c["prec"], res["ap"], self.names, str(self.save_dir / "PR_curve.png"))
        plot_mc_curve(c["px"], c["f1"], self.names, str(self.save_dir / "F1_curve.png"), ylabel="F1")
        plot_mc_curve(c["px"], c["p"], self.names, str(self.save_dir / "P_curve.png"), ylabel="Precision")
        plot_mc_curve(c["px"], c["r"], self.names, str(self.save_dir / "R_curve.png"), ylabel="Recall")
        plot_confusion_matrix(self.confusion.matrix, self.names, str(self.save_dir / "confusion_matrix.png"))
        LOGGER.info(f"Saved validation plots to {self.save_dir}")

    def _save_json(self, summary):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info(f"Saved metrics to {self.save_dir / 'metrics.json'}")
