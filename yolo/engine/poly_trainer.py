"""Trainer for the polygon + distance model.

Trains on the *merged* dataloader (polygon-only + polygon+distance, each with its
own batch size) using :class:`v8PolygonDistanceLoss`, and validates with the
bounding-box F1 metric.
"""

from pathlib import Path

import numpy as np
import torch

from ..data.poly_dataset import build_merged_dataloader, build_poly_dataloader
from ..utils import LOGGER
from ..utils.poly_loss import v8PolygonDistanceLoss
from .poly_validator import PolygonValidator
from .trainer import build_optimizer


class PolygonTrainer:
    """Train a polygon model on the merged polygon / polygon+distance datasets."""

    def __init__(
        self,
        model,
        poly_train=None,
        dist_train=None,
        val_data=None,
        val_has_distance=False,
        epochs=100,
        poly_batch=8,
        dist_batch=8,
        imgsz=640,
        device="cpu",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        optimizer="auto",
        workers=4,
        hyp=None,
        close_mosaic=10,
        project="runs/polygon",
        save=True,
        plots=True,
        tensorboard=False,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lr0 = lr0
        self.lrf = lrf
        self.warmup_epochs = warmup_epochs
        self.close_mosaic = close_mosaic
        self.save = save
        self.plots = plots
        self.project = Path(project)

        self.train_loader, self.train_concat = build_merged_dataloader(
            poly_train, dist_train, imgsz=imgsz, poly_batch=poly_batch, dist_batch=dist_batch,
            augment=True, workers=workers, hyp=hyp, num_angles=model.num_angles, angle_step=model.angle_step,
        )
        self.val_loader = (
            build_poly_dataloader(val_data, has_distance=val_has_distance, imgsz=imgsz, batch=max(poly_batch, dist_batch),
                                  augment=False, workers=workers, num_angles=model.num_angles, angle_step=model.angle_step)
            if val_data else None
        )

        self.optimizer = build_optimizer(self.model, optimizer, lr=lr0, momentum=momentum, decay=weight_decay)
        self.criterion = v8PolygonDistanceLoss(self.model)
        self.nb = len(self.train_loader)
        self.lf = lambda x: (1 - x / self.epochs) * (1.0 - lrf) + lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.warmup_iters = max(round(self.warmup_epochs * self.nb), 100)

        # Logging: results.csv + optional TensorBoard + end-of-run results.png.
        self.csv_path = self.project / "results.csv"
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.project.mkdir(parents=True, exist_ok=True)
                self.tb = SummaryWriter(str(self.project))
                LOGGER.info(f"TensorBoard: logging to {self.project}")
            except ImportError:
                LOGGER.info("tensorboard not installed; skipping TensorBoard logging")

    def _init_plots(self):
        """Fresh results.csv and a class-distribution histogram (labels.png)."""
        if self.csv_path.exists():
            self.csv_path.unlink()
        if not self.plots:
            return
        self.project.mkdir(parents=True, exist_ok=True)
        from ..utils.plotting import plot_label_histogram

        all_cls = []
        for ds in getattr(self.train_concat, "datasets_list", []):
            for i in range(len(ds)):
                _, cls, _ = ds.parse_label_file(i)
                all_cls.extend(cls.tolist())
        plot_label_histogram(all_cls, self.model.names, str(self.project / "labels.png"))

    def _log_epoch(self, epoch, mloss, lr, metrics):
        row = {
            "epoch": epoch + 1,
            "train/box": float(mloss[0]), "train/cls": float(mloss[1]), "train/dfl": float(mloss[2]),
            "train/poly": float(mloss[3]), "train/dist": float(mloss[4]), "lr": float(lr),
            "metrics/precision": float(metrics.get("precision", 0.0)),
            "metrics/recall": float(metrics.get("recall", 0.0)),
            "metrics/f1": float(metrics.get("f1", 0.0)),
        }
        self.project.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a") as f:
            if write_header:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(f"{v:g}" for v in row.values()) + "\n")
        if self.tb is not None:
            for k, v in row.items():
                if k != "epoch":
                    self.tb.add_scalar(k, v, epoch + 1)

    def _warmup(self, ni, epoch):
        if ni > self.warmup_iters:
            return
        xi = [0, self.warmup_iters]
        for j, pg in enumerate(self.optimizer.param_groups):
            warmup_bias_lr = 0.1 if j == 0 else 0.0
            pg["lr"] = np.interp(ni, xi, [warmup_bias_lr, self.lr0 * self.lf(epoch)])
            if "momentum" in pg:
                pg["momentum"] = np.interp(ni, xi, [0.8, 0.937])

    def _draw_train_batch(self, batch, path):
        """Save a 3x3 grid of a polygon training batch (bbox + star polygon) per epoch."""
        import cv2

        from ..utils.plotting import draw_poly_star, image_grid

        n = min(9, batch["img"].shape[0])
        tiles = []
        for k in range(n):
            arr = (batch["img"][k].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
            idx = batch["batch_idx"] == k
            tiles.append(draw_poly_star(np.ascontiguousarray(arr), batch["cls"][idx].numpy().reshape(-1),
                                        batch["bboxes"][idx].numpy(), batch["poly"][idx].numpy(),
                                        self.model.num_angles, self.model.names))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image_grid(tiles, cols=3))

    def train(self):
        LOGGER.info(f"Starting polygon training for {self.epochs} epochs on {self.device}...")
        self._init_plots()
        for epoch in range(self.epochs):
            self.model.train()
            if self.close_mosaic and epoch == self.epochs - self.close_mosaic:
                LOGGER.info(f"Closing mosaic augmentation for the last {self.close_mosaic} epochs")
                for ds in getattr(self.train_concat, "datasets_list", []):
                    ds.close_mosaic()
            mloss = torch.zeros(5, device=self.device)
            for i, batch in enumerate(self.train_loader):
                if i == 0 and self.plots:  # per-epoch train batch visualization
                    try:
                        self._draw_train_batch(batch, self.project / "train_batches" / f"train_epoch{epoch + 1:03d}.jpg")
                    except Exception as e:
                        LOGGER.info(f"train batch plot skipped: {e}")
                self._warmup(i + self.nb * epoch, epoch)
                imgs = batch["img"].to(self.device, non_blocking=True)
                loss, items = self.criterion(self.model(imgs), batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                mloss = (mloss * i + items) / (i + 1)
            self.scheduler.step()
            LOGGER.info(
                f"Epoch {epoch + 1}/{self.epochs}  box={mloss[0]:.3f} cls={mloss[1]:.3f} dfl={mloss[2]:.3f} "
                f"poly={mloss[3]:.4f} dist={mloss[4]:.4f}  lr={self.optimizer.param_groups[0]['lr']:.5f}"
            )
            metrics = {}
            if self.val_loader is not None:
                metrics = self.validate(epoch)
            self._log_epoch(epoch, mloss, self.optimizer.param_groups[0]["lr"], metrics)

        if self.plots and self.csv_path.exists():
            from ..utils.plotting import plot_results

            plot_results(str(self.csv_path))
            LOGGER.info(f"Saved training plots to {self.project}")
        if self.tb is not None:
            self.tb.close()
        if self.save:
            self._save()
        return self.model

    def validate(self, epoch=None):
        prefix = None
        if self.plots and epoch is not None:
            prefix = str(self.project / "val_batches" / f"val_epoch{epoch + 1:03d}")
        metrics = PolygonValidator(
            self.model, self.val_loader, device=self.device, names=self.model.names,
            num_angles=self.model.num_angles, plot_samples=self.plots and epoch is not None, sample_prefix=prefix,
        )()
        self.model.train()
        return metrics

    def _save(self):
        self.project.mkdir(parents=True, exist_ok=True)
        path = self.project / "polygon_last.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "nc": self.model.nc,
                "names": self.model.names,
                "num_angles": self.model.num_angles,
                "angle_step": self.model.angle_step,
            },
            path,
        )
        LOGGER.info(f"Saved checkpoint to {path}")
        return path
