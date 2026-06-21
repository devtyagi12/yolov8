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
        poly_train,
        dist_train,
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
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lr0 = lr0
        self.lrf = lrf
        self.warmup_epochs = warmup_epochs
        self.close_mosaic = close_mosaic
        self.save = save
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

    def _warmup(self, ni, epoch):
        if ni > self.warmup_iters:
            return
        xi = [0, self.warmup_iters]
        for j, pg in enumerate(self.optimizer.param_groups):
            warmup_bias_lr = 0.1 if j == 0 else 0.0
            pg["lr"] = np.interp(ni, xi, [warmup_bias_lr, self.lr0 * self.lf(epoch)])
            if "momentum" in pg:
                pg["momentum"] = np.interp(ni, xi, [0.8, 0.937])

    def train(self):
        LOGGER.info(f"Starting polygon training for {self.epochs} epochs on {self.device}...")
        for epoch in range(self.epochs):
            self.model.train()
            if self.close_mosaic and epoch == self.epochs - self.close_mosaic:
                LOGGER.info(f"Closing mosaic augmentation for the last {self.close_mosaic} epochs")
                for ds in getattr(self.train_concat, "datasets_list", []):
                    ds.close_mosaic()
            mloss = torch.zeros(5, device=self.device)
            for i, batch in enumerate(self.train_loader):
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
            if self.val_loader is not None:
                self.validate()

        if self.save:
            self._save()
        return self.model

    def validate(self):
        metrics = PolygonValidator(self.model, self.val_loader, device=self.device)()
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
