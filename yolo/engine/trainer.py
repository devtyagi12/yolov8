"""Detection trainer: a compact but complete YOLOv8 training loop."""

from pathlib import Path

import numpy as np
import torch

from ..data.build import build_dataloader
from ..utils import LOGGER
from ..utils.loss import v8DetectionLoss
from .validator import DetectionValidator


def build_optimizer(model, name="auto", lr=0.01, momentum=0.937, decay=5e-4):
    """Create an optimizer with weight-decay applied only to conv/linear weights."""
    g_bn, g_w, g_bias = [], [], []
    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pname == "bias":
                g_bias.append(p)
            elif isinstance(module, torch.nn.BatchNorm2d):
                g_bn.append(p)
            else:
                g_w.append(p)

    if name == "auto":
        name = "SGD"
    if name == "Adam":
        opt = torch.optim.Adam(g_bias, lr=lr, betas=(momentum, 0.999))
    elif name == "AdamW":
        opt = torch.optim.AdamW(g_bias, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    else:
        opt = torch.optim.SGD(g_bias, lr=lr, momentum=momentum, nesterov=True)
    opt.add_param_group({"params": g_w, "weight_decay": decay})  # weights with decay
    opt.add_param_group({"params": g_bn, "weight_decay": 0.0})  # BN without decay
    LOGGER.info(f"optimizer: {name}(lr={lr}) groups: {len(g_w)} weight(decay={decay}), {len(g_bias)} bias, {len(g_bn)} bn")
    return opt


class DetectionTrainer:
    """Train a :class:`DetectionModel` on a YOLO-format dataset."""

    def __init__(
        self,
        model,
        data_train,
        data_val=None,
        epochs=100,
        batch=16,
        imgsz=640,
        device="cpu",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        optimizer="auto",
        workers=4,
        project="runs/train",
        save=True,
        hyp=None,
        mosaic=None,
        close_mosaic=10,
    ):
        hyp = dict(hyp or {})
        if mosaic is not None:  # convenience override
            hyp["mosaic"] = mosaic
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.imgsz = imgsz
        self.lr0 = lr0
        self.lrf = lrf
        self.warmup_epochs = warmup_epochs
        self.save = save
        self.project = Path(project)
        self.close_mosaic = close_mosaic  # disable mosaic/mixup/copy-paste for the final N epochs

        self.train_loader = build_dataloader(
            data_train, imgsz=imgsz, batch=batch, augment=True, workers=workers, hyp=hyp
        )
        self.val_loader = (
            build_dataloader(data_val, imgsz=imgsz, batch=batch, augment=False, workers=workers, shuffle=False)
            if data_val
            else None
        )
        self.optimizer = build_optimizer(self.model, optimizer, lr=lr0, momentum=momentum, decay=weight_decay)
        self.criterion = v8DetectionLoss(self.model)
        self.nb = len(self.train_loader)
        self.lf = lambda x: (1 - x / self.epochs) * (1.0 - lrf) + lrf  # linear LR schedule
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
        LOGGER.info(f"Starting training for {self.epochs} epochs on {self.device}...")
        self.model.train()
        for epoch in range(self.epochs):
            self.model.train()
            # Turn off mosaic for the final epochs so the model finishes on clean images.
            if self.close_mosaic and epoch == self.epochs - self.close_mosaic:
                LOGGER.info(f"Closing mosaic augmentation for the last {self.close_mosaic} epochs")
                self.train_loader.dataset.close_mosaic()
            mloss = torch.zeros(3, device=self.device)
            for i, batch in enumerate(self.train_loader):
                ni = i + self.nb * epoch
                self._warmup(ni, epoch)

                imgs = batch["img"].to(self.device, non_blocking=True)
                preds = self.model(imgs)
                loss, loss_items = self.criterion(preds, batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                mloss = (mloss * i + loss_items) / (i + 1)
            self.scheduler.step()

            LOGGER.info(
                f"Epoch {epoch + 1}/{self.epochs}  "
                f"box={mloss[0]:.4f} cls={mloss[1]:.4f} dfl={mloss[2]:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.5f}"
            )

            if self.val_loader is not None and (epoch + 1) % 1 == 0:
                self.validate()

        if self.save:
            self._save_checkpoint()
        return self.model

    def validate(self):
        validator = DetectionValidator(self.model, self.val_loader, device=self.device)
        metrics = validator()
        self.model.train()
        return metrics

    def _save_checkpoint(self):
        self.project.mkdir(parents=True, exist_ok=True)
        path = self.project / "weights_last.pt"
        torch.save({"model_state_dict": self.model.state_dict(), "nc": self.model.nc, "names": self.model.names}, path)
        LOGGER.info(f"Saved checkpoint to {path}")
        return path
