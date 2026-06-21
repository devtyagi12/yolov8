"""Detection trainer with AMP, EMA, multi-GPU, cosine LR, resume and image caching."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.build import build_dataloader
from ..utils import LOGGER
from ..utils.loss import v8DetectionLoss
from ..utils.torch_utils import Autocast, ModelEMA, cosine_lr, de_parallel, linear_lr, select_device
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
        amp=False,
        ema=True,
        cos_lr=False,
        cache=False,
        resume=None,
        tensorboard=False,
        plots=True,
    ):
        hyp = dict(hyp or {})
        if mosaic is not None:
            hyp["mosaic"] = mosaic

        self.device, self.device_ids = select_device(device)
        self.model = model.to(self.device)
        self.epochs = epochs
        self.imgsz = imgsz
        self.lr0 = lr0
        self.lrf = lrf
        self.warmup_epochs = warmup_epochs
        self.save = save
        self.project = Path(project)
        self.close_mosaic = close_mosaic
        self.best_fitness = 0.0
        self.start_epoch = 0

        # Multi-GPU (DataParallel) wrapper used only for the forward pass.
        self.parallel_model = self.model
        if len(self.device_ids) > 1:
            self.parallel_model = nn.DataParallel(self.model, device_ids=self.device_ids)
            LOGGER.info(f"Using DataParallel across GPUs {self.device_ids}")

        self.train_loader = build_dataloader(
            data_train, imgsz=imgsz, batch=batch, augment=True, workers=workers, hyp=hyp, cache=cache
        )
        self.val_loader = (
            build_dataloader(data_val, imgsz=imgsz, batch=batch, augment=False, workers=workers, shuffle=False, cache=cache)
            if data_val else None
        )

        self.optimizer = build_optimizer(self.model, optimizer, lr=lr0, momentum=momentum, decay=weight_decay)
        self.criterion = v8DetectionLoss(self.model)
        self.nb = len(self.train_loader)
        self.lf = (cosine_lr if cos_lr else linear_lr)(lrf, self.epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.warmup_iters = max(round(self.warmup_epochs * self.nb), 100)

        # AMP (CUDA only) and EMA.
        self.amp = bool(amp) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp) if self.device.type == "cuda" else None
        self.ema = ModelEMA(self.model) if ema else None

        # Logging: results.csv, optional TensorBoard, and end-of-run plots.
        self.plots = plots
        self.csv_path = self.project / "results.csv"
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.project.mkdir(parents=True, exist_ok=True)
                self.tb = SummaryWriter(str(self.project))
                LOGGER.info(f"TensorBoard: logging to {self.project} (run: tensorboard --logdir {self.project})")
            except ImportError:
                LOGGER.info("tensorboard not installed; skipping TensorBoard logging")

        if resume:
            self._resume(resume)

    # ------------------------------------------------------------------ logging
    def _draw_train_batch(self, batch, path):
        """Save a 3x3 grid of a training batch with its (augmented) labels drawn."""
        import cv2

        from ..utils.plotting import draw_normalized, image_grid

        tiles = []
        for k in range(min(9, batch["img"].shape[0])):
            arr = (batch["img"][k].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
            idx = batch["batch_idx"] == k
            tiles.append(draw_normalized(np.ascontiguousarray(arr), batch["cls"][idx].numpy().reshape(-1),
                                         batch["bboxes"][idx].numpy(), self.model.names))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image_grid(tiles, cols=3))

    def _init_plots(self):
        # Start a fresh results.csv for a new run (kept on resume).
        if self.csv_path.exists():
            self.csv_path.unlink()
        if not self.plots:
            return
        self.project.mkdir(parents=True, exist_ok=True)
        from ..utils.plotting import plot_label_histogram

        ds = self.train_loader.dataset
        # Class-distribution histogram (labels.png).
        all_cls = []
        for i in range(len(ds)):
            lb = ds.load_labels(i)
            if lb.shape[0]:
                all_cls.extend(lb[:, 0].tolist())
        plot_label_histogram(all_cls, self.model.names, str(self.project / "labels.png"))

    def _log_epoch(self, epoch, mloss, lr, metrics):
        row = {
            "epoch": epoch + 1,
            "train/box": float(mloss[0]), "train/cls": float(mloss[1]), "train/dfl": float(mloss[2]),
            "lr": float(lr),
            "metrics/precision": float(metrics.get("mp", 0.0)), "metrics/recall": float(metrics.get("mr", 0.0)),
            "metrics/mAP50": float(metrics.get("map50", 0.0)), "metrics/mAP50-95": float(metrics.get("map", 0.0)),
        }
        # results.csv (header written once, when the file does not yet exist)
        self.project.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a") as f:
            if write_header:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(f"{v:g}" for v in row.values()) + "\n")
        # TensorBoard
        if self.tb is not None:
            for k, v in row.items():
                if k != "epoch":
                    self.tb.add_scalar(k, v, epoch + 1)

    # ------------------------------------------------------------------ schedule
    def _warmup(self, ni, epoch):
        if ni > self.warmup_iters:
            return
        xi = [0, self.warmup_iters]
        for j, pg in enumerate(self.optimizer.param_groups):
            warmup_bias_lr = 0.1 if j == 0 else 0.0
            pg["lr"] = np.interp(ni, xi, [warmup_bias_lr, self.lr0 * self.lf(epoch)])
            if "momentum" in pg:
                pg["momentum"] = np.interp(ni, xi, [0.8, 0.937])

    # ------------------------------------------------------------------ train loop
    def train(self):
        LOGGER.info(f"Starting training for {self.epochs} epochs on {self.device} "
                    f"(amp={self.amp}, ema={self.ema is not None})...")
        if self.start_epoch == 0:
            self._init_plots()
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            if self.close_mosaic and epoch == self.epochs - self.close_mosaic:
                LOGGER.info(f"Closing mosaic augmentation for the last {self.close_mosaic} epochs")
                self.train_loader.dataset.close_mosaic()
            mloss = torch.zeros(3, device=self.device)
            for i, batch in enumerate(self.train_loader):
                if i == 0 and self.plots:  # per-epoch train batch visualization
                    try:
                        self._draw_train_batch(batch, self.project / "train_batches" / f"train_epoch{epoch + 1:03d}.jpg")
                    except Exception as e:
                        LOGGER.info(f"train batch plot skipped: {e}")
                self._warmup(i + self.nb * epoch, epoch)
                imgs = batch["img"].to(self.device, non_blocking=True)

                with Autocast(self.amp, self.device.type):
                    preds = self.parallel_model(imgs)
                    loss, loss_items = self.criterion(preds, batch)

                self.optimizer.zero_grad()
                if self.scaler is not None and self.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)

                mloss = (mloss * i + loss_items) / (i + 1)
            self.scheduler.step()

            LOGGER.info(
                f"Epoch {epoch + 1}/{self.epochs}  box={mloss[0]:.4f} cls={mloss[1]:.4f} dfl={mloss[2]:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.5f}"
            )

            fitness = 0.0
            metrics = {}
            if self.val_loader is not None:
                metrics = self.validate(epoch)
                fitness = 0.1 * metrics.get("map50", 0.0) + 0.9 * metrics.get("map", 0.0)
            self._log_epoch(epoch, mloss, self.optimizer.param_groups[0]["lr"], metrics)
            if self.save:
                self._save_checkpoint(epoch, fitness)

        # End-of-training results curves (results.png).
        if self.plots and self.csv_path.exists():
            from ..utils.plotting import plot_results

            plot_results(str(self.csv_path))
            LOGGER.info(f"Saved training plots to {self.project}")
        if self.tb is not None:
            self.tb.close()
        return self.eval_model()

    def eval_model(self):
        """Return the EMA model if available, else the trained model."""
        return self.ema.ema if self.ema is not None else self.model

    def validate(self, epoch=None):
        model = self.eval_model()
        prefix = None
        if self.plots and epoch is not None:
            prefix = str(self.project / "val_batches" / f"val_epoch{epoch + 1:03d}")
        metrics = DetectionValidator(
            model, self.val_loader, device=self.device, names=self.model.names,
            plot_samples=self.plots and epoch is not None, sample_prefix=prefix,
        )()
        self.model.train()
        return metrics

    # ------------------------------------------------------------------ checkpoints
    def _ckpt(self, epoch, fitness):
        return {
            "epoch": epoch,
            "best_fitness": max(self.best_fitness, fitness),
            "model_state_dict": de_parallel(self.model).state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema else None,
            "ema_updates": self.ema.updates if self.ema else 0,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "nc": self.model.nc,
            "names": self.model.names,
        }

    def _save_checkpoint(self, epoch, fitness):
        self.project.mkdir(parents=True, exist_ok=True)
        ckpt = self._ckpt(epoch, fitness)
        torch.save(ckpt, self.project / "last.pt")
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            torch.save(ckpt, self.project / "best.pt")
        return self.project / "last.pt"

    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        de_parallel(self.model).load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.ema is not None and ckpt.get("ema_state_dict"):
            self.ema.ema.load_state_dict(ckpt["ema_state_dict"])
            self.ema.updates = ckpt.get("ema_updates", 0)
        self.best_fitness = ckpt.get("best_fitness", 0.0)
        self.start_epoch = ckpt.get("epoch", -1) + 1
        LOGGER.info(f"Resumed from {path} at epoch {self.start_epoch}")
