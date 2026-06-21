"""Torch helpers: EMA, parallel unwrapping, LR schedules, device selection, AMP."""

import math
from copy import deepcopy

import torch
import torch.nn as nn

from . import LOGGER


def de_parallel(model):
    """Return the underlying model from a DataParallel / DistributedDataParallel wrapper."""
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def select_device(device=""):
    """Resolve a device spec ('cpu', '0', '0,1', 'cuda') into (torch.device, list[int])."""
    if device in ("cpu", None, ""):
        if torch.cuda.is_available() and device != "cpu":
            ids = list(range(torch.cuda.device_count()))
            return torch.device("cuda:0"), ids
        return torch.device("cpu"), []
    device = str(device).replace("cuda:", "").replace("cuda", "").replace(" ", "")
    if device == "cpu":
        return torch.device("cpu"), []
    ids = [int(x) for x in device.split(",") if x != ""]
    if torch.cuda.is_available() and ids:
        return torch.device(f"cuda:{ids[0]}"), ids
    LOGGER.info("CUDA not available; falling back to CPU")
    return torch.device("cpu"), []


def cosine_lr(lrf, epochs):
    """Cosine LR factor schedule from 1.0 down to ``lrf`` over ``epochs``."""
    return lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1


def linear_lr(lrf, epochs):
    """Linear LR factor schedule from 1.0 down to ``lrf`` over ``epochs``."""
    return lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf


class ModelEMA:
    """Exponential Moving Average of model weights (updated every optimizer step).

    Keeps a shadow copy whose parameters track the model with a ramped decay, which
    usually validates / exports better than the raw weights.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # ramp up early in training
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        msd = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1.0 - d) * msd[k].detach().to(v.dtype)

    def state_dict(self):
        return self.ema.state_dict()


class Autocast:
    """Context manager wrapping torch autocast; a no-op when AMP is disabled / on CPU."""

    def __init__(self, enabled, device_type="cuda"):
        self.enabled = enabled and device_type == "cuda"
        self.device_type = device_type

    def __enter__(self):
        if self.enabled:
            self._cm = torch.autocast(device_type=self.device_type, dtype=torch.float16)
            self._cm.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self._cm.__exit__(*args)
