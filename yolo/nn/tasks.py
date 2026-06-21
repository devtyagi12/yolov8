"""Detection model assembly: parse a YOLOv8 config into an ``nn.Module``.

``parse_model`` reproduces the channel / depth scaling logic of the official
Ultralytics parser so that the resulting module tree is byte-for-byte compatible
with official checkpoints.
"""

import contextlib
from copy import deepcopy

import torch
import torch.nn as nn

from ..cfg import get_cfg
from ..utils import LOGGER
from ..utils.ops import make_divisible
from .modules import C2f, Concat, Conv, Detect, SPPF
from .poly_modules import DetectPolygon

# Registry mapping config strings to module classes.
_MODULES = {
    "Conv": Conv,
    "C2f": C2f,
    "SPPF": SPPF,
    "Concat": Concat,
    "Detect": Detect,
    "DetectPolygon": DetectPolygon,
    "nn.Upsample": nn.Upsample,
}

# Heads that need the list of input channels appended to their args.
_HEADS = (Detect, DetectPolygon)


def parse_model(d, ch, verbose=True):
    """Build a YOLOv8 module list from a config dict ``d`` given input channels ``ch``.

    Returns:
        (nn.Sequential, list[int]): the model layers and the sorted 'savelist' of
        layer indices whose outputs must be cached for later ``from`` references.
    """
    nc = d["nc"]
    scales = d.get("scales")
    scale = d.get("scale", "n")
    depth, width, max_channels = 1.0, 1.0, float("inf")
    if scales:
        depth, width, max_channels = scales[scale]

    if verbose:
        LOGGER.info(f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

    ch = [ch]  # input channels list, indexed by layer output
    layers, save = [], []
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m_cls = _MODULES[m] if isinstance(m, str) else m
        for j, a in enumerate(args):
            if isinstance(a, str):
                # eval literals like "None"; leave bare names (e.g. "nearest", "nc") as strings
                with contextlib.suppress(NameError, SyntaxError):
                    args[j] = eval(a)
        n_ = n = max(round(n * depth), 1) if n > 1 else n  # depth gain

        if m_cls in (Conv, C2f, SPPF):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # not the final output layer
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m_cls is C2f:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m_cls is nn.Upsample:
            c2 = ch[f]
        elif m_cls is Concat:
            c2 = sum(ch[x] for x in f)
        elif m_cls in _HEADS:
            # Detect(nc, ch[, ...]) / DetectPolygon(nc, ch, num_angles, angle_step, num_dist_blocks)
            args[0] = nc
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m_cls(*args) for _ in range(n))) if n > 1 else m_cls(*args)
        t = str(m_cls)[8:-2].replace("__main__.", "")  # module type string
        n_params = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, n_params
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{n_params:10.0f}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class DetectionModel(nn.Module):
    """YOLOv8 detection model."""

    def __init__(self, cfg="n", ch=3, nc=None, verbose=True):
        super().__init__()
        if isinstance(cfg, str) and cfg in ("n", "s", "m", "l", "x"):
            cfg = get_cfg(scale=cfg, nc=nc or 80)
        elif isinstance(cfg, dict):
            cfg = deepcopy(cfg)
            if nc is not None:
                cfg["nc"] = nc
        else:
            raise ValueError(f"Unsupported cfg: {cfg!r}. Use one of n/s/m/l/x or a config dict.")
        self.yaml = cfg
        self.nc = cfg["nc"]

        self.model, self.save = parse_model(deepcopy(cfg), ch=ch, verbose=verbose)
        self.names = {i: f"class{i}" for i in range(self.nc)}
        self.inplace = True

        # Build strides from a single forward pass.
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            was_training = self.training
            self.train()  # ensure the head returns per-level feature maps
            out = self.forward(torch.zeros(1, ch, s, s))
            feats = out[0] if isinstance(out, tuple) else out  # DetectPolygon returns (feats, *extras)
            m.stride = torch.tensor([s / x.shape[-2] for x in feats])
            self.train(was_training)
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.tensor([32.0])

        initialize_weights(self)
        if verbose:
            LOGGER.info(f"DetectionModel summary: {self.nc} classes, {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x):
        """Run a forward pass, routing cached outputs to layers that need them."""
        y = []  # cached outputs
        for m in self.model:
            if m.f != -1:  # not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def load_state_dict_compat(self, state_dict, strict=False):
        """Load a (possibly remapped) state dict, reporting mismatches."""
        model_sd = self.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing = [k for k in model_sd if k not in matched]
        unexpected = [k for k in state_dict if k not in model_sd]
        self.load_state_dict(matched, strict=False)
        LOGGER.info(
            f"Transferred {len(matched)}/{len(model_sd)} items from checkpoint "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict load failed: {len(missing)} missing, {len(unexpected)} unexpected")
        return self


def initialize_weights(model):
    """Apply sensible default initialisations (eps / momentum for BN, inplace acts)."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # use default kaiming init
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
            m.inplace = True
