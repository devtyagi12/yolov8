"""Load official Ultralytics ``yolov8*.pt`` checkpoints without importing ``ultralytics``.

An official checkpoint is a pickled ``dict`` whose ``"model"`` entry is a live
``ultralytics.nn.tasks.DetectionModel`` instance. Unpickling it normally requires
the ``ultralytics`` package to reconstruct those classes. We instead drive
``torch.load`` with a custom unpickler that substitutes a lightweight *stub* for
every ``ultralytics.*`` class. The stub simply absorbs the pickled ``__dict__``
(which still contains the real ``torch`` parameters/buffers), letting us walk the
module tree and emit a plain ``state_dict`` with keys identical to this package's
``DetectionModel`` (``model.0.conv.weight`` ...).
"""

import pickle

import torch

from . import LOGGER


class _Stub:
    """Placeholder standing in for any ``ultralytics.*`` (or otherwise missing) class."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
            self.__dict__.update(state[0])
        else:
            self.__dict__["_state"] = state

    # Some objects are reconstructed via __reduce__ with positional args; tolerate calls.
    def __call__(self, *args, **kwargs):
        return self


def _make_stub(name):
    return type(name, (_Stub,), {})


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that returns a stub for unknown / ultralytics classes."""

    def find_class(self, module, name):
        if module.startswith("ultralytics") or module.startswith("__main__"):
            return _make_stub(name)
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _make_stub(name)


class _SafePickleModule:
    """Drop-in ``pickle_module`` for ``torch.load``."""

    Unpickler = _SafeUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _SafeUnpickler(file).load()


def _flatten_module(obj, prefix, out):
    """Recursively collect parameters / buffers from a (stubbed) nn.Module tree."""
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return
    for name, p in (d.get("_parameters") or {}).items():
        if p is not None:
            out[prefix + name] = p.detach() if torch.is_tensor(p) else p
    for name, b in (d.get("_buffers") or {}).items():
        if b is not None:
            out[prefix + name] = b.detach() if torch.is_tensor(b) else b
    for name, sub in (d.get("_modules") or {}).items():
        if sub is not None:
            _flatten_module(sub, prefix + name + ".", out)


def _extract_state_dict(model_obj):
    """Build a flat state dict from a stubbed model object."""
    out = {}
    _flatten_module(model_obj, "", out)
    return out


def load_checkpoint(path, map_location="cpu"):
    """Load any YOLOv8 checkpoint and return ``(state_dict, meta)``.

    Handles three input formats transparently:
      * an official Ultralytics ``.pt`` (pickled DetectionModel),
      * a plain ``state_dict`` produced by this package, and
      * a ``dict`` with a ``"model"`` state_dict / ``"state_dict"`` entry.

    Returns:
        (dict, dict): the tensor state dict and a metadata dict that may contain
        ``names`` (class id -> name) and ``nc`` when available.
    """
    try:
        ckpt = torch.load(
            path, map_location=map_location, pickle_module=_SafePickleModule, weights_only=False
        )
    except TypeError:  # very old torch without pickle_module kwarg
        ckpt = torch.load(path, map_location=map_location)

    meta = {}

    # Case 1: plain state dict of tensors.
    if isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, meta

    # Case 2: training checkpoint dict.
    if isinstance(ckpt, dict):
        # Metadata that may sit alongside the weights.
        if isinstance(ckpt.get("names"), (dict, list, tuple)):
            _collect_meta(_DictAttr(ckpt), meta)

        # 2a) a plain state dict stored under a known key.
        for key in ("model_state_dict", "state_dict"):
            v = ckpt.get(key)
            if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
                return v, meta

        # 2b) a live (stubbed) model object under "model"/"ema".
        model_obj = ckpt.get("ema") or ckpt.get("model")
        if model_obj is not None and not torch.is_tensor(model_obj):
            if isinstance(model_obj, dict) and model_obj and all(torch.is_tensor(t) for t in model_obj.values()):
                return model_obj, meta
            sd = _extract_state_dict(model_obj)
            _collect_meta(model_obj, meta)
            if sd:
                return sd, meta

    # Case 3: a bare (stubbed) model object.
    if not isinstance(ckpt, dict):
        sd = _extract_state_dict(ckpt)
        _collect_meta(ckpt, meta)
        if sd:
            return sd, meta

    raise ValueError(f"Unrecognized checkpoint format at {path!r}")


class _DictAttr:
    """Expose a plain dict's items through ``__dict__`` for ``_collect_meta``."""

    def __init__(self, d):
        self.__dict__ = d


def _collect_meta(model_obj, meta):
    """Pull class names / count out of a stubbed model object if present."""
    d = getattr(model_obj, "__dict__", {})
    names = d.get("names")
    if isinstance(names, dict):
        meta["names"] = names
        meta["nc"] = len(names)
    elif isinstance(names, (list, tuple)):
        meta["names"] = {i: n for i, n in enumerate(names)}
        meta["nc"] = len(names)
    yaml = d.get("yaml")
    if isinstance(yaml, dict) and "nc" in yaml:
        meta.setdefault("nc", yaml["nc"])


def remap_state_dict(state_dict):
    """Normalise key prefixes so they line up with this package's ``DetectionModel``.

    Official keys already look like ``model.0.conv.weight``; some exports prefix
    them with ``model.model.`` (when the whole detector is nested). Strip a leading
    ``model.`` only when doing so still yields ``model.<int>.`` keys.
    """
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("model.model.") for k in keys):
        return {k[len("model.") :]: v for k, v in state_dict.items()}
    return state_dict
