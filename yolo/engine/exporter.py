"""Model export and quantization for the detection model.

Supported formats (no ``ultralytics`` dependency):
  * ``torchscript``  - traced ``*.torchscript`` (works everywhere torch runs);
  * ``onnx``         - ``*.onnx`` via ``torch.onnx.export`` (optional dynamic axes);
  * ``fp16`` flag    - half-precision export for the above;
  * ``int8``         - dynamic INT8 quantization (CPU) saved as a TorchScript file.

Before export the model is fused (Conv+BN) and switched to inference mode with the
detection head's ``export`` flag set, so the exported graph emits a single decoded
output tensor ``(B, 4 + nc, num_anchors)``.
"""

from copy import deepcopy
from pathlib import Path

import torch

from ..nn.modules import Detect
from ..utils import LOGGER


def _prepare(model, fuse=True):
    """Return a deep-copied, fused, eval-mode model with the Detect head in export mode."""
    m = deepcopy(model).eval()
    if fuse and hasattr(m, "fuse") and not getattr(m, "is_fused", False):
        m.fuse()
    for module in m.modules():
        if isinstance(module, Detect):
            module.export = True
    return m


class Exporter:
    """Export / quantize a detection model to deployable formats."""

    def __init__(self, model, imgsz=640, device="cpu", half=False):
        self.model = model
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        self.device = device
        self.half = half

    def _example(self, model):
        im = torch.zeros(1, 3, *self.imgsz, device=self.device)
        if self.half:
            model.half()
            im = im.half()
        return model.to(self.device), im

    def export_torchscript(self, file=None):
        file = str(file or "model.torchscript")
        model, im = self._example(_prepare(self.model))
        # check_trace=False: the head caches anchors for the fixed export size, so the
        # tracer's re-run sanity check (which expects identical constants) is not applicable.
        with torch.no_grad():
            model(im)  # warm up the cached anchors at the export resolution
            ts = torch.jit.trace(model, im, strict=False, check_trace=False)
        ts.save(file)
        LOGGER.info(f"TorchScript export: {file}")
        return file

    def export_onnx(self, file=None, opset=12, dynamic=False, simplify=True):
        file = str(file or "model.onnx")
        model, im = self._example(_prepare(self.model))
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "anchors"}}
        export_kwargs = dict(
            opset_version=opset, input_names=["images"], output_names=["output"],
            dynamic_axes=dynamic_axes, do_constant_folding=True,
        )
        with torch.no_grad():
            try:  # torch>=2.x defaults to the dynamo exporter (needs onnxscript); fall back to legacy
                torch.onnx.export(model, im, file, dynamo=False, **export_kwargs)
            except TypeError:
                torch.onnx.export(model, im, file, **export_kwargs)
        try:  # optional validation / simplification
            import onnx

            onnx_model = onnx.load(file)
            onnx.checker.check_model(onnx_model)
            if simplify:
                try:
                    import onnxslim

                    onnx_model = onnxslim.slim(onnx_model)
                    onnx.save(onnx_model, file)
                except ImportError:
                    pass
        except ImportError:
            LOGGER.info("onnx not installed; skipped model validation")
        LOGGER.info(f"ONNX export: {file} (opset {opset}, dynamic={dynamic})")
        return file

    def export_int8_onnx(self, file=None, opset=12):
        """Dynamic INT8 quantization via ONNX Runtime (quantizes Conv/MatMul to int8)."""
        file = str(file or "model_int8.onnx")
        fp32 = self.export_onnx(Path(file).with_suffix(".fp32.onnx"), opset=opset, simplify=False)
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as e:
            raise RuntimeError("INT8 export needs onnxruntime: pip install onnxruntime") from e
        quantize_dynamic(fp32, file, weight_type=QuantType.QInt8)
        Path(fp32).unlink(missing_ok=True)
        LOGGER.info(f"INT8 (dynamic, ONNX Runtime) export: {file}")
        return file

    def export(self, format="torchscript", file=None, **kwargs):
        """Dispatch to the requested export format."""
        fmt = format.lower()
        stem = Path(file).stem if file else "model"
        if fmt == "torchscript":
            return self.export_torchscript(file or f"{stem}.torchscript")
        if fmt == "onnx":
            return self.export_onnx(file or f"{stem}.onnx", **kwargs)
        if fmt == "int8":
            return self.export_int8_onnx(file or f"{stem}_int8.onnx", **kwargs)
        raise ValueError(f"Unsupported export format: {format!r} (use torchscript / onnx / int8)")
