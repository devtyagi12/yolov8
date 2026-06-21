"""Tests for Conv+BN fusion and model export / quantization.

Run:  python tests/test_export.py   (or)   python -m pytest tests/test_export.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo import YOLO


def test_fuse_is_output_equivalent():
    m = YOLO("yolov8n", verbose=False)
    m.model.eval()
    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        before = m.model(x)[0]
    m.fuse()
    assert getattr(m.model, "is_fused", False)
    # no BatchNorm layers should remain
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.model.modules())
    with torch.no_grad():
        after = m.model(x)[0]
    assert torch.allclose(before, after, atol=1e-5)


def test_torchscript_export_runs(tmp_path="/tmp"):
    m = YOLO("yolov8n", verbose=False)
    f = m.export("torchscript", file=os.path.join(tmp_path, "t.torchscript"), imgsz=320)
    assert os.path.exists(f)
    x = torch.zeros(1, 3, 320, 320)
    out = torch.jit.load(f)(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert out.shape == (1, 84, 8400 // 4)  # 320px -> 2100 anchors


def test_onnx_export_matches_torch(tmp_path="/tmp"):
    import numpy as np
    import onnxruntime as ort

    m = YOLO("yolov8n", verbose=False)
    f = m.export("onnx", file=os.path.join(tmp_path, "t.onnx"), imgsz=320)
    x = torch.zeros(1, 3, 320, 320)
    onnx_out = ort.InferenceSession(f).run(None, {"images": x.numpy()})[0]
    m.model.eval()
    with torch.no_grad():
        torch_out = m.model(x)[0].numpy()
    assert np.abs(onnx_out - torch_out).max() < 1e-4


def test_int8_onnx_is_smaller(tmp_path="/tmp"):
    m = YOLO("yolov8n", verbose=False)
    fp32 = m.export("onnx", file=os.path.join(tmp_path, "q.onnx"), imgsz=320)
    int8 = m.export("int8", file=os.path.join(tmp_path, "q_int8.onnx"), imgsz=320)
    assert os.path.getsize(int8) < 0.5 * os.path.getsize(fp32)  # at least 2x smaller


if __name__ == "__main__":
    test_fuse_is_output_equivalent()
    test_torchscript_export_runs()
    test_onnx_export_matches_torch()
    test_int8_onnx_is_smaller()
    print("All export tests passed.")
