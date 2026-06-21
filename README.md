# YOLOv8 — standalone (ultralytics-free)

A self-contained re-implementation of **YOLOv8 object detection** in plain
PyTorch. It reproduces the model, training, validation and inference pipeline of
the official Ultralytics release **without importing the `ultralytics` library**,
while staying **weight-compatible** with the official `yolov8{n,s,m,l,x}.pt`
checkpoints.

> Scope: object **detection** (architecture + inference + training + validation).
> Segmentation / pose / classify / OBB are intentionally out of scope.

## Why

`ultralytics` is a large dependency with its own CLI, config system and licensing.
This package extracts just the detection model into a small, readable, dependency-light
codebase you can vendor, audit and modify — and it can still load the official
pretrained weights so you get real accuracy out of the box.

## Install

```bash
pip install torch torchvision numpy opencv-python
# then just use the `yolo` package in this repo (no pip install needed)
```

## Quick start

### Inference with official weights

The loader reads an official `yolov8n.pt` **without** `ultralytics` installed — a
custom unpickler substitutes a lightweight stub for every `ultralytics.*` class and
extracts the tensors into a state dict whose keys match this model exactly.

```python
from yolo import YOLO

model = YOLO("yolov8n.pt")            # official checkpoint, no ultralytics required
results = model.predict("bus.jpg", conf=0.25)
print(results[0].summary())           # list of {name, class, confidence, box}
results[0].save("out.jpg")            # annotated image
```

Or from the command line:

```bash
python examples/predict.py --weights yolov8n.pt --source bus.jpg --out out.jpg
```

### Training

Standard YOLO dataset layout (identical to Ultralytics):

```
datasets/mydata/
  images/train/*.jpg   labels/train/*.txt    # each line: "cls cx cy w h" (normalised)
  images/val/*.jpg     labels/val/*.txt
```

```python
from yolo import YOLO

model = YOLO("yolov8n", nc=3)          # fresh model, or "yolov8n.pt" to fine-tune
model.train(
    data_train="datasets/mydata/images/train",
    data_val="datasets/mydata/images/val",
    epochs=100, batch=16, imgsz=640,
)
metrics = model.val("datasets/mydata/images/val")   # {'map50':..., 'map':...}
```

Or:

```bash
python examples/train.py --model yolov8n --nc 3 \
    --train datasets/mydata/images/train --val datasets/mydata/images/val \
    --epochs 100 --batch 16
```

### Convert official weights to a clean checkpoint

Produce a pure `state_dict` file that loads with stock PyTorch (no `ultralytics`):

```bash
python tools/convert_weights.py yolov8n.pt yolov8n_clean.pt
```

## What's implemented

| Area | Details |
|------|---------|
| Architecture | `Conv`, `Bottleneck`, `C2f`, `SPPF`, `Concat`, `Detect`, `DFL`; n/s/m/l/x via depth/width scaling. Parameter counts match official (3.16M / 11.17M / 25.9M / 43.7M / 68.2M). |
| Weight loading | Loads official `.pt` with no `ultralytics` dependency via a safe unpickler + state-dict extractor. Verified bit-exact round trip. |
| Loss | `v8DetectionLoss` = CIoU box loss + Distribution Focal Loss + BCE classification, with the Task-Aligned Assigner. |
| Data | YOLO-format `YOLODataset` + the full `v8_transforms` chain (mosaic, copy-paste, random-perspective, mixup, albumentations, HSV, vertical/horizontal flip), collate function, `DataLoader` builder. |
| Engine | `DetectionPredictor` (preprocess → forward → NMS → `Results`), `DetectionTrainer` (warmup, LR schedule, param-group weight decay), `DetectionValidator` (COCO-style mAP@0.5 / mAP@0.5:0.95). |

## Augmentation & visualization

Training uses the full Ultralytics `v8_transforms` chain, ported to plain
NumPy/OpenCV as composable transform classes:

```
Mosaic -> CopyPaste -> RandomPerspective -> MixUp -> Albumentations
       -> RandomHSV -> RandomFlip(vertical) -> RandomFlip(horizontal) -> Format
```

| Transform | Hyper-parameter(s) | Default |
|-----------|--------------------|---------|
| `Mosaic` (4-image) | `mosaic` | 1.0 |
| `CopyPaste` (box-level) | `copy_paste` | 0.0 |
| `RandomPerspective` (rotate/scale/shear/translate/perspective) | `degrees`, `scale`, `shear`, `translate`, `perspective` | 0.0, 0.5, 0.0, 0.1, 0.0 |
| `MixUp` | `mixup` | 0.0 |
| `Albumentations` (blur/CLAHE/gray — only if `albumentations` installed) | — | p≈0.01 each |
| `RandomHSV` | `hsv_h`, `hsv_s`, `hsv_v` | 0.015, 0.7, 0.4 |
| `RandomFlip` vertical / horizontal | `flipud`, `fliplr` | 0.0, 0.5 |

Defaults match Ultralytics exactly (see `yolo.data.augment.DEFAULT_HYP`). Mosaic /
mixup / copy-paste are automatically **closed for the final epochs**
(`close_mosaic`, default 10) so the model finishes on clean images. Override any
hyper-parameter via the `hyp` dict:

```python
model.train(
    data_train="datasets/mydata/images/train",
    hyp={"degrees": 10.0, "mixup": 0.1, "scale": 0.7, "flipud": 0.1},
    close_mosaic=10,
)
```

To confirm the augmentation keeps labels aligned with the pixels, render a grid
that draws the *post-augmentation* boxes back onto the images:

```bash
# default v8 pipeline
python tools/visualize_augment.py --data datasets/mydata/images/train --n 9 --out aug.png
# exercise every transform at once
python tools/visualize_augment.py --data datasets/mydata/images/train \
    --degrees 20 --shear 10 --perspective 0.0006 --mixup 1.0 --flipud 0.5 --copy-paste 0.5 --out aug_full.png
# mosaic off / all augmentation off
python tools/visualize_augment.py --data datasets/mydata/images/train --no-mosaic --out plain.png
python tools/visualize_augment.py --data datasets/mydata/images/train --no-augment --out raw.png
```

If the drawn boxes sit on the objects, the transforms + label bookkeeping are correct.

## Layout

```
yolo/
  cfg/         model topology + scale multipliers
  nn/          modules.py (blocks) + tasks.py (DetectionModel, parse_model)
  data/        augment.py, dataset.py, build.py
  engine/      predictor.py, trainer.py, validator.py
  utils/       ops.py, tal.py, loss.py, metrics.py, plotting.py, checkpoint.py
  model.py     high-level YOLO API
examples/      predict.py, train.py
tools/         convert_weights.py, visualize_augment.py
tests/         test_forward.py, test_augment.py
```

## Tests

```bash
python tests/test_forward.py          # or: python -m pytest tests/ -q
```

Covers: official parameter counts, forward output shape, bit-exact checkpoint
round trip, the predict API, and loss convergence on an overfit batch.

## Polygon + distance extension

An extended head (`YOLOPolygon`) adds per-object **polygon** (star-shaped) and
**distance** prediction on top of the standard box + class outputs, keeping the box
and class branches unchanged. See **[POLYGON.md](POLYGON.md)** for the full design
(datasets, parsers, `DetectPolygon` head, loss, decode, and the bbox-F1 metric).

```python
from yolo.poly_model import YOLOPolygon
model = YOLOPolygon("yolov8s.pt", nc=3)
model.train(poly_train="poly/images/train", dist_train="polyd/images/train", epochs=100)
results = model.predict("img.jpg")   # bbox, cls, conf, distance, polygon
```

## Deployment, training & inference features

| Area | Features |
|------|----------|
| Export | `model.export("torchscript" / "onnx" / "int8")` — Conv+BN fuse, ONNX (legacy exporter, optional dynamic axes), INT8 ONNX via ONNX Runtime (~4× smaller). `model.fuse()` for faster inference. |
| Training | AMP (CUDA), multi-GPU (`device="0,1"`, DataParallel), EMA, resume (`resume=path`), cosine LR (`cos_lr=True`), image caching (`cache="ram"`/`"disk"`), best/last checkpoints. |
| Validation | Per-class P/R/F1 + mAP@0.5 / mAP@0.5:0.95 table, confusion matrix, PR/F1/P/R curves, per-image speed, `metrics.json` (`model.val(..., plots=True, save_json=True)`). |
| Inference | Image / array / **directory** / glob sources, `save`, `save_txt`, `save_conf`, `show`, and backbone **feature-map visualization** (`model.predict(src, save=True, save_txt=True, visualize=True)`). |
| Logging | TensorBoard (`tensorboard=True`), `results.csv` + `results.png` curves, `labels.png` class histogram, `train_batch0.jpg`. |

```python
from yolo import YOLO
m = YOLO("yolov8n.pt")
m.train("data/images/train", data_val="data/images/val", epochs=100,
        cos_lr=True, ema=True, amp=True, cache="ram", tensorboard=True)   # Steps 2 & 5
m.val("data/images/val", plots=True, save_json=True)                      # Step 3
m.predict("data/images/test", save=True, save_txt=True, visualize=True)   # Step 4
m.export("onnx", dynamic=True)                                            # Step 1
```

CLI: `examples/export.py`, and `examples/predict.py` (`--save --save-txt --visualize`).

## Notes & limitations

- Detection only (no seg/pose/cls/obb).
- BatchNorm needs a real batch size; like the original, don't train at `batch=1`.
- The full `v8_transforms` augmentation chain is implemented (mosaic, copy-paste,
  random-perspective, mixup, albumentations, HSV, flips). CopyPaste is a box-level
  analog of the official segment-mask version, and Albumentations is a no-op unless
  the optional `albumentations` package is installed — both matching Ultralytics'
  detection defaults (`copy_paste=0`).
- Not affiliated with or endorsed by Ultralytics. YOLOv8 weights you load remain
  subject to their original (AGPL-3.0) license.
```
