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
| Data | YOLO-format `YOLODataset`, letterbox + **mosaic (4-image)** + HSV/flip augmentation, collate function, `DataLoader` builder. |
| Engine | `DetectionPredictor` (preprocess → forward → NMS → `Results`), `DetectionTrainer` (warmup, LR schedule, param-group weight decay), `DetectionValidator` (COCO-style mAP@0.5 / mAP@0.5:0.95). |

## Augmentation & visualization

Training uses **mosaic** (4 images stitched into one, then centre-cropped), HSV
jitter, horizontal flip and letterbox. Mosaic is on by default and is
automatically **closed for the final epochs** (`close_mosaic`, default 10) so the
model finishes on clean images:

```python
model.train(..., mosaic=1.0, close_mosaic=10)   # both tunable
```

To confirm augmentation keeps labels aligned with the pixels, render a grid that
draws the *post-augmentation* boxes back onto the images:

```bash
# mosaic samples (default)
python tools/visualize_augment.py --data datasets/mydata/images/train --n 9 --out mosaic.png
# letterbox-only augmentation
python tools/visualize_augment.py --data datasets/mydata/images/train --no-mosaic --out plain.png
# raw images, no augmentation
python tools/visualize_augment.py --data datasets/mydata/images/train --no-augment --out raw.png
```

If the drawn boxes sit on the objects, mosaic + label bookkeeping are correct.

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

## Notes & limitations

- Detection only (no seg/pose/cls/obb).
- BatchNorm needs a real batch size; like the original, don't train at `batch=1`.
- Mosaic, HSV, horizontal flip and letterbox are implemented; mixup / copy-paste /
  random-perspective are not.
- Not affiliated with or endorsed by Ultralytics. YOLOv8 weights you load remain
  subject to their original (AGPL-3.0) license.
```
