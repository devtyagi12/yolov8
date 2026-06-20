"""Dataloader construction helpers."""

from torch.utils.data import DataLoader

from .dataset import YOLODataset


def build_dataloader(path, imgsz=640, batch=16, augment=False, workers=4, shuffle=None, hyp=None, mosaic=None):
    """Create a ``DataLoader`` over a :class:`YOLODataset`."""
    dataset = YOLODataset(path, imgsz=imgsz, augment=augment, hyp=hyp, mosaic=mosaic)
    if shuffle is None:
        shuffle = augment
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        collate_fn=YOLODataset.collate_fn,
        drop_last=False,
    )
