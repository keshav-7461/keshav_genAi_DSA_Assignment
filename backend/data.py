"""MNIST loading from the raw IDX .gz files in data/raw (no torchvision needed)."""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an MNIST image file")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an MNIST label file")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(split: str = "train", raw_dir: Path = RAW_DIR):
    """Returns (images float32 in [0,1] with shape N x 28 x 28, labels uint8)."""
    prefix = {"train": "train", "test": "t10k"}[split]
    images = _read_images(raw_dir / f"{prefix}-images-idx3-ubyte.gz")
    labels = _read_labels(raw_dir / f"{prefix}-labels-idx1-ubyte.gz")
    return images.astype(np.float32) / 255.0, labels
