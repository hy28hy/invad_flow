from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Memory-mapped (when supported) raw EfficientNet feature cache."""

    def __init__(self, path: str | Path, split: str = "all"):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Feature cache not found: {self.path}. Run cache_features.py first."
            )
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)

        required = {"features", "labels", "filenames", "split", "mean", "std", "meta"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Malformed feature cache; missing keys: {sorted(missing)}")
        self.features = payload["features"]
        self.labels = payload["labels"].long()
        self.filenames = list(payload["filenames"])
        self.split_flags = payload["split"].to(torch.uint8)
        self.mean = payload["mean"].float()
        self.std = payload["std"].float()
        self.count = payload.get("count")
        self.meta = dict(payload["meta"])

        if split == "train":
            self.indices = torch.where(self.split_flags == 0)[0].tolist()
        elif split in {"val", "validation"}:
            self.indices = torch.where(self.split_flags == 1)[0].tolist()
        elif split == "all":
            self.indices = list(range(len(self.labels)))
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        real_index = self.indices[index]
        return {
            "feature": self.features[real_index],
            "clslabel": self.labels[real_index],
            "filename": self.filenames[real_index],
        }

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.features.shape[1:])


class FeatureNormalizer(nn.Module):
    """Per-class, per-channel statistics fitted only on training-normal features."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, floor: float = 1e-4):
        super().__init__()
        if mean.ndim != 4 or std.shape != mean.shape:
            raise ValueError("mean/std must have shape [num_classes, C, 1, 1]")
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(float(floor)))

    def _stats(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = labels.long().to(self.mean.device)
        if labels.min() < 0 or labels.max() >= self.mean.shape[0]:
            raise IndexError("Class label is outside the cached normalizer range")
        return self.mean.index_select(0, labels), self.std.index_select(0, labels)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(labels)
        return (x - mean) / std

    def decode(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(labels)
        return x * std + mean


def normalizer_from_cache(dataset: CachedFeatureDataset, floor: float = 1e-4) -> FeatureNormalizer:
    return FeatureNormalizer(dataset.mean, dataset.std, floor=floor)


def validate_cache_compatibility(dataset: CachedFeatureDataset, config: dict) -> None:
    expected_size = int(config["data"]["img_size"])
    expected_shape = (272, expected_size // 16, expected_size // 16)
    if dataset.feature_shape != expected_shape:
        raise ValueError(
            f"Cache shape {dataset.feature_shape} does not match expected {expected_shape}"
        )
    cached_data = dataset.meta.get("data", {})
    for key in ("dataset_name", "img_size", "transform_type"):
        if key in cached_data and cached_data[key] != config["data"].get(key):
            raise ValueError(
                f"Cache mismatch for data.{key}: {cached_data[key]!r} != "
                f"{config['data'].get(key)!r}"
            )
