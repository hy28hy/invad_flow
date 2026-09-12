from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import get_backbone
from src.datasets import build_dataset


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stratified_split(labels: torch.Tensor, val_fraction: float,
                     seed: int) -> torch.Tensor:
    flags = torch.zeros(len(labels), dtype=torch.uint8)
    generator = torch.Generator().manual_seed(seed)
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        if val_fraction <= 0 or len(indices) < 2:
            continue
        n_val = min(len(indices) - 1, max(1, round(len(indices) * val_fraction)))
        order = torch.randperm(len(indices), generator=generator)
        flags[indices[order[:n_val]]] = 1
    return flags


def compute_stats(features: torch.Tensor, labels: torch.Tensor,
                  split: torch.Tensor, num_classes: int,
                  statistics_split: str = "all") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if statistics_split not in {"all", "train"}:
        raise ValueError("cache.statistics_split must be 'all' or 'train'")
    channels = features.shape[1]
    means = torch.empty(num_classes, channels, 1, 1, dtype=torch.float32)
    stds = torch.empty_like(means)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for cls in range(num_classes):
        mask = labels == cls
        if statistics_split == "train":
            mask &= split == 0
        selected = features[mask].float()
        if selected.numel() == 0:
            raise RuntimeError(f"No training-normal feature found for class {cls}")
        if not torch.isfinite(selected).all():
            raise RuntimeError(f"Class {cls} contains NaN/Inf before statistics")
        counts[cls] = selected.shape[0]
        means[cls, :, 0, 0] = selected.mean(dim=(0, 2, 3))
        raw_std = selected.std(dim=(0, 2, 3), unbiased=False)
        if not torch.isfinite(raw_std).all():
            raise RuntimeError(f"Class {cls} produced non-finite FP32 standard deviations")
        zero_channels = torch.where(raw_std <= 0)[0]
        if zero_channels.numel():
            raise RuntimeError(
                f"Class {cls} has {zero_channels.numel()} zero-variance channels: "
                f"{zero_channels[:20].tolist()}"
            )
        stds[cls, :, 0, 0] = raw_std
    return means, stds, counts


def cache_diagnostics(features: torch.Tensor, labels: torch.Tensor,
                      split: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                      filenames: list[str], num_classes: int, *,
                      expected_dtype: torch.dtype = torch.float32,
                      expected_samples: int | None = None) -> dict:
    expected = len(features)
    if not (len(labels) == len(split) == len(filenames) == expected):
        raise RuntimeError(
            "Cache cardinality mismatch: "
            f"features={expected}, labels={len(labels)}, split={len(split)}, "
            f"filenames={len(filenames)}"
        )
    if expected == 0:
        raise RuntimeError("Feature cache is empty")
    if expected_samples is not None and expected != expected_samples:
        raise RuntimeError(
            f"Incomplete cache: found {expected} samples, expected {expected_samples}"
        )
    if features.dtype != expected_dtype:
        raise RuntimeError(
            f"Persisted features must be {expected_dtype}, got {features.dtype}"
        )
    duplicate_count = expected - len(set(filenames))
    if duplicate_count:
        raise RuntimeError(f"Cache contains {duplicate_count} duplicate filenames")
    for name, tensor in (("features", features), ("mean", mean), ("std", std)):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
    if mean.dtype != torch.float32 or std.dtype != torch.float32:
        raise RuntimeError(f"Statistics must be FP32, got mean={mean.dtype}, std={std.dtype}")
    if (std <= 0).any():
        indices = torch.nonzero(std <= 0, as_tuple=False)[:20].tolist()
        raise RuntimeError(f"Statistics contain zero/non-positive std at {indices}")
    if not torch.all((split == 0) | (split == 1)):
        raise RuntimeError("Split flags contain values other than 0/1")
    if labels.min().item() < 0 or labels.max().item() >= num_classes:
        raise RuntimeError(
            f"Labels outside [0, {num_classes - 1}]: "
            f"min={labels.min().item()}, max={labels.max().item()}"
        )

    per_class = []
    for cls in range(num_classes):
        selected = features[labels == cls].float()
        train_count = int(((labels == cls) & (split == 0)).sum())
        val_count = int(((labels == cls) & (split == 1)).sum())
        if selected.numel() == 0 or train_count == 0:
            raise RuntimeError(f"Class {cls} is absent or has no training samples")
        flat = selected.flatten()
        class_std = std[cls].flatten()
        # torch.quantile has an implementation limit for very large tensors.
        # Deterministic striding is sufficient for diagnostics; moments below
        # still use every element.
        quantile_limit = 1_000_000
        stride = max(1, (flat.numel() + quantile_limit - 1) // quantile_limit)
        quantile_sample = flat[::stride][:quantile_limit]
        quantiles = torch.quantile(
            quantile_sample, torch.tensor([0.01, 0.5, 0.99])
        )
        per_class.append({
            "class_id": cls,
            "samples": int(selected.shape[0]),
            "train": train_count,
            "validation": val_count,
            "feature_min": float(flat.min()),
            "feature_max": float(flat.max()),
            "feature_mean": float(flat.mean()),
            "feature_std": float(flat.std(unbiased=False)),
            "feature_q01": float(quantiles[0]),
            "feature_q50": float(quantiles[1]),
            "feature_q99": float(quantiles[2]),
            "channel_std_min": float(class_std.min()),
            "channel_std_mean": float(class_std.mean()),
            "channel_std_max": float(class_std.max()),
        })

    return {
        "passed": True,
        "samples": expected,
        "feature_dtype": str(features.dtype),
        "statistics_dtype": str(mean.dtype),
        "feature_shape": list(features.shape[1:]),
        "unique_filenames": len(set(filenames)),
        "duplicate_filenames": duplicate_count,
        "train": int((split == 0).sum()),
        "validation": int((split == 1).sum()),
        "per_class": per_class,
    }


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed = int(config["meta"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(config["meta"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable")

    data_cfg = copy.deepcopy(config["data"])
    data_cfg["train"] = True
    dataset = build_dataset(**data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(config["cache"].get("batch_size", data_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    backbone = get_backbone(**config["backbone"]).to(device).eval()
    backbone.requires_grad_(False)
    dtype_name = str(config["cache"].get("feature_dtype", "float32")).lower()
    dtype_by_name = {"float32": torch.float32, "fp32": torch.float32,
                     "float16": torch.float16, "fp16": torch.float16}
    if dtype_name not in dtype_by_name:
        raise ValueError("cache.feature_dtype must be float32 or float16")
    feature_dtype = dtype_by_name[dtype_name]
    statistics_split = str(config["cache"].get("statistics_split", "all"))
    features, statistics_features, labels, filenames = [], [], [], []
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Caching EfficientNet features")):
            images = batch["samples"].to(device, non_blocking=True)
            # Keep the frozen extractor and the repaired cache in FP32,
            # matching the original InvAD feature path without quantization.
            feat, _ = backbone(images)
            feat_fp32 = feat.float()
            if not torch.isfinite(feat_fp32).all():
                raise RuntimeError(
                    f"NaN/Inf from backbone at batch {batch_index}: "
                    f"{list(batch['filenames'])[:4]}"
                )
            feat_cpu = feat_fp32.cpu()
            persisted_feat = feat_cpu.to(feature_dtype)
            if not torch.isfinite(persisted_feat).all():
                raise RuntimeError(
                    f"{feature_dtype} conversion overflow/NaN at batch {batch_index}: "
                    f"{list(batch['filenames'])[:4]}"
                )
            features.append(persisted_feat)
            # Statistics always use the unquantized FP32 extractor output.
            statistics_features.append(feat_cpu)
            labels.append(batch["clslabels"].long().cpu())
            filenames.extend(str(name) for name in batch["filenames"])

    all_features = torch.cat(features, dim=0).contiguous()
    all_statistics_features = torch.cat(statistics_features, dim=0).contiguous()
    all_labels = torch.cat(labels, dim=0).contiguous()
    split = stratified_split(
        all_labels,
        float(config["cache"].get("validation_fraction", 0.1)),
        int(config["cache"].get("split_seed", seed)),
    )
    mean, std, count = compute_stats(
        all_statistics_features, all_labels, split, int(config["model"]["num_classes"]),
        statistics_split=statistics_split,
    )
    del all_statistics_features
    diagnostics = cache_diagnostics(
        all_features, all_labels, split, mean, std, filenames,
        int(config["model"]["num_classes"]),
        expected_dtype=feature_dtype, expected_samples=len(dataset),
    )
    train_split = str(config["cache"].get("train_split", "all"))
    diagnostics.update({
        "statistics_split": statistics_split,
        "configured_train_split": train_split,
        "configured_training_samples": (
            len(all_features) if train_split == "all" else int((split == 0).sum())
        ),
    })

    output = Path(args.output or config["cache"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": all_features,
        "labels": all_labels,
        "filenames": filenames,
        "split": split,
        "mean": mean,
        "std": std,
        "count": count,
        "meta": {
            "format_version": 2,
            "data": {key: data_cfg.get(key) for key in (
                "dataset_name", "data_root", "img_size", "transform_type", "category"
            )},
            "backbone": copy.deepcopy(config["backbone"]),
            "torch_version": torch.__version__,
            "feature_shape": list(all_features.shape[1:]),
            "feature_dtype": str(feature_dtype),
            "source_samples": len(dataset),
            "statistics_split": statistics_split,
            "validation_fraction": float(config["cache"].get("validation_fraction", 0.1)),
            "split_seed": int(config["cache"].get("split_seed", seed)),
            "diagnostics_passed": True,
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    diagnostics_path = output.with_suffix(output.suffix + ".diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)
    summary = {
        "path": str(output),
        "diagnostics_path": str(diagnostics_path),
        **diagnostics,
        "class_statistics_counts": count.tolist(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache frozen EfficientNet-B4 features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
