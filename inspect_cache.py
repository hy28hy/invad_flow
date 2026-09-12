from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from cache_features import cache_diagnostics
from src.datasets import build_dataset


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    path = Path(args.cache or config["cache"]["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {"features", "labels", "filenames", "split", "mean", "std", "meta"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Malformed cache; missing keys: {sorted(missing)}")
    dtype_name = str(config["cache"].get("feature_dtype", "float32")).lower()
    expected_dtype = torch.float32 if dtype_name in {"float32", "fp32"} else torch.float16
    data_cfg = dict(config["data"])
    data_cfg["train"] = True
    expected_samples = len(build_dataset(**data_cfg))
    report = cache_diagnostics(
        payload["features"], payload["labels"], payload["split"],
        payload["mean"], payload["std"], list(payload["filenames"]),
        int(config["model"]["num_classes"]),
        expected_dtype=expected_dtype, expected_samples=expected_samples,
    )
    train_split = str(config["cache"].get("train_split", "all"))
    report.update({
        "statistics_split": payload["meta"].get("statistics_split", "train"),
        "configured_train_split": train_split,
        "configured_training_samples": (
            len(payload["features"])
            if train_split == "all" else int((payload["split"] == 0).sum())
        ),
    })
    report.update({"cache": str(path.resolve()), "bytes": path.stat().st_size})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fail-fast feature-cache integrity check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default=None)
    main(parser.parse_args())
