from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from cache_features import cache_diagnostics


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    path = Path(args.cache or config["cache"]["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {"features", "labels", "filenames", "split", "mean", "std", "meta"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Malformed cache; missing keys: {sorted(missing)}")
    report = cache_diagnostics(
        payload["features"], payload["labels"], payload["split"],
        payload["mean"], payload["std"], list(payload["filenames"]),
        int(config["model"]["num_classes"]),
    )
    report.update({"cache": str(path.resolve()), "bytes": path.stat().st_size})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fail-fast feature-cache integrity check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default=None)
    main(parser.parse_args())
