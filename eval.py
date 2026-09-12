from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adeval.au_pro import calculate_au_pro
from src.backbones import get_backbone
from src.datasets import build_dataset
from src.feature_cache import FeatureNormalizer
from src.flow_model import build_flow_model
from src.operators import (
    deterministic_anchor_bank,
    expand_shared_anchor,
    postprocess_map,
    probe_flow_maps,
    topk_pool,
)

METRIC_ORDER = (
    "I-AUROC", "I-AP", "I-F1Max", "P-AUROC", "P-AP", "P-F1Max", "AU-PRO", "mAD"
)


def validate_runtime_config(runtime: dict, trained: dict) -> None:
    """Fail before evaluation when the extractor contract changed."""
    for key in ("dataset_name", "img_size", "transform_type", "category"):
        if runtime["data"].get(key) != trained["data"].get(key):
            raise ValueError(
                f"Runtime data.{key}={runtime['data'].get(key)!r} differs from "
                f"checkpoint value {trained['data'].get(key)!r}"
            )
    runtime_root = Path(runtime["data"]["data_root"]).resolve()
    trained_root = Path(trained["data"]["data_root"]).resolve()
    if runtime_root != trained_root:
        raise ValueError(f"Runtime data root {runtime_root} != checkpoint root {trained_root}")
    if runtime["backbone"] != trained["backbone"]:
        raise ValueError("Runtime backbone config differs from the training checkpoint")


def percent_summary(metrics: dict[str, float], fps: float | None) -> dict[str, float | None]:
    summary = {key: 100.0 * float(metrics[key]) for key in METRIC_ORDER}
    summary["FPS"] = fps
    return summary


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if not args.config:
        raise ValueError("Provide --checkpoint or --config")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return Path(config["logging"]["save_dir"]) / "flow_latest.pth"


def f1_max(target: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(target.astype(np.uint8), score)
    values = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.nanmax(values))


def metric_bundle(labels: np.ndarray, masks: np.ndarray,
                  image_scores: np.ndarray, maps: np.ndarray) -> dict[str, float]:
    flat_mask = (masks.reshape(-1) > 0).astype(np.uint8)
    flat_map = maps.reshape(-1)
    au_pro, _ = calculate_au_pro(
        gts=(masks > 0).astype(np.uint8), predictions=maps,
        integration_limit=0.3, num_thresholds=200,
    )
    return {
        "I-AUROC": float(roc_auc_score(labels, image_scores)),
        "I-AP": float(average_precision_score(labels, image_scores)),
        "I-F1Max": f1_max(labels, image_scores),
        "P-AUROC": float(roc_auc_score(flat_mask, flat_map)),
        "P-AP": float(average_precision_score(flat_mask, flat_map)),
        "P-F1Max": f1_max(flat_mask, flat_map),
        "AU-PRO": float(au_pro),
    }


def macro_average(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = next(iter(per_class.values())).keys()
    result = {key: float(np.mean([row[key] for row in per_class.values()])) for key in keys}
    result["mAD"] = float(np.mean(list(result.values())))
    return result


def evaluate_operator(records: dict, operator: str) -> dict:
    labels = np.asarray(records["labels"], dtype=np.uint8)
    class_labels = np.asarray(records["class_labels"], dtype=np.int64)
    class_names = np.asarray(records["class_names"])
    masks = np.concatenate(records["masks"], axis=0)
    maps = np.concatenate(records[operator]["maps"], axis=0)
    image_scores = np.concatenate(records[operator]["scores"], axis=0)

    per_class = {}
    for class_label in np.unique(class_labels):
        selected = class_labels == class_label
        name_values = np.unique(class_names[selected])
        name = str(name_values[0]) if len(name_values) else str(class_label)
        per_class[name] = metric_bundle(
            labels[selected], masks[selected], image_scores[selected], maps[selected]
        )
    return {
        "macro_average": macro_average(per_class),
        "per_class": per_class,
    }


def main(args: argparse.Namespace) -> None:
    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            runtime_config = yaml.safe_load(handle)
    else:
        runtime_config = copy.deepcopy(train_config)
    validate_runtime_config(runtime_config, train_config)

    eval_cfg = runtime_config["evaluation"]
    operator = args.operator or eval_cfg.get("operator", "angular")
    if operator not in {"angular", "curvature", "both"}:
        raise ValueError("operator must be angular, curvature, or both")
    requested = ["angular", "curvature"] if operator == "both" else [operator]

    seed = int(runtime_config["meta"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(runtime_config["meta"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable")

    data_cfg = copy.deepcopy(runtime_config["data"])
    data_cfg.update(train=False, anom_only=False, normal_only=False)
    dataset = build_dataset(**data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
    )

    feature_shape = tuple(int(v) for v in checkpoint["feature_shape"])
    model = build_flow_model(train_config, feature_shape).to(device).eval()
    state_key = "model" if args.no_ema else "ema"
    model.load_state_dict(checkpoint[state_key], strict=True)
    normalizer = FeatureNormalizer(
        checkpoint["normalizer"]["mean"], checkpoint["normalizer"]["std"]
    ).to(device).eval()
    backbone = get_backbone(**runtime_config["backbone"]).to(device).eval()
    backbone.requires_grad_(False)
    anchor_mode = str(eval_cfg.get("anchor_mode", "shared"))
    if anchor_mode != "shared":
        raise ValueError("Only path-independent evaluation.anchor_mode='shared' is supported")
    num_anchors = int(args.num_anchors or eval_cfg.get("num_anchors", 1))
    anchor_bank = deterministic_anchor_bank(
        feature_shape, seed=int(eval_cfg.get("anchor_seed", seed + 17)),
        num_anchors=num_anchors, device=device, dtype=torch.float32,
    )

    records = {
        "labels": [], "class_labels": [], "class_names": [], "filenames": [], "masks": [],
        **{name: {"maps": [], "raw_maps": [], "scores": []} for name in requested},
    }
    timed_images = 0
    elapsed = 0.0
    warmup_batches = int(eval_cfg.get("warmup_batches", 2))
    amp_dtype = torch.bfloat16 if runtime_config["meta"].get("amp", "bf16") == "bf16" else torch.float16

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Evaluating {operator}")):
            images = batch["samples"].to(device, non_blocking=True)
            labels = batch["clslabels"].to(device, non_blocking=True).long()
            filenames = list(batch["filenames"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()

            # Match cache extraction: backbone and normalization remain FP32.
            raw_feature, _ = backbone(images)
            feature = normalizer.encode(raw_feature.float(), labels)
            raw_maps = {name: torch.zeros(
                feature.shape[0], feature.shape[2], feature.shape[3],
                device=device, dtype=torch.float32,
            ) for name in requested}
            for shared_anchor in anchor_bank:
                anchor = expand_shared_anchor(shared_anchor, feature.shape[0])
                anchor_maps = probe_flow_maps(
                    model, feature, anchor,
                    labels if model.class_conditioned else None,
                    need_angular="angular" in requested,
                    need_curvature="curvature" in requested,
                    probe_t=float(eval_cfg.get("probe_t", 0.4)),
                    curvature_dt=float(eval_cfg.get("curvature_dt", 0.2)),
                    curvature_mode=eval_cfg.get("curvature_mode", "normalized_change"),
                    amp_dtype=amp_dtype if device.type == "cuda" else None,
                )
                for name in requested:
                    raw_maps[name].add_(anchor_maps[name])
            for name in requested:
                raw_maps[name].div_(num_anchors)
            # Geometry, resize, filtering and pooling stay FP32 so score
            # rankings are not quantized to BF16/FP16 steps.
            processed = {
                name: postprocess_map(
                    raw_maps[name].float(), (images.shape[-2], images.shape[-1]),
                    sigma=float(eval_cfg.get("gaussian_sigma", images.shape[-1] / 64.0)),
                )
                for name in requested
            }
            scores = {
                name: topk_pool(value, float(eval_cfg.get("topk_fraction", 0.01)))
                for name, value in processed.items()
            }

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - start
            if batch_index >= warmup_batches:
                elapsed += batch_elapsed
                timed_images += images.shape[0]

            records["labels"].extend(batch["labels"].cpu().numpy().tolist())
            records["class_labels"].extend(batch["clslabels"].cpu().numpy().tolist())
            records["class_names"].extend(str(name) for name in batch["clsnames"])
            records["filenames"].extend(str(name) for name in filenames)
            mask = batch["masks"].cpu().numpy()
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask[:, 0]
            records["masks"].append(mask)
            for name in requested:
                if args.save_raw_scores:
                    records[name]["raw_maps"].append(raw_maps[name].float().cpu().numpy())
                records[name]["maps"].append(processed[name].float().cpu().numpy())
                records[name]["scores"].append(scores[name].float().cpu().numpy())

    fps = timed_images / elapsed if elapsed > 0 else None
    results = {name: evaluate_operator(records, name) for name in requested}
    for name in requested:
        results[name]["summary_percent"] = percent_summary(
            results[name]["macro_average"], fps
        )
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "operator": operator,
        "nfe": num_anchors * (2 if "curvature" in requested else 1),
        "samples": len(records["labels"]),
        "compute_fps_excluding_dataloader": fps,
        "metric_order": [*METRIC_ORDER, "FPS"],
        "fps_scope": "backbone + flow operator + FP32 postprocess; excludes DataLoader and metrics",
        "settings": {
            key: eval_cfg.get(key) for key in (
                "probe_t", "curvature_dt", "curvature_mode",
                "gaussian_sigma", "topk_fraction", "anchor_seed",
                "anchor_mode", "num_anchors"
            )
        },
        "results": results,
        "summary_percent": {
            name: results[name]["summary_percent"] for name in requested
        },
    }
    report["settings"].update({"anchor_mode": anchor_mode, "num_anchors": num_anchors})
    reference = runtime_config.get("reference_baseline")
    if reference:
        baseline = reference["metrics_percent"]
        report["reference_baseline"] = reference
        report["delta_vs_reference_percent"] = {
            name: {
                key: (results[name]["summary_percent"][key] - baseline[key])
                for key in (*METRIC_ORDER, "FPS")
                if results[name]["summary_percent"][key] is not None and key in baseline
            }
            for name in requested
        }
    if args.output:
        output = Path(args.output)
    else:
        configured = Path(eval_cfg.get("output", "results/eval_angular.json"))
        configured_operator = eval_cfg.get("operator", "angular")
        output = configured if operator == configured_operator else configured.with_name(
            f"eval_{operator}{configured.suffix}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output)
    if args.save_raw_scores:
        raw_output = output.with_suffix(".raw_scores.npz")
        raw_payload = {
            "labels": np.asarray(records["labels"], dtype=np.uint8),
            "class_labels": np.asarray(records["class_labels"], dtype=np.int64),
            "class_names": np.asarray(records["class_names"]),
            "filenames": np.asarray(records["filenames"]),
            "masks": np.concatenate(records["masks"], axis=0).astype(np.uint8),
        }
        for name in requested:
            raw_payload[f"{name}_feature_maps"] = np.concatenate(
                records[name]["raw_maps"], axis=0
            )
            raw_payload[f"{name}_anomaly_maps"] = np.concatenate(
                records[name]["maps"], axis=0
            )
            raw_payload[f"{name}_image_scores"] = np.concatenate(
                records[name]["scores"], axis=0
            )
        np.savez_compressed(raw_output, **raw_payload)
        report["raw_scores"] = str(raw_output.resolve())
        temporary = output.with_suffix(output.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate native Flow Matching operators")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None, help="Runtime config; also infers latest checkpoint")
    parser.add_argument("--operator", choices=["angular", "curvature", "both"], default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--num-anchors", "--num_anchors", type=int, default=None)
    parser.add_argument("--save-raw-scores", "--save_raw_scores", action="store_true")
    main(parser.parse_args())
