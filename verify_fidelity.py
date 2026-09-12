from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.feature_cache import CachedFeatureDataset, FeatureNormalizer, validate_checkpoint_cache
from src.flow_model import build_flow_model


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if not args.config:
        raise ValueError("Provide --checkpoint or --config")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return Path(config["logging"]["save_dir"]) / "flow_latest.pth"


@torch.inference_mode()
def euler_generate(model, initial: torch.Tensor, labels: torch.Tensor,
                   steps: int, amp_dtype: torch.dtype) -> torch.Tensor:
    x = initial.clone()
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((x.shape[0],), index * dt, device=x.device, dtype=torch.float32)
        with torch.autocast(
            device_type=x.device.type, dtype=amp_dtype, enabled=x.device.type == "cuda"
        ):
            velocity = model(x, t, labels)
        x = x + dt * velocity.float()
    return x


def sliced_patch_wasserstein(a: torch.Tensor, b: torch.Tensor, *,
                             projections: int, max_patches: int,
                             seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    a = a.permute(0, 2, 3, 1).reshape(-1, a.shape[1]).float()
    b = b.permute(0, 2, 3, 1).reshape(-1, b.shape[1]).float()
    count = min(len(a), len(b), max_patches)
    ia = torch.randperm(len(a), generator=generator)[:count]
    ib = torch.randperm(len(b), generator=generator)[:count]
    direction = torch.randn(a.shape[1], projections, generator=generator)
    direction = direction / direction.norm(dim=0, keepdim=True).clamp_min(1e-8)
    pa = (a[ia] @ direction).sort(dim=0).values
    pb = (b[ib] @ direction).sort(dim=0).values
    return (pa - pb).abs().mean().item()


def main(args: argparse.Namespace) -> None:
    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    steps = [int(value.strip()) for value in args.steps.split(",") if value.strip()]
    if len(steps) != 2 or min(steps) <= 0 or steps[0] >= steps[1]:
        raise ValueError("--steps must be two increasing positive integers, e.g. 5,20")
    coarse_steps, reference_steps = steps
    cache_path = args.cache or config["cache"]["path"]
    dataset = CachedFeatureDataset(cache_path, split="val")
    validate_checkpoint_cache(checkpoint, dataset)
    if len(dataset) == 0:
        raise RuntimeError("No normal validation features are present in the cache")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(config["meta"].get("device", "cuda"))
    model = build_flow_model(config, tuple(checkpoint["feature_shape"])).to(device).eval()
    model.load_state_dict(checkpoint["ema"], strict=True)
    normalizer = FeatureNormalizer(
        checkpoint["normalizer"]["mean"], checkpoint["normalizer"]["std"]
    ).to(device)
    amp_dtype = torch.bfloat16 if config["meta"].get("amp", "bf16") == "bf16" else torch.float16

    real_all, noise_all, coarse_all, reference_all = [], [], [], []
    generator = torch.Generator(device=device).manual_seed(int(config["meta"]["seed"]) + 1009)
    for batch in loader:
        labels = batch["clslabel"].to(device).long()
        real = normalizer.encode(batch["feature"].to(device).float(), labels)
        initial = torch.randn(real.shape, generator=generator, device=device)
        coarse = euler_generate(model, initial, labels, coarse_steps, amp_dtype)
        reference = euler_generate(model, initial, labels, reference_steps, amp_dtype)
        real_all.append(real.cpu())
        noise_all.append(initial.cpu())
        coarse_all.append(coarse.cpu())
        reference_all.append(reference.cpu())

    real = torch.cat(real_all)
    noise = torch.cat(noise_all)
    coarse = torch.cat(coarse_all)
    reference = torch.cat(reference_all)
    finite = bool(torch.isfinite(coarse).all() and torch.isfinite(reference).all())
    spatial_mean_mae = (coarse.mean(0) - real.mean(0)).abs().mean().item()
    spatial_logstd_mae = (
        (coarse.std(0, unbiased=False).clamp_min(1e-5).log()
         - real.std(0, unbiased=False).clamp_min(1e-5).log()).abs().mean().item()
    )
    euler_relative_error = (
        (coarse - reference).square().mean().sqrt()
        / reference.square().mean().sqrt().clamp_min(1e-8)
    ).item()
    swd_gen = sliced_patch_wasserstein(
        coarse, real, projections=args.projections,
        max_patches=args.max_patches, seed=123,
    )
    swd_noise = sliced_patch_wasserstein(
        noise, real, projections=args.projections,
        max_patches=args.max_patches, seed=123,
    )
    swd_ratio = swd_gen / max(swd_noise, 1e-8)

    thresholds = config.get("fidelity", {})
    checks = {
        "finite": finite,
        "spatial_mean_mae": spatial_mean_mae <= float(thresholds.get("max_mean_mae", 0.15)),
        "spatial_logstd_mae": spatial_logstd_mae <= float(thresholds.get("max_logstd_mae", 0.20)),
        f"euler_{coarse_steps}_vs_{reference_steps}": euler_relative_error <= float(
            thresholds.get("max_euler_relative_error", 0.15)
        ),
        "swd_improvement": swd_ratio <= float(thresholds.get("max_swd_ratio", 0.70)),
    }
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "cache": str(Path(cache_path).resolve()),
        "steps": steps,
        "normal_validation_samples": len(real),
        "finite": finite,
        "spatial_mean_mae": spatial_mean_mae,
        "spatial_logstd_mae": spatial_logstd_mae,
        f"euler_{coarse_steps}_vs_{reference_steps}_relative_error": euler_relative_error,
        "sliced_patch_wasserstein_gen_to_normal": swd_gen,
        "sliced_patch_wasserstein_noise_to_normal": swd_noise,
        "swd_ratio": swd_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output) if args.output else checkpoint_path.parent / "fidelity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal-only five-step Euler fidelity check")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None, help="Infers <logging.save_dir>/flow_latest.pth")
    parser.add_argument("--cache", default=None, help="Cache matching the checkpoint normalizer")
    parser.add_argument("--steps", default="5,20")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=16384)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    main(parser.parse_args())
