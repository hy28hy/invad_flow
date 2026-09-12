from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.feature_cache import (
    CachedFeatureDataset,
    cache_contract,
    normalizer_from_cache,
    validate_cache_compatibility,
    validate_checkpoint_cache,
)
from src.flow_model import build_flow_model, make_ema, update_ema, warm_start_from_ddpm


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def distributed_context(timeout_minutes: int) -> tuple[bool, int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
        return True, rank, world_size, local_rank, torch.device("cuda", local_rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict,
                    steps_per_epoch: int, num_epochs: int):
    cfg = config["optimizer"]
    total = max(1, num_epochs * steps_per_epoch)
    warmup = min(total - 1, int(cfg.get("warmup_epochs", 0)) * steps_per_epoch)
    initial = float(cfg["init_lr"])
    peak = float(cfg.get("peak_lr", initial))
    final = float(cfg.get("final_lr", initial))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            lr = initial + (peak - initial) * step / warmup
        else:
            progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
            lr = final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))
        return lr / initial

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, ema, normalizer, optimizer, scheduler,
                       config: dict, epoch: int, global_step: int,
                       feature_shape: tuple[int, int, int], save_optimizer: bool,
                       cache_identity: dict[str, object]) -> dict:
    payload = {
        "format_version": 2,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "normalizer": normalizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "feature_shape": feature_shape,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "cache_contract": cache_identity,
    }
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    return payload


def field_diagnostics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred = prediction.detach().float().flatten(1)
    truth = target.detach().float().flatten(1)
    return {
        "norm_u_t": truth.norm(dim=1).mean(),
        "norm_v_theta": pred.norm(dim=1).mean(),
        "cos_sim": F.cosine_similarity(pred, truth, dim=1, eps=1e-8).mean(),
        "rms_u_t": truth.square().mean().sqrt(),
        "rms_v_theta": pred.square().mean().sqrt(),
    }


def last_layer_grad_diagnostics(model) -> dict[str, float | int | bool]:
    grad = model.net.final_layer.linear.weight.grad
    if grad is None:
        return {"present": False}
    grad = grad.detach().float()
    finite = torch.isfinite(grad)
    values = grad[finite]
    if values.numel() == 0:
        return {"present": True, "finite_fraction": 0.0}
    return {
        "present": True,
        "numel": grad.numel(),
        "finite_fraction": float(finite.float().mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "l2_norm": float(values.norm()),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    try:
        from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
    except ImportError as exc:
        raise RuntimeError("Install requirements-flow.txt in the invad_flow environment") from exc

    config = load_config(args.config)
    if args.log_interval is not None:
        config["logging"]["log_interval"] = args.log_interval
    schedule_epochs = int(config["optimizer"]["num_epochs"])
    stop_epoch = int(args.epochs) if args.epochs is not None else schedule_epochs
    if not 0 < stop_epoch <= schedule_epochs:
        raise ValueError(
            f"--epochs must be in [1, {schedule_epochs}]; got {stop_epoch}"
        )
    timeout_minutes = int(config.get("distributed", {}).get("timeout_minutes", 15))
    distributed, rank, world_size, local_rank, device = distributed_context(timeout_minutes)
    is_main = rank == 0

    seed = int(config["meta"]["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(config["logging"]["save_dir"])
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_log = diagnostics_dir / "train_metrics.jsonl"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        if args.save_diagnostics and diagnostics_log.exists() and not args.resume:
            os.replace(
                diagnostics_log,
                diagnostics_log.with_name(f"train_metrics.{int(time.time())}.jsonl"),
            )

    cache = CachedFeatureDataset(config["cache"]["path"], split="all")
    validate_cache_compatibility(cache, config)
    train_split = str(config["cache"].get("train_split", "all"))
    if train_split not in {"all", "train"}:
        raise ValueError("cache.train_split must be 'all' or 'train'")
    train_set = CachedFeatureDataset(config["cache"]["path"], split=train_split)
    if len(train_set) == 0:
        raise RuntimeError("The cached training split is empty")
    sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True,
        seed=int(config["meta"]["seed"]),
    ) if distributed else None
    loader = DataLoader(
        train_set,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        # Keep the tail batch: OT-CFM supports variable local batch sizes and
        # dropping it would silently omit normal samples every epoch.
        drop_last=bool(config["data"].get("drop_last", False)),
        persistent_workers=int(config["data"].get("num_workers", 4)) > 0,
    )

    base_model = build_flow_model(config, cache.feature_shape).to(device)
    normalizer = normalizer_from_cache(
        cache, floor=float(config["flow"].get("normalizer_floor", 1e-4))
    ).to(device)
    warm_start = config["model"].get("warm_start_ddpm")
    if warm_start:
        report = warm_start_from_ddpm(base_model, warm_start)
        if is_main:
            print("DDPM warm-start report:", json.dumps(report, indent=2))

    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint_cache(resume_payload, cache)
        base_model.load_state_dict(resume_payload["model"], strict=True)
        normalizer.load_state_dict(resume_payload["normalizer"], strict=True)

    # Rank 0 exclusively owns EMA and checkpoint serialization.
    ema = make_ema(base_model).to(device) if is_main else None
    if resume_payload is not None and is_main:
        ema.load_state_dict(resume_payload["ema"], strict=True)
    model = DDP(
        base_model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
    ) if distributed else base_model

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(opt_cfg["init_lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = build_scheduler(optimizer, config, len(loader), schedule_epochs)
    start_epoch = 0
    global_step = 0
    if resume_payload is not None:
        if "optimizer" not in resume_payload:
            raise ValueError("Resume checkpoint has no optimizer state; use it as warm_start instead")
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload["global_step"])

    sigma = float(config["flow"].get("sigma", 0.0))
    if sigma != 0.0:
        raise ValueError("flow.sigma must be 0 for straight OT-CFM")
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    amp_dtype = torch.bfloat16 if config["meta"].get("amp", "bf16") == "bf16" else torch.float16
    class_conditioned = base_model.class_conditioned
    ema_decay = float(config["model"].get("ema_decay", 0.999))
    grad_clip = float(opt_cfg.get("grad_clip", 1.0))
    log_interval = int(config["logging"].get("log_interval", 20))
    save_interval = int(config["logging"].get("save_interval", 25))
    save_optimizer = bool(config["logging"].get("save_optimizer", False))
    nonpositive_cos_steps = 0

    writer = None
    if is_main and args.save_diagnostics:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(diagnostics_dir / "tensorboard"))
        except ImportError:
            print("TensorBoard unavailable; JSONL diagnostics remain enabled")

    if is_main:
        startup = {
            "event": "startup",
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "rank0_owns_ema": ema is not None,
            "parameters_m": sum(p.numel() for p in base_model.parameters()) / 1e6,
            "trainable_parameters_m": sum(
                p.numel() for p in base_model.parameters() if p.requires_grad
            ) / 1e6,
            "train_samples": len(train_set),
            "train_split": train_split,
            "schedule_epochs": schedule_epochs,
            "stop_epoch": stop_epoch,
            "feature_shape": cache.feature_shape,
            "class_conditioned": class_conditioned,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        print(json.dumps(startup, indent=2))
        if args.save_diagnostics:
            append_jsonl(diagnostics_log, startup)

    model.train()
    for epoch in range(start_epoch, stop_epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for iteration, batch in enumerate(loader):
            x1 = batch["feature"].to(device, non_blocking=True).float()
            original_labels = batch["clslabel"].to(device, non_blocking=True).long()
            x1 = normalizer.encode(x1, original_labels)
            x0 = torch.randn_like(x1)
            if class_conditioned:
                t, xt, ut, _, matched_labels = matcher.guided_sample_location_and_conditional_flow(
                    x0, x1, y0=None, y1=original_labels
                )
            else:
                t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
                matched_labels = None

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                prediction = model(xt, t, matched_labels)
            loss = (prediction.float() - ut.float()).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss on rank={rank}, epoch={epoch}, iteration={iteration}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip > 0 else float("inf")
            )
            diag = field_diagnostics(prediction, ut)
            diag["loss"] = loss.detach()
            diag["global_grad_norm"] = grad_norm.detach().float()
            reduced = {
                key: float(reduce_mean(value, world_size).cpu())
                for key, value in diag.items()
            }

            if global_step < 100 and reduced["cos_sim"] <= 0:
                nonpositive_cos_steps += 1
            high_grad = global_step < 100 and reduced["global_grad_norm"] > 1e4
            cos_alert = global_step == 99 and nonpositive_cos_steps == 100
            if (high_grad or cos_alert) and is_main:
                reason = "grad_norm_gt_1e4" if high_grad else "cos_nonpositive_first_100_steps"
                alert = {
                    "event": "alert",
                    "reason": reason,
                    "global_step": global_step,
                    "metrics": reduced,
                    "last_layer_gradient": last_layer_grad_diagnostics(base_model),
                }
                print("TRAINING ALERT:", json.dumps(alert, indent=2))
                append_jsonl(diagnostics_dir / "alerts.jsonl", alert)

            optimizer.step()
            scheduler.step()
            if is_main:
                update_ema(ema, base_model, ema_decay)
            epoch_loss += reduced["loss"]
            if is_main and global_step % log_interval == 0:
                record = {
                    "event": "train_step", "epoch": epoch, "iteration": iteration,
                    "global_step": global_step, "lr": optimizer.param_groups[0]["lr"],
                    **reduced,
                }
                print(json.dumps(record))
                if args.save_diagnostics:
                    append_jsonl(diagnostics_log, record)
                    if writer is not None:
                        for key, value in reduced.items():
                            writer.add_scalar(f"train/{key}", value, global_step)
                        writer.add_scalar("train/lr", record["lr"], global_step)
            global_step += 1

        if is_main:
            epoch_record = {
                "event": "epoch_end", "epoch": epoch, "global_step": global_step,
                "mean_loss": epoch_loss / max(1, len(loader)),
            }
            print(json.dumps(epoch_record))
            if args.save_diagnostics:
                append_jsonl(diagnostics_log, epoch_record)
        if distributed:
            dist.barrier(device_ids=[local_rank])
        should_save = (epoch + 1) % save_interval == 0 or epoch + 1 == stop_epoch
        if is_main and should_save:
            payload = checkpoint_payload(
                base_model, ema, normalizer, optimizer, scheduler, config,
                epoch, global_step, cache.feature_shape, save_optimizer,
                cache_contract(cache),
            )
            atomic_save(payload, output_dir / "flow_latest.pth")
            if (epoch + 1) % save_interval == 0:
                atomic_save(payload, output_dir / f"flow_epoch_{epoch + 1:04d}.pth")
        if distributed:
            dist.barrier(device_ids=[local_rank])

    if writer is not None:
        writer.close()
    if distributed:
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def main(args: argparse.Namespace) -> None:
    try:
        run(args)
    except BaseException:
        rank = int(os.environ.get("RANK", 0))
        try:
            config = load_config(args.config)
            error_dir = Path(config["logging"]["save_dir"]) / "diagnostics"
            error_dir.mkdir(parents=True, exist_ok=True)
            with open(error_dir / f"ddp_error_rank{rank}.log", "a", encoding="utf-8") as handle:
                handle.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                handle.write(traceback.format_exc())
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP OT-CFM training on frozen InvAD features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Stop after this epoch; LR schedule still uses optimizer.num_epochs",
    )
    parser.add_argument("--log-interval", "--log_interval", dest="log_interval", type=int, default=None)
    parser.add_argument(
        "--save-diagnostics", "--save_diagnostics", dest="save_diagnostics", action="store_true"
    )
    main(parser.parse_args())
