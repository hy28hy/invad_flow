from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.dit import DiT


class FlowDiT(nn.Module):
    """DiT velocity field with one canonical continuous-time adapter."""

    def __init__(self, net: DiT, *, time_scale: float = 999.0,
                 class_conditioned: bool = False):
        super().__init__()
        self.net = net
        self.time_scale = float(time_scale)
        self.class_conditioned = bool(class_conditioned)
        # Flow inputs are always BCHW, so the token-input projection is unused.
        # Freeze known-unused parameters instead of paying DDP's find_unused cost.
        self.net.x_embedder_linear.requires_grad_(False)
        if not self.class_conditioned:
            self.net.y_embedder.requires_grad_(False)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.shape != (x.shape[0],):
            raise ValueError(f"t must have shape ({x.shape[0]},), got {tuple(t.shape)}")
        model_y = y if self.class_conditioned else None
        return self.net(x, t.float() * self.time_scale, y=model_y)


def build_flow_model(config: dict[str, Any], input_shape: tuple[int, int, int]) -> FlowDiT:
    cfg = config["model"]
    channels, height, width = input_shape
    if height != width:
        raise ValueError("The current DiT positional embedding requires a square feature grid")
    if int(cfg.get("patch_size", 1)) != 1:
        raise ValueError("InvAD feature maps must use patch_size=1 to preserve localization")

    net = DiT(
        input_size=height,
        patch_size=int(cfg.get("patch_size", 1)),
        in_channels=channels,
        cond_channels=int(cfg.get("z_channels", 768)),
        hidden_size=int(cfg["width"]),
        depth=int(cfg["depth"]),
        num_heads=int(cfg.get("num_heads", 8)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        class_dropout_prob=float(cfg.get("class_dropout_prob", 0.0)),
        num_classes=int(cfg["num_classes"]),
        learn_sigma=False,
    )
    return FlowDiT(
        net,
        time_scale=float(cfg.get("time_scale", 999.0)),
        class_conditioned=bool(cfg.get("class_conditioned", False)),
    )


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param, alpha=1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, value in model.named_buffers():
        ema_buffers[name].copy_(value)


def make_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    return ema


def warm_start_from_ddpm(model: FlowDiT, checkpoint_path: str | Path) -> dict[str, list[str]]:
    """Load reusable DDPM DiT weights while resetting velocity-specific output layers."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]

    reusable = {}
    skipped = []
    for key, value in payload.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("net.")
        if key.startswith("final_layer."):
            skipped.append(key)
            continue
        reusable[f"net.{key}"] = value
    incompatible = model.load_state_dict(reusable, strict=False)
    return {
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "reset": skipped,
    }
