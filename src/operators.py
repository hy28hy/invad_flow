from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def deterministic_anchor_bank(
    feature_shape: Sequence[int], *, seed: int, num_anchors: int,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a path-independent Gaussian anchor bank shared by every image."""
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (num_anchors, *tuple(feature_shape)), generator=generator,
        device=device, dtype=dtype,
    )


def expand_shared_anchor(anchor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast one anchor to a batch without allocating per-image noise."""
    if anchor.ndim != 3:
        raise ValueError(f"anchor must be [C,H,W], got {tuple(anchor.shape)}")
    return anchor.unsqueeze(0).expand(batch_size, -1, -1, -1)


@torch.inference_mode()
def probe_flow_maps(
    model,
    feature: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    need_angular: bool,
    need_curvature: bool,
    probe_t: float = 0.4,
    curvature_dt: float = 0.2,
    curvature_mode: str = "normalized_change",
    amp_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute native flow maps; angular=1 NFE, curvature/both=2 NFE."""
    if not need_angular and not need_curvature:
        raise ValueError("At least one operator must be requested")
    if not 0.0 <= probe_t < 1.0:
        raise ValueError("probe_t must be in [0, 1)")
    if curvature_dt <= 0 or probe_t + curvature_dt > 1.0:
        raise ValueError("curvature_dt must be positive and probe_t + dt <= 1")

    batch = feature.shape[0]
    ta = torch.full((batch,), probe_t, device=feature.device, dtype=torch.float32)
    xa = (1.0 - probe_t) * anchor + probe_t * feature
    with torch.autocast(
        device_type=feature.device.type,
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_dtype is not None and feature.device.type == "cuda",
    ):
        va = model(xa, ta, labels).float()
    result: dict[str, torch.Tensor] = {}

    if need_angular:
        displacement = (feature - anchor).float()
        cosine = F.cosine_similarity(va, displacement, dim=1, eps=eps)
        result["angular"] = (1.0 - cosine).clamp_(0.0, 2.0)

    if need_curvature:
        tb = torch.full(
            (batch,), probe_t + curvature_dt,
            device=feature.device, dtype=torch.float32,
        )
        xb = xa.float() + curvature_dt * va
        with torch.autocast(
            device_type=feature.device.type,
            dtype=amp_dtype or torch.bfloat16,
            enabled=amp_dtype is not None and feature.device.type == "cuda",
        ):
            vb = model(xb, tb, labels).float()
        delta = vb - va

        if curvature_mode == "perpendicular":
            tangent = va / va.norm(dim=1, keepdim=True).clamp_min(eps)
            parallel = (delta * tangent).sum(dim=1, keepdim=True) * tangent
            delta = delta - parallel
        elif curvature_mode != "normalized_change":
            raise ValueError(f"Unknown curvature_mode: {curvature_mode}")

        mean_speed = 0.5 * (va.norm(dim=1) + vb.norm(dim=1))
        result["curvature"] = delta.norm(dim=1) / (
            curvature_dt * mean_speed + eps
        )
    return result


def _gaussian_kernel1d(sigma: float, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    return kernel / kernel.sum()


def gaussian_blur(map_4d: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return map_4d
    kernel = _gaussian_kernel1d(float(sigma), map_4d.device, map_4d.dtype)
    radius = kernel.numel() // 2
    padded = F.pad(map_4d, (radius, radius, radius, radius), mode="reflect")
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    blurred = F.conv2d(padded, horizontal)
    return F.conv2d(blurred, vertical)


def postprocess_map(raw_map: torch.Tensor, image_size: tuple[int, int],
                    sigma: float) -> torch.Tensor:
    if raw_map.ndim != 3:
        raise ValueError(f"raw_map must be [B,H,W], got {tuple(raw_map.shape)}")
    upsampled = F.interpolate(
        raw_map[:, None].float(), size=image_size,
        mode="bilinear", align_corners=False,
    )
    return gaussian_blur(upsampled, sigma).squeeze(1)


def topk_pool(score_map: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Top-k fraction must be in (0, 1]")
    flat = score_map.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * fraction)))
    return flat.topk(k, dim=1).values.mean(dim=1)
