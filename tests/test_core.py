import torch

from src.feature_cache import FeatureNormalizer
from src.operators import (
    deterministic_anchor_bank,
    expand_shared_anchor,
    probe_flow_maps,
    topk_pool,
)


class ConstantVelocity(torch.nn.Module):
    class_conditioned = False

    def forward(self, x, t, y=None):
        return torch.ones_like(x)


def test_angular_and_curvature_are_zero_for_straight_unit_flow():
    anchor = torch.zeros(2, 4, 3, 3)
    feature = torch.ones_like(anchor)
    maps = probe_flow_maps(
        ConstantVelocity(), feature, anchor, None,
        need_angular=True, need_curvature=True,
    )
    torch.testing.assert_close(maps["angular"], torch.zeros_like(maps["angular"]), atol=1e-6, rtol=0)
    torch.testing.assert_close(maps["curvature"], torch.zeros_like(maps["curvature"]), atol=1e-6, rtol=0)


def test_normalizer_round_trip_and_class_selection():
    mean = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
    std = torch.tensor([2.0, 4.0]).view(2, 1, 1, 1)
    normalizer = FeatureNormalizer(mean, std)
    x = torch.tensor([2.0, 9.0]).view(2, 1, 1, 1)
    labels = torch.tensor([0, 1])
    encoded = normalizer.encode(x, labels)
    torch.testing.assert_close(encoded.flatten(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(normalizer.decode(encoded, labels), x)


def test_shared_anchor_bank_is_stable_and_topk_uses_largest_values():
    first = deterministic_anchor_bank(
        (2, 2, 2), seed=7, num_anchors=2, device=torch.device("cpu")
    )
    second = deterministic_anchor_bank(
        (2, 2, 2), seed=7, num_anchors=2, device=torch.device("cpu")
    )
    torch.testing.assert_close(first, second)
    expanded = expand_shared_anchor(first[0], batch_size=3)
    torch.testing.assert_close(expanded[0], expanded[2])
    score = topk_pool(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), 0.5)
    torch.testing.assert_close(score, torch.tensor([3.5]))
