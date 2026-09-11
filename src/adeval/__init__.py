"""Metrics needed by InvAD Flow.

Keep package import CPU-safe: the flow evaluator only needs AU-PRO and should not
eagerly import the legacy CUDA accumulators.
"""

from .au_pro import calculate_au_pro

__all__ = ["calculate_au_pro"]
