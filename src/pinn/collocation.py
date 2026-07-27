"""Unlabeled collocation-point sampling for NaPINN-Q.

At every transition interval ``[s_k, s_{k+1}]`` the sampler draws
``points_per_transition`` unlabeled coordinates. Feature vectors are held
constant across the interval (left-anchor policy) — an honest choice for the
one-shot RPT anchor forecasting task where no future covariate is measured
between anchors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class CollocationBatch:
    stress: torch.Tensor
    features: torch.Tensor
    row_index: torch.Tensor
    cell_index: torch.Tensor
    weight: torch.Tensor


def sample_collocation_points(*, stress_current: torch.Tensor,
                              stress_delta: torch.Tensor,
                              features: torch.Tensor,
                              cell_index: torch.Tensor,
                              points_per_transition: int,
                              generator: torch.Generator | None = None,
                              sampling: str = "uniform") -> CollocationBatch:
    """Sample per-transition collocation coordinates.

    Returns a :class:`CollocationBatch` where every collocation coordinate lies
    strictly inside its transition interval ``(s_k, s_{k+1}]`` and features are
    repeated from the left anchor.
    """
    if points_per_transition < 1:
        raise ValueError("points_per_transition must be >= 1")
    n, feature_dim = features.shape
    j = int(points_per_transition)
    if sampling != "uniform":
        raise ValueError(f"unsupported collocation sampling: {sampling}")

    xi = torch.rand((n, j), device=features.device, dtype=stress_current.dtype,
                    generator=generator)
    # Enforce strictly inside the interval; the anchor point itself is enforced
    # separately by the initial-condition loss.
    xi = xi.clamp(min=1e-4, max=1.0)
    stress_grid = stress_current.unsqueeze(-1) + xi * stress_delta.unsqueeze(-1)
    stress_flat = stress_grid.reshape(-1)
    features_flat = features.unsqueeze(1).expand(-1, j, -1).reshape(-1, feature_dim)
    row_index = torch.arange(n, device=features.device).repeat_interleave(j)
    cell_flat = cell_index.repeat_interleave(j) if cell_index.numel() else cell_index

    per_row_weight = 1.0 / j
    weight = torch.full((n * j,), per_row_weight, device=features.device,
                        dtype=stress_current.dtype)
    return CollocationBatch(
        stress=stress_flat,
        features=features_flat,
        row_index=row_index,
        cell_index=cell_flat,
        weight=weight,
    )


def enforce_collocation_purity(candidate_columns: list[str]) -> None:
    """Assert no target column has been passed as a collocation feature."""
    forbidden = {
        "next_rpt_Q_Ah", "next_rpt_SOH_pct",
        "delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct",
        "u_true_next",
    }
    leak = sorted(set(candidate_columns) & forbidden)
    if leak:
        raise ValueError(f"collocation features contain target-derived columns: {leak}")


def summarize_collocation_batch(batch: CollocationBatch) -> dict[str, Any]:
    stress = batch.stress.detach().cpu().numpy()
    return {
        "n_collocation": int(stress.size),
        "stress_min": float(np.nanmin(stress)) if stress.size else float("nan"),
        "stress_max": float(np.nanmax(stress)) if stress.size else float("nan"),
        "stress_mean": float(np.nanmean(stress)) if stress.size else float("nan"),
    }
