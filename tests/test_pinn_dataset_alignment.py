"""Synthetic transition-alignment tests for the Stage-1 dataset rewrite.

These tests enforce the physical time alignment required by diagnosis issues
#1-#5:

* ``u_current`` and ``stress_current`` must refer to the same physical time
  (the previous RPT).
* ``stress_delta`` must equal the exposure of the *current* CYC block.
* ``stress_delta_normalized`` must equal ``stress_delta / stress_std``.
* No ``stress_delta`` may be negative.
* No backward-fill of the stress coordinate may occur.
* No future-anchor exposure may enter dataset outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pinn.dataset import (
    FoldScaler, ForecastHorizonError, StressCoordinateError, build_anchor_dataset,
)


STRESS_CFG = {
    "preferred_coordinate": "EFC_cum",
    "fallback_coordinate": "cumulative_Ah_throughput",
    "allow_elapsed_time_fallback": True,
    "time_fallback_column": "cycle_end_time_s",
    "require_known_forecast_horizon": True,
    "epsilon": 1.0e-6,
}


def _synthetic_frame(n_cells: int = 3, anchors_per_cell: int = 4) -> pd.DataFrame:
    """Build a preprocessing frame with known EFC + Q trajectories."""
    rows = []
    Q0 = 1.20
    for i in range(n_cells):
        cell = f"cell_{i:02d}"
        # Fabricate strictly monotone EFC values and known previous / next RPT
        # capacities so we can assert alignment exactly.
        efc = np.array([50.0, 120.0, 220.0, 350.0])[:anchors_per_cell]
        # Choose previous RPT capacity per anchor: initial for the first,
        # then decaying linearly.
        previous_q = np.array([Q0, 1.18, 1.16, 1.14])[:anchors_per_cell]
        next_q = np.array([1.18, 1.16, 1.14, 1.12])[:anchors_per_cell]
        for a in range(anchors_per_cell):
            rows.append({
                "cell": cell,
                "condition": f"C{i % 2}",
                "visit": a + 1,
                "global_cycle": int(efc[a] * 10),  # unique per anchor per cell
                "is_prediction_anchor": True,
                "EFC_cum": float(efc[a]),
                "initial_rpt_Q_Ah": Q0,
                "previous_rpt_Q_Ah": float(previous_q[a]),
                "next_rpt_Q_Ah": float(next_q[a]),
                "next_rpt_SOH_pct": 100.0 * float(next_q[a]) / Q0,
                "T_mean": 25.0,
                "DOD_pct": 100.0,
                "C_ch": 1.0,
                "C_dis": 1.0,
            })
    return pd.DataFrame(rows)


def test_stress_current_equals_previous_anchor_end_within_cell() -> None:
    frame = _synthetic_frame()
    ds = build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)
    for cell, sub in ds.frame.groupby("cell", sort=False):
        sub = sub.sort_values("visit").reset_index(drop=True)
        # First anchor's stress_current must be 0 (initial RPT at cell start).
        assert sub.loc[0, "stress_current"] == pytest.approx(0.0)
        # Subsequent stress_current == previous anchor's stress_next.
        for k in range(1, len(sub)):
            assert sub.loc[k, "stress_current"] == pytest.approx(sub.loc[k - 1, "stress_next"])


def test_stress_delta_equals_current_block_exposure() -> None:
    frame = _synthetic_frame()
    ds = build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)
    for cell, sub in ds.frame.groupby("cell", sort=False):
        sub = sub.sort_values("visit").reset_index(drop=True)
        raw_efc = sub["EFC_cum"].to_numpy(dtype=float)
        expected_delta = np.concatenate([[raw_efc[0]], np.diff(raw_efc)])
        np.testing.assert_allclose(
            sub["stress_delta"].to_numpy(dtype=float), expected_delta,
            atol=1e-9,
        )


def test_no_stress_delta_is_negative_or_zero() -> None:
    frame = _synthetic_frame()
    ds = build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)
    delta = ds.frame["stress_delta"].to_numpy(dtype=float)
    assert (delta > 0).all(), delta


def test_backward_fill_rejected() -> None:
    """The first anchor of a cell may not receive its stress from a later row."""
    frame = _synthetic_frame(n_cells=1)
    # Mask the first anchor's raw stress — with only ffill it should raise.
    frame.loc[frame.index[0], "EFC_cum"] = np.nan
    with pytest.raises(StressCoordinateError):
        build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)


def test_scaler_horizon_normalization_is_pure_scaling() -> None:
    frame = _synthetic_frame()
    ds = build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)
    scaler = FoldScaler().fit(
        ds.frame, ds.feature_columns,
        u_max_source="outer_training", u_max_constant=1.05, u_max_margin=0.05,
        tolerance_source="constant", tolerance_constant=0.02,
        tolerance_quantile=0.95,
    )
    delta = ds.frame["stress_delta"].to_numpy(dtype=float)
    normalized_delta = scaler.transform_horizon(delta)
    # A pure interval / std has no additive offset; check exact equivalence.
    np.testing.assert_allclose(
        normalized_delta, delta / scaler.stress_std, atol=1e-12,
    )
    # Also verify: stress_next_norm - stress_current_norm == delta_norm.
    s_curr_norm = scaler.transform_stress(
        ds.frame["stress_current"].to_numpy(dtype=float)
    )
    s_next_norm = scaler.transform_stress(
        ds.frame["stress_next"].to_numpy(dtype=float)
    )
    np.testing.assert_allclose(s_next_norm - s_curr_norm, normalized_delta, atol=1e-9)


def test_no_future_anchor_exposure_in_stress_current() -> None:
    """Regression guard for the pre-fix bug where Δs jumped an anchor forward."""
    frame = _synthetic_frame(n_cells=1, anchors_per_cell=4)
    ds = build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)
    sub = ds.frame.sort_values("visit").reset_index(drop=True)
    # If future info leaked, stress_current[k] would exceed EFC[k]; enforce the
    # opposite invariant.
    for k in range(len(sub)):
        assert sub.loc[k, "stress_current"] < sub.loc[k, "EFC_cum"] + 1e-9


def test_u_current_and_stress_current_share_previous_rpt_time() -> None:
    """First-anchor u_current = 1.0 (initial), stress_current = 0.0."""
    frame = _synthetic_frame(n_cells=1, anchors_per_cell=4)
    ds = build_anchor_dataset(frame, preprocessing="p2", stress_cfg=STRESS_CFG)
    sub = ds.frame.sort_values("visit").reset_index(drop=True)
    assert sub.loc[0, "u_current"] == pytest.approx(1.0)
    assert sub.loc[0, "stress_current"] == pytest.approx(0.0)


def test_all_nan_columns_are_dropped_by_scaler() -> None:
    """FoldScaler must drop entirely-NaN columns (Stage 3 fix #15)."""
    frame = _synthetic_frame()
    frame["always_nan_feature"] = np.nan
    ds = build_anchor_dataset(
        frame, preprocessing="p2", stress_cfg=STRESS_CFG,
        audit_cfg={"allow_unaudited": True},
    )
    assert "always_nan_feature" in ds.feature_columns
    scaler = FoldScaler().fit(
        ds.frame, ds.feature_columns,
        u_max_source="constant", u_max_constant=1.05, u_max_margin=0.0,
        tolerance_source="constant", tolerance_constant=0.02,
        tolerance_quantile=0.95,
    )
    assert "always_nan_feature" in scaler.dropped_all_nan_columns
    assert "always_nan_feature" not in scaler.feature_columns_used
    assert not np.isnan(scaler.feature_mean).any()
    assert not np.isnan(scaler.feature_std).any()
