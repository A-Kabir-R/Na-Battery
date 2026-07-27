"""Combined classical/PINN result schema tests."""
from __future__ import annotations

import pandas as pd

from src.pinn.evaluation import target_metric_rows


def _dummy_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "architecture": ["NaPINN-Q"] * 6,
        "preprocessing": ["p2"] * 6,
        "fold": [0] * 6,
        "seed": [42] * 6,
        "evaluation_role": ["outer_validation"] * 6,
        "cell_id": ["A", "A", "B", "B", "C", "C"],
        "condition_id": ["c0"] * 6,
        "true_next_Q_Ah": [1.0, 0.98, 0.95, 0.94, 0.90, 0.88],
        "predicted_next_Q_Ah": [1.02, 0.97, 0.94, 0.94, 0.89, 0.87],
        "true_next_SOH_pct": [100.0, 98.0, 95.0, 94.0, 90.0, 88.0],
        "predicted_next_SOH_pct": [102.0, 97.0, 94.0, 94.0, 89.0, 87.0],
        "true_delta_Q_Ah": [0.0, -0.02, -0.03, -0.01, -0.04, -0.02],
        "predicted_delta_Q_Ah": [0.02, -0.05, -0.02, -0.02, -0.05, -0.03],
        "true_delta_SOH_pct": [0.0, -2.0, -3.0, -1.0, -4.0, -2.0],
        "predicted_delta_SOH_pct": [2.0, -5.0, -2.0, -2.0, -5.0, -3.0],
        "predicted_degradation_rate": [0.01] * 6,
        "du_ds": [-0.01] * 6,
        "pde_residual": [0.0] * 6,
        "initial_condition_error": [0.0] * 6,
        "integral_consistency_error": [0.0] * 6,
        "monotonicity_violation": [0.0] * 6,
        "lower_bound_violation": [0.0] * 6,
        "upper_bound_violation": [0.0] * 6,
        "horizon_days": [1.0] * 6,
    })


def test_target_metric_rows_contains_all_targets_and_aggregations() -> None:
    rows = target_metric_rows(_dummy_predictions())
    assert not rows.empty
    targets = set(rows["target"].unique())
    assert {"next_rpt_Q_Ah", "next_rpt_SOH_pct",
             "delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct"} <= targets
    aggregations = set(rows["aggregation"].unique())
    assert {"pooled", "cell_macro", "condition_macro"} <= aggregations


def test_metric_rows_have_numeric_MAE() -> None:
    rows = target_metric_rows(_dummy_predictions())
    for value in rows["MAE"]:
        assert isinstance(value, float)
        assert value >= 0 or value != value  # NaN allowed for degenerate delta rows
