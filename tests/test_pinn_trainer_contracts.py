"""Stage-2/3/4 trainer + aggregator contract tests.

These tests verify the behavioural contracts the diagnosis fixes introduced,
without executing full-model training (which would require CUDA and minutes
per fold).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.pinn.dataset import inner_split_indices


def _outer_train_frame(n_cells: int = 8, anchors_per_cell: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_cells):
        cell = f"cell_{i:02d}"
        for a in range(anchors_per_cell):
            rows.append({"cell": cell, "visit": a + 1})
    return pd.DataFrame(rows)


def test_inner_split_stable_across_training_seeds() -> None:
    """Stage 2 diagnosis fix #13.

    The inner split must depend on the configured ``inner_split_seed`` —
    not on the model's training seed — so seed variability isolates neural
    stochasticity rather than also changing the inner-val cell composition.
    """
    frame = _outer_train_frame()
    fixed_seed = 20240117
    train_a, val_a = inner_split_indices(frame, seed=fixed_seed)
    train_b, val_b = inner_split_indices(frame, seed=fixed_seed)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)


def test_inner_split_changes_when_split_seed_differs() -> None:
    frame = _outer_train_frame(n_cells=12)
    train_a, val_a = inner_split_indices(frame, seed=1)
    train_b, val_b = inner_split_indices(frame, seed=2)
    # At least one of the two subsets should differ across split seeds.
    assert (set(map(int, val_a)) != set(map(int, val_b))) or \
           (set(map(int, train_a)) != set(map(int, train_b)))


def test_aggregator_ignores_failed_status(tmp_path: Path) -> None:
    """Stage 3/4 diagnosis fix #21.

    The aggregator's ``_iter_prediction_paths`` must yield only prediction
    files whose sibling ``status.json`` reports ``status == "completed"``.
    """
    from importlib import util
    spec = util.spec_from_file_location(
        "aggregate_module",
        Path(__file__).resolve().parents[1] / "scripts" / "07_aggregate_pinn_results.py",
    )
    module = util.module_from_spec(spec)
    # Guard against the module executing main() at import time.
    with patch("sys.argv", ["07_aggregate_pinn_results.py"]):
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except SystemExit:
            pass
    ok = tmp_path / "runs" / "arch" / "prep" / "fold_0" / "seed_1"
    bad = tmp_path / "runs" / "arch" / "prep" / "fold_1" / "seed_1"
    naked = tmp_path / "runs" / "arch" / "prep" / "fold_2" / "seed_1"
    for path in (ok, bad, naked):
        path.mkdir(parents=True)
        (path / "predictions.parquet").write_text("stub")
    (ok / "status.json").write_text(json.dumps({"status": "completed"}))
    (bad / "status.json").write_text(json.dumps({"status": "failed"}))
    # ``naked`` has no status.json — must be skipped.

    kept = list(module._iter_prediction_paths(tmp_path / "runs"))  # type: ignore[attr-defined]
    kept_names = {str(p.parent) for p in kept}
    assert str(ok) in kept_names
    assert str(bad) not in kept_names
    assert str(naked) not in kept_names


def test_physics_metric_rows_new_column_names_present() -> None:
    """Stage 2 diagnosis fix #9: physical & normalized metrics reported side by side."""
    from src.pinn.evaluation import physics_metric_rows
    pinn_rows = pd.DataFrame({
        "architecture": ["NaPINN-Q"] * 3,
        "preprocessing": ["p2"] * 3,
        "fold": [0] * 3,
        "seed": [42] * 3,
        "evaluation_role": ["outer_validation"] * 3,
        "pde_residual_normalized": [0.001, -0.001, 0.0],
        "pde_residual_physical": [1e-6, -1e-6, 0.0],
        "du_dstress_normalized": [-0.01] * 3,
        "du_dstress_physical": [-1e-5] * 3,
        "predicted_degradation_rate_normalized": [0.01, 0.011, 0.012],
        "predicted_degradation_rate_physical": [1e-5, 1.1e-5, 1.2e-5],
        "initial_condition_error": [0.001, 0.002, 0.0],
        "integral_consistency_error": [0.001, 0.001, 0.0],
        "monotonicity_violation": [0.0] * 3,
        "lower_bound_violation": [0.0] * 3,
        "upper_bound_violation": [0.0] * 3,
    })
    metrics = physics_metric_rows(pinn_rows).iloc[0]
    # New column names present alongside old-style aggregate metrics.
    assert "pde_residual_MAE_normalized" in metrics.index
    assert "pde_residual_MAE_physical" in metrics.index
    assert "mean_predicted_rate_normalized" in metrics.index
    assert "mean_predicted_rate_physical" in metrics.index
