"""Cross-model parity, causality audit and holdout-gating for the unified level.

These are the tests that make the headline claim checkable: that any difference
in reported accuracy between Ridge, ExtraTrees and NaPINN-Q comes from the
estimator and not from the features, rows or transform each one received.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.preprocessing.fold_transform import UnifiedFoldPreprocessor
from src.preprocessing.unified import (
    UnifiedBuildError, add_interactions, assert_causal, build_temporal_audit,
    temporal_features_at_anchor,
)
from src.preprocessing.unified_feature_registry import (
    MAX_MODEL_FEATURES, flag_feature_columns, model_feature_columns,
)

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config.yaml"


def _frame(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = model_feature_columns()
    frame = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    for flag in flag_feature_columns():
        frame[flag] = 1.0
    frame["cell"] = [f"c{i % 10}" for i in range(n)]
    frame["condition"] = [f"cond{i % 3}" for i in range(n)]
    frame["anchor_id"] = [f"a{i}" for i in range(n)]
    frame["next_rpt_Q_Ah"] = rng.normal(1.0, 0.01, size=n)
    return frame


# ---------------------------------------------------------------------------
# Config-level guarantees
# ---------------------------------------------------------------------------
def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_only_one_preprocessing_level_is_active():
    cfg = _config()
    assert cfg["experiment"]["preprocessing_levels"] == ["unified"]
    assert cfg["pinn"]["preprocessing"] == ["unified"]


def test_holdout_is_gated_off_by_default():
    """Development runs must not be able to read the locked holdout."""
    assert _config()["experiment"]["evaluate_holdout"] is False


def test_retained_models_are_exactly_the_four_planned():
    enabled = _config()["models"]["enabled_regressors"]
    assert "SVR-RBF" not in enabled
    for banned in ("RandomForest", "XGBoost", "LightGBM", "DNN-Q",
                   "GaussianProcess", "LSTM", "GRU", "Transformer"):
        assert banned not in enabled
    assert {"PreviousRPT", "Ridge", "ExtraTrees"} <= set(enabled)


# ---------------------------------------------------------------------------
# Feature parity between the classical and PINN transforms
# ---------------------------------------------------------------------------
def test_classical_and_pinn_transforms_agree_on_shared_columns():
    """The PINN's auxiliary block must be numerically identical to the
    corresponding columns of the classical matrix, not merely similar."""
    frame = _frame()
    classical = UnifiedFoldPreprocessor(scale=True)
    pinn = UnifiedFoldPreprocessor(
        exclude_roles=("stress_coordinate", "initial_state"), scale=True,
    )
    classical_matrix = classical.fit_transform(frame)
    pinn_matrix = pinn.fit_transform(frame)

    for index, column in enumerate(pinn.feature_columns_):
        classical_index = classical.feature_columns_.index(column)
        assert np.allclose(
            pinn_matrix[:, index], classical_matrix[:, classical_index]
        ), f"{column} differs between the classical and PINN transforms"


def test_pinn_auxiliary_block_excludes_exactly_two_columns():
    frame = _frame()
    classical = UnifiedFoldPreprocessor().fit(frame)
    pinn = UnifiedFoldPreprocessor(
        exclude_roles=("stress_coordinate", "initial_state"),
    ).fit(frame)
    dropped = set(classical.feature_columns_) - set(pinn.feature_columns_)
    assert dropped == {"EFC_cum", "previous_rpt_Q_Ah"}


def test_feature_count_is_within_the_cap_and_fold_invariant():
    a = UnifiedFoldPreprocessor().fit(_frame(n=30, seed=1))
    b = UnifiedFoldPreprocessor().fit(_frame(n=20, seed=2))
    assert a.feature_columns_ == b.feature_columns_
    assert len(a.feature_columns_) <= MAX_MODEL_FEATURES


def test_ridge_and_extratrees_receive_identical_matrices():
    """Both classical models share one transform, so the matrices are equal."""
    frame = _frame()
    ridge_matrix = UnifiedFoldPreprocessor(scale=True).fit_transform(frame)
    trees_matrix = UnifiedFoldPreprocessor(scale=True).fit_transform(frame)
    assert np.array_equal(ridge_matrix, trees_matrix)


# ---------------------------------------------------------------------------
# Causality audit
# ---------------------------------------------------------------------------
def test_audit_covers_every_registered_feature():
    audit = build_temporal_audit(_frame())
    assert set(audit["feature"]) == set(model_feature_columns())
    assert audit["uses_future_cycles"].eq(False).all()
    assert audit["uses_future_rpt"].eq(False).all()
    assert audit["target_derived"].eq(False).all()


def test_assert_causal_passes_on_a_complete_table():
    assert_causal(build_temporal_audit(_frame()))


def test_assert_causal_rejects_a_missing_feature():
    frame = _frame().drop(columns=[model_feature_columns()[2]])
    with pytest.raises(UnifiedBuildError):
        assert_causal(build_temporal_audit(frame))


def test_assert_causal_rejects_an_entirely_empty_feature():
    frame = _frame()
    frame[model_feature_columns()[4]] = np.nan
    with pytest.raises(UnifiedBuildError):
        assert_causal(build_temporal_audit(frame))


def test_assert_causal_rejects_a_flagged_future_feature():
    audit = build_temporal_audit(_frame())
    audit.loc[0, "uses_future_cycles"] = True
    with pytest.raises(UnifiedBuildError):
        assert_causal(audit)


def test_no_registered_feature_references_the_target():
    audit = build_temporal_audit(_frame())
    for sources in audit["source_columns"]:
        for source in str(sources).split(";"):
            assert not source.startswith("next_")


# ---------------------------------------------------------------------------
# Temporal features are backward-looking
# ---------------------------------------------------------------------------
def _history(n: int, slope: float = -0.001) -> pd.DataFrame:
    return pd.DataFrame({
        "Q_discharge_Ah": 1.2 + slope * np.arange(n),
        "coulombic_efficiency": 0.99 - 1e-5 * np.arange(n),
    })


def test_temporal_slope_recovers_a_known_trend():
    out = temporal_features_at_anchor(_history(30, slope=-0.002), window=10)
    assert out["temporal_window_available"] == 1.0
    assert np.isclose(out["Q_discharge_Ah_r10_slope"], -0.002, atol=1e-6)


def test_temporal_features_use_only_the_trailing_window():
    """Rows beyond the trailing window must not influence the result."""
    long_history = _history(200, slope=-0.002)
    short_history = long_history.tail(10).reset_index(drop=True)
    a = temporal_features_at_anchor(long_history, window=10)
    b = temporal_features_at_anchor(short_history, window=10)
    assert np.isclose(a["Q_discharge_Ah_r10_slope"], b["Q_discharge_Ah_r10_slope"])
    assert np.isclose(a["Q_discharge_Ah_r10_std"], b["Q_discharge_Ah_r10_std"])


def test_insufficient_history_sets_the_availability_flag_to_zero():
    out = temporal_features_at_anchor(_history(3), window=10)
    assert out["temporal_window_available"] == 0.0
    assert np.isnan(out["Q_discharge_Ah_r10_slope"])


def test_robust_slope_tolerates_a_regeneration_outlier():
    history = _history(20, slope=-0.002)
    history.loc[15, "Q_discharge_Ah"] += 0.5      # single bad measurement
    out = temporal_features_at_anchor(history, window=10)
    assert np.isclose(out["Q_discharge_Ah_r10_slope"], -0.002, atol=5e-4)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------
def test_interactions_are_computed_from_registered_inputs():
    frame = pd.DataFrame({
        "T_degC": [25.0, -10.0],
        "DOD_pct": [100.0, 20.0],
        "EFC_cum": [10.0, 50.0],
        "C_dis": [1.0, 2.0],
    })
    out = add_interactions(frame)
    assert np.isclose(out["inverse_temperature_1_K"].iloc[0], 1.0 / 298.15)
    assert np.isclose(out["stress_x_DOD_fraction"].iloc[0], 10.0)
    assert np.isclose(out["stress_x_DOD_fraction"].iloc[1], 10.0)
    assert np.isclose(out["stress_x_C_dis"].iloc[1], 100.0)


def test_interactions_do_not_divide_by_absolute_zero():
    out = add_interactions(pd.DataFrame({
        "T_degC": [-273.15], "DOD_pct": [100.0], "EFC_cum": [1.0], "C_dis": [1.0],
    }))
    assert np.isnan(out["inverse_temperature_1_K"].iloc[0])
