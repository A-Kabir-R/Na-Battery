"""Registry contract: fixed order, capped width, no target/identity collisions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.fold_transform import (
    PreprocessorError, UnifiedFoldPreprocessor,
)
from src.preprocessing.unified_feature_registry import (
    FEATURE_SPECS,
    IDENTITY_COLUMNS,
    MAX_MODEL_FEATURES,
    TARGET_COLUMNS,
    continuous_feature_columns,
    flag_feature_columns,
    initial_state_column,
    model_feature_columns,
    registry_hash,
    stress_coordinate_column,
)


def _synthetic_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = model_feature_columns()
    frame = pd.DataFrame(
        rng.normal(size=(n, len(columns))), columns=columns,
    )
    for flag in flag_feature_columns():
        frame[flag] = 1.0
    frame["cell"] = [f"c{i % 5}" for i in range(n)]
    frame["condition"] = "cond"
    frame["next_rpt_Q_Ah"] = rng.normal(1.0, 0.01, size=n)
    return frame


def test_feature_count_within_hard_cap():
    assert len(model_feature_columns()) <= MAX_MODEL_FEATURES


def test_registry_has_no_duplicate_names():
    names = [spec.name for spec in FEATURE_SPECS]
    assert len(names) == len(set(names))


def test_registry_does_not_collide_with_targets_or_identity():
    names = set(model_feature_columns())
    assert not names & set(TARGET_COLUMNS)
    assert not names & set(IDENTITY_COLUMNS)


def test_no_target_derived_feature_is_registered():
    """No registered feature may be a function of the next RPT."""
    for spec in FEATURE_SPECS:
        assert not spec.name.startswith("next_")
        assert "next_rpt" not in spec.name
        for source in spec.sources:
            assert not source.startswith("next_")


def test_exactly_one_stress_and_one_initial_state():
    assert stress_coordinate_column() == "EFC_cum"
    assert initial_state_column() == "previous_rpt_Q_Ah"


def test_pinn_excludes_stress_and_initial_state_from_auxiliary_tensor():
    aux = model_feature_columns(
        exclude_roles=("stress_coordinate", "initial_state"),
    )
    assert stress_coordinate_column() not in aux
    assert initial_state_column() not in aux
    assert len(aux) == len(model_feature_columns()) - 2


def test_classical_and_pinn_share_one_registry():
    classical = model_feature_columns()
    aux = model_feature_columns(
        exclude_roles=("stress_coordinate", "initial_state"),
    )
    # The PINN's auxiliary block is a strict subsequence of the classical list,
    # in the same order — not an independently chosen subset.
    assert aux == [c for c in classical if c in set(aux)]


def test_registry_hash_is_stable():
    assert registry_hash() == registry_hash()
    assert len(registry_hash()) == 64


def test_fixed_column_order_is_preserved_through_transform():
    frame = _synthetic_frame()
    pre = UnifiedFoldPreprocessor()
    pre.fit(frame)
    assert pre.feature_columns_ == model_feature_columns()


def test_transform_is_finite_with_nan_and_inf_inputs():
    frame = _synthetic_frame()
    continuous = continuous_feature_columns()
    frame.loc[0, continuous[0]] = np.nan
    frame.loc[1, continuous[1]] = np.inf
    frame.loc[2, continuous[2]] = -np.inf
    matrix = UnifiedFoldPreprocessor().fit_transform(frame)
    assert np.isfinite(matrix).all()


def test_flags_are_not_scaled():
    frame = _synthetic_frame()
    pre = UnifiedFoldPreprocessor()
    matrix = pre.fit_transform(frame)
    columns = pre.feature_columns_
    for flag in flag_feature_columns():
        assert np.allclose(matrix[:, columns.index(flag)], 1.0)


def test_statistics_are_fitted_on_training_rows_only():
    """Adding rows to the validation split must not change the transform."""
    train = _synthetic_frame(n=30, seed=1)
    validation = _synthetic_frame(n=10, seed=2)
    pre = UnifiedFoldPreprocessor().fit(train)
    first = pre.transform(validation)

    extreme = validation.copy()
    extreme.loc[:, continuous_feature_columns()[0]] += 1000.0
    # Re-transforming with the SAME fitted state must reuse training medians and
    # scales, so the first column shifts but the fitted statistics do not move.
    second = pre.transform(extreme)
    assert pre.medians_ == UnifiedFoldPreprocessor().fit(train).medians_
    assert not np.allclose(first, second)


def test_all_nan_training_column_keeps_its_slot():
    """Fold-invariant width: a column missing on one fold is not dropped."""
    frame = _synthetic_frame()
    column = continuous_feature_columns()[3]
    frame[column] = np.nan
    pre = UnifiedFoldPreprocessor()
    matrix = pre.fit_transform(frame)
    assert matrix.shape[1] == len(model_feature_columns())
    assert np.isfinite(matrix).all()


def test_missing_column_raises():
    frame = _synthetic_frame()
    pre = UnifiedFoldPreprocessor().fit(frame)
    with pytest.raises(PreprocessorError):
        pre.transform(frame.drop(columns=[model_feature_columns()[0]]))


def test_state_dict_round_trip_is_exact():
    frame = _synthetic_frame()
    pre = UnifiedFoldPreprocessor().fit(frame)
    restored = UnifiedFoldPreprocessor.from_state_dict(pre.state_dict())
    assert np.allclose(pre.transform(frame), restored.transform(frame))
    assert restored.feature_hash() == pre.feature_hash()
