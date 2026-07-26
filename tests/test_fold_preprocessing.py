import numpy as np
import pandas as pd

from src.features.selection import FoldPreprocessor


def test_test_values_do_not_change_training_imputation():
    train = pd.DataFrame({
        "cell": ["A", "B", "C"],
        "feature": [1.0, np.nan, 5.0],
        "noise": [3.0, 7.0, 4.0],
        "next_rpt_Q_Ah": [1.1, 1.0, 0.9],
    })
    test = pd.DataFrame({
        "cell": ["D"],
        "feature": [1e9],
        "noise": [-1e9],
        "next_rpt_Q_Ah": [999.0],
    })
    processor = FoldPreprocessor().fit(train, target="next_rpt_Q_Ah")
    assert processor.medians_["feature"] == 3.0
    transformed = processor.transform(test)
    assert transformed.shape == (1, len(processor.feature_names_))
    assert processor.medians_["feature"] == 3.0


def test_target_defining_capacity_is_excluded_from_soh_features():
    train = pd.DataFrame({
        "cell": ["A", "B", "C"],
        "Q_discharge_Ah": [1.2, 1.1, 1.0],
        "temperature_feature": [24.0, 25.0, 27.0],
        "SOH_pct": [100.0, 91.0, 83.0],
    })
    processor = FoldPreprocessor().fit(train, target="SOH_pct")
    assert "Q_discharge_Ah" not in processor.feature_names_
    assert "temperature_feature" in processor.feature_names_


def test_future_target_metadata_is_excluded_from_features():
    train = pd.DataFrame({
        "cell": ["A", "B", "C"],
        "past_temperature": [24.0, 25.0, 27.0],
        "next_rpt_visit": [2, 4, 6],
        "next_rpt_horizon_days": [1.0, 2.0, 3.0],
        "is_prediction_anchor": [True, True, True],
        "next_rpt_Q_Ah": [1.2, 1.1, 1.0],
    })
    processor = FoldPreprocessor().fit(train, target="next_rpt_Q_Ah")
    assert processor.feature_names_ == ["past_temperature"]


def test_new_targets_and_segment_index_are_excluded():
    train = pd.DataFrame({
        "cell": ["A", "B", "C"],
        "segment_index": [1, 2, 3],
        "past_temperature": [24.0, 25.0, 27.0],
        "next_rpt_Q_Ah": [1.2, 1.1, 1.0],
        "next_rpt_SOH_pct": [100.0, 92.0, 84.0],
        "delta_next_rpt_Q_Ah": [0.0, -0.01, -0.02],
        "delta_next_rpt_SOH_pct": [0.0, -1.0, -2.0],
    })
    processor = FoldPreprocessor().fit(train, target="next_rpt_SOH_pct")
    assert processor.feature_names_ == ["past_temperature"]
