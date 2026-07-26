import pandas as pd
import pytest

from src.targets.next_rpt import DuplicateRPTError, add_target


def _cycles():
    return pd.DataFrame({
        "condition": ["DOD20_2C2C_25degC"] * 5,
        "cell": ["CELL1"] * 5,
        "path": ["cyc1"] * 3 + ["cyc2"] * 2,
        "file_start_time": pd.to_datetime([
            "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-12", "2024-01-12",
        ]),
        "visit": [1, 1, 1, 3, 3],
        "cycle_in_file": [1, 2, 3, 1, 2],
        "global_cycle": [1, 2, 3, 4, 5],
        "cycle_end_time_s": [100.0, 200.0, 300.0, 100.0, 200.0],
        "cycle_complete": [True] * 5,
    })


def _rpt_cache():
    return pd.DataFrame({
        "condition": ["DOD20_2C2C_25degC"] * 3,
        "cell": ["CELL1"] * 3,
        "cu_visit": [0, 2, 4],
        "cu_path": ["rpt0", "rpt2", "rpt4"],
        "cu_test_id": [0, 20, 40],
        "cu_start_time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-13"]),
        "cu_Q_Ah": [1.25, 1.2, 1.1],
        "rpt_qc_status": ["valid"] * 3,
    })


def test_target_is_attached_once_per_cycling_file():
    result = add_target(_cycles(), _rpt_cache())
    anchors = result[result["is_prediction_anchor"]]
    assert anchors["global_cycle"].tolist() == [3, 5]
    assert anchors["next_rpt_Q_Ah"].tolist() == pytest.approx([1.2, 1.1])
    assert anchors["next_rpt_SOH_pct"].tolist() == pytest.approx([96.0, 88.0])
    assert anchors["previous_rpt_Q_Ah"].tolist() == pytest.approx([1.25, 1.2])
    assert anchors["delta_next_rpt_Q_Ah"].tolist() == pytest.approx([-0.05, -0.1])
    assert anchors["delta_next_rpt_SOH_pct"].tolist() == pytest.approx([-4.0, -8.0])
    assert set(anchors["next_rpt_target_status"]) == {"available"}
    assert result["next_rpt_Q_Ah"].notna().sum() == 2
    assert set(anchors["next_rpt_match_method"]) == {"cycle_end_time"}
    assert anchors.iloc[0]["next_rpt_horizon_days"] == pytest.approx(
        1.0 - 300.0 / 86400.0
    )


def test_duplicate_valid_rpt_is_blocking():
    duplicate = pd.concat([_rpt_cache(), _rpt_cache().iloc[[0]]], ignore_index=True)
    with pytest.raises(DuplicateRPTError):
        add_target(_cycles(), duplicate)


def test_final_cycle_remains_an_anchor_when_no_future_rpt_exists():
    cache = _rpt_cache().iloc[[0]].copy()
    result = add_target(_cycles(), cache)
    anchors = result[result["is_prediction_anchor"]]
    assert anchors["global_cycle"].tolist() == [3, 5]
    assert anchors["next_rpt_Q_Ah"].isna().all()
    assert set(anchors["target_unavailable_reason"]) == {"no_future_valid_rpt"}


def test_distinct_same_visit_rpts_are_matched_by_chronology():
    cycles = pd.DataFrame({
        "condition": ["DOD80_2C2C_40degC", "DOD80_2C2C_40degC"],
        "cell": ["CELL1", "CELL1"],
        "path": ["cyc15", "cyc17"],
        "file_start_time": pd.to_datetime(["2024-01-01", "2024-01-04"]),
        "visit": [15, 17],
        "cycle_in_file": [1, 1],
        "global_cycle": [1, 2],
        "cycle_end_time_s": [100.0, 100.0],
        "cycle_complete": [True, True],
    })
    cache = pd.DataFrame({
        "condition": ["DOD80_2C2C_40degC", "DOD80_2C2C_40degC"],
        "cell": ["CELL1", "CELL1"],
        "cu_visit": [16, 16],
        "cu_path": ["rpt-jan-03", "rpt-jan-06"],
        "cu_test_id": [5311, 5353],
        "cu_start_time": pd.to_datetime(["2024-01-03", "2024-01-06"]),
        "cu_Q_Ah": [1.16, 1.15],
        "rpt_qc_status": ["valid", "valid"],
    })
    result = add_target(cycles, cache)
    assert result["next_rpt_Q_Ah"].tolist() == pytest.approx([1.16, 1.15])
    assert result["next_rpt_path"].tolist() == ["rpt-jan-03", "rpt-jan-06"]


def test_visit_fallback_is_used_when_rpt_timestamps_are_missing():
    cache = _rpt_cache()
    cache["cu_start_time"] = pd.NaT
    result = add_target(_cycles(), cache)
    anchors = result[result["is_prediction_anchor"]]
    assert anchors["next_rpt_Q_Ah"].tolist() == pytest.approx([1.2, 1.1])
    assert set(anchors["next_rpt_match_method"]) == {"visit_fallback_missing_timestamp"}


def test_mixed_timestamps_use_latest_causal_visit_as_previous_rpt():
    cycles = _cycles().iloc[[3]].copy()
    cycles["path"] = "cyc3"
    cycles["visit"] = 3
    cycles["file_start_time"] = pd.Timestamp("2024-01-05")
    cache = pd.DataFrame({
        "condition": ["DOD20_2C2C_25degC"] * 3,
        "cell": ["CELL1"] * 3,
        "cu_visit": [0, 2, 4],
        "cu_path": ["rpt0", "rpt2", "rpt4"],
        "cu_test_id": [0, 20, 40],
        "cu_start_time": pd.to_datetime(["2024-01-01", None, "2024-01-06"]),
        "cu_Q_Ah": [1.25, 1.15, 1.05],
        "rpt_qc_status": ["valid"] * 3,
    })
    anchor = add_target(cycles, cache).iloc[0]
    assert anchor["previous_rpt_visit"] == 2
    assert anchor["previous_rpt_Q_Ah"] == pytest.approx(1.15)
    assert anchor["previous_rpt_match_method"] == "visit_fallback_missing_timestamp"
    assert anchor["delta_next_rpt_Q_Ah"] == pytest.approx(-0.10)


def test_chronologically_late_initial_rpt_does_not_normalize_soh():
    cycles = _cycles().iloc[[0]].copy()
    cycles["path"] = "cyc1"
    cycles["visit"] = 1
    cycles["file_start_time"] = pd.Timestamp("2024-01-05")
    cache = pd.DataFrame({
        "condition": ["DOD20_2C2C_25degC", "DOD20_2C2C_25degC"],
        "cell": ["CELL1", "CELL1"],
        "cu_visit": [0, 2],
        "cu_path": ["late-rpt0", "rpt2"],
        "cu_test_id": [0, 20],
        "cu_start_time": pd.to_datetime(["2024-01-10", "2024-01-06"]),
        "cu_Q_Ah": [1.25, 1.1],
        "rpt_qc_status": ["valid", "valid"],
    })
    anchor = add_target(cycles, cache).iloc[0]
    assert anchor["next_rpt_Q_Ah"] == pytest.approx(1.1)
    assert pd.isna(anchor["initial_rpt_Q_Ah"])
    assert pd.isna(anchor["next_rpt_SOH_pct"])
    assert anchor["next_rpt_soh_status"] == "missing_initial_rpt"
