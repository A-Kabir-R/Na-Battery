import pandas as pd
import pytest

from src.evaluation.degradation import (
    build_rpt_longitudinal,
    cell_degradation_summary,
)


def test_rpt_soh_and_projection_are_explicitly_derived():
    rpt = pd.DataFrame({
        "condition": ["condition"] * 4,
        "cell": ["cell"] * 4,
        "visit": [0, 2, 4, 6],
        "test_id": [0, 2, 4, 6],
        "file_start_time": pd.to_datetime([
            "2024-01-01", "2024-01-11", "2024-01-21", "2024-01-31",
        ]),
        "reference_capacity_Ah": [1.2, 1.14, 1.08, 1.02],
        "rpt_qc_status": ["valid"] * 4,
    })
    cycles = pd.DataFrame({
        "condition": ["condition"] * 3,
        "cell": ["cell"] * 3,
        "cycle_complete": [True] * 3,
        "file_start_time": pd.to_datetime(["2024-01-02", "2024-01-12", "2024-01-22"]),
        "cycle_end_time_s": [100.0] * 3,
        "global_cycle": [100, 200, 300],
        "cumulative_Ah_throughput": [120.0, 240.0, 360.0],
    })
    longitudinal = build_rpt_longitudinal(rpt, cycles)
    assert longitudinal["rpt_SOH_pct"].tolist() == pytest.approx([100.0, 95.0, 90.0, 85.0])
    summary = cell_degradation_summary(longitudinal)
    assert summary.iloc[0]["observed_80pct_event"] == False  # noqa: E712
    assert summary.iloc[0]["projection_status"] == (
        "exploratory_linear_extrapolation_quality_eligible"
    )
    assert summary.iloc[0]["projected_day_at_80pct"] > 30
