import pandas as pd

from src.evaluation.metrics import aggregated_regression_metrics
from src.evaluation.report import prediction_metric_tables


def _predictions():
    return pd.DataFrame({
        "preprocessing": ["p1"] * 6,
        "target": ["next_rpt_Q_Ah"] * 6,
        "model": ["model"] * 6,
        "evaluation_role": ["cv_oof"] * 6,
        "condition": ["c1", "c1", "c1", "c1", "c2", "c2"],
        "cell": ["A", "A", "A", "A", "B", "B"],
        "y_true": [1.0, 1.1, 1.2, 1.3, 1.0, 1.1],
        "y_pred": [1.0, 1.1, 1.2, 1.3, 1.2, 1.3],
        "next_rpt_horizon_days": [0.5, 2, 6, 20, 0.5, 2],
        "prediction_valid": [True] * 6,
    })


def test_cell_macro_differs_from_row_weighted_metric():
    predictions = _predictions()
    pooled = aggregated_regression_metrics(predictions, "pooled")
    macro = aggregated_regression_metrics(predictions, "cell_macro")
    assert pooled["MAE"] != macro["MAE"]


def test_prediction_tables_include_bootstrap_and_horizons():
    summary, groups, horizon, plausibility = prediction_metric_tables(
        _predictions(), bootstrap_replicates=20, random_state=7
    )
    assert {"pooled", "cell_macro", "condition_macro"} <= set(summary["aggregation"])
    assert summary["ci_low"].notna().any()
    assert set(groups["group_type"]) == {"cell", "condition"}
    assert set(horizon["horizon_bin"]) == {"<1d", "1-5d", "5-15d", ">15d"}
    assert plausibility.iloc[0]["invalid_prediction_count"] == 0


def test_macro_r2_reports_contributing_group_count():
    predictions = pd.concat([
        _predictions(),
        pd.DataFrame({
            "preprocessing": ["p1"], "target": ["next_rpt_Q_Ah"],
            "model": ["model"], "evaluation_role": ["cv_oof"],
            "condition": ["c3"], "cell": ["single"],
            "y_true": [1.0], "y_pred": [1.0],
            "next_rpt_horizon_days": [float("nan")], "prediction_valid": [True],
        }),
    ], ignore_index=True)
    summary, _, _, _ = prediction_metric_tables(predictions, bootstrap_replicates=0)
    row = summary[
        (summary["aggregation"] == "cell_macro") & (summary["metric"] == "R2")
    ].iloc[0]
    assert row["n_groups_total"] == 3
    assert row["n_groups_metric_valid"] == 2


def test_empty_prediction_tables_keep_stable_schemas():
    summary, groups, horizon, plausibility = prediction_metric_tables(pd.DataFrame())
    assert "metric" in summary.columns
    assert "group_type" in groups.columns
    assert "horizon_bin" in horizon.columns
    assert "invalid_prediction_count" in plausibility.columns
