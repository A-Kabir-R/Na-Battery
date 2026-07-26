import json

import pandas as pd
from sklearn.linear_model import Ridge

import src.pipeline.run_experiment as experiment


def _data():
    return pd.DataFrame({
        "condition": [f"condition-{index % 2}" for index in range(9)],
        "cell": [f"cell-{index}" for index in range(9)],
        "feature_a": [float(index) for index in range(9)],
        "feature_b": [float(index * index + 1) for index in range(9)],
        "next_rpt_Q_Ah": [1.2 - 0.01 * index for index in range(9)],
    })


def test_completed_fold_is_reused_only_when_fingerprint_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment,
        "_fold_dir",
        lambda preprocessing, target, model_name, fold: tmp_path / f"fold-{fold}",
    )
    monkeypatch.setattr(experiment, "_curve_dir", lambda: tmp_path / "curves")
    monkeypatch.setattr(
        experiment,
        "_holdout_dir",
        lambda preprocessing, target, model_name: tmp_path / "holdout",
    )
    first = experiment.run_one(
        _data(), "p1", "next_rpt_Q_Ah", "Ridge", Ridge(), "regression", n_splits=3
    )
    assert first
    status_path = tmp_path / "fold-0" / "fold_status.json"
    first_fingerprint = json.loads(status_path.read_text())["fingerprint"]

    changed = _data()
    changed.loc[0, "feature_a"] = 999.0
    second = experiment.run_one(
        changed, "p1", "next_rpt_Q_Ah", "Ridge", Ridge(), "regression", n_splits=3
    )
    assert second
    second_fingerprint = json.loads(status_path.read_text())["fingerprint"]
    assert second_fingerprint != first_fingerprint


def test_failed_rerun_removes_stale_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment,
        "_fold_dir",
        lambda preprocessing, target, model_name, fold: tmp_path / f"fold-{fold}",
    )
    monkeypatch.setattr(experiment, "_curve_dir", lambda: tmp_path / "curves")
    monkeypatch.setattr(
        experiment,
        "_holdout_dir",
        lambda preprocessing, target, model_name: tmp_path / "holdout",
    )
    experiment.run_one(
        _data(), "p1", "next_rpt_Q_Ah", "Ridge", Ridge(), "regression", n_splits=3
    )
    prediction_path = tmp_path / "fold-0" / "predictions.parquet"
    assert prediction_path.exists()
    no_features = _data()[["condition", "cell", "next_rpt_Q_Ah"]]
    rows = experiment.run_one(
        no_features, "p1", "next_rpt_Q_Ah", "Ridge", Ridge(), "regression", n_splits=3
    )
    assert any(row["status"] == "failed" for row in rows)
    assert not prediction_path.exists()
