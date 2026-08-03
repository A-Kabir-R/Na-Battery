"""Condition- and factor-disjoint splits must never leak a cell."""
from __future__ import annotations

import pandas as pd
import pytest

from src.splits.condition_holdout import (
    ConditionSplitError,
    describe_manifest,
    iter_splits,
    leave_one_condition_out,
    leave_one_factor_level_out,
)


def _frame() -> pd.DataFrame:
    """Three conditions x two cells x two anchors, plus a single-cell condition."""
    rows = []
    spec = [
        ("DOD100_1C1C_25degC", 25, 100, 1.0, 1.0, ["a1", "a2"]),
        ("DOD100_1C1C_40degC", 40, 100, 1.0, 1.0, ["b1", "b2"]),
        ("DOD20_2C2C_25degC", 25, 20, 2.0, 2.0, ["c1", "c2"]),
        ("DOD20_2C2C_40degC", 40, 20, 2.0, 2.0, ["d1"]),  # single cell
    ]
    for condition, t, dod, c_ch, c_dis, cells in spec:
        for cell in cells:
            for _ in range(2):
                rows.append({
                    "cell": cell, "condition": condition, "T_degC": t,
                    "DOD_pct": dod, "C_ch": c_ch, "C_dis": c_dis, "y": 1.0,
                })
    return pd.DataFrame(rows)


def test_loco_holds_out_every_condition_exactly_once():
    manifest = leave_one_condition_out(_frame(), min_train_cells=2)
    assert manifest["split_name"].nunique() == 4
    held = manifest.loc[manifest["outer_role"].eq("test"), "condition"].unique()
    assert sorted(held) == sorted(_frame()["condition"].unique())


def test_loco_test_and_train_never_share_a_cell():
    frame = _frame()
    manifest = leave_one_condition_out(frame, min_train_cells=2)
    for _, train_idx, test_idx, _ in iter_splits(manifest, frame):
        train_cells = set(frame.iloc[train_idx]["cell"])
        test_cells = set(frame.iloc[test_idx]["cell"])
        assert not (train_cells & test_cells)


def test_loco_test_set_contains_only_the_held_out_condition():
    frame = _frame()
    manifest = leave_one_condition_out(frame, min_train_cells=2)
    for split_name, _, test_idx, info in iter_splits(manifest, frame):
        conditions = set(frame.iloc[test_idx]["condition"])
        assert conditions == {info["held_out_value"]}, split_name


def test_single_cell_condition_is_flagged_degenerate_not_dropped():
    manifest = leave_one_condition_out(_frame(), min_train_cells=2)
    summary = describe_manifest(manifest).set_index("held_out_value")
    assert bool(summary.loc["DOD20_2C2C_40degC", "degenerate"]) is True
    assert bool(summary.loc["DOD100_1C1C_25degC", "degenerate"]) is False
    # Dropping it silently would misreport the study's coverage.
    assert "DOD20_2C2C_40degC" in set(manifest["held_out_value"])


def test_factor_holdout_removes_every_cell_at_the_level():
    frame = _frame()
    manifest = leave_one_factor_level_out(
        frame, factors=("T_degC",), min_train_cells=2,
    )
    for _, train_idx, test_idx, info in iter_splits(manifest, frame):
        level = float(info["held_out_value"])
        assert set(frame.iloc[test_idx]["T_degC"]) == {level}
        assert level not in set(frame.iloc[train_idx]["T_degC"]), (
            "an omitted factor level must not survive anywhere in training"
        )


def test_factor_holdout_skips_levels_that_would_exhaust_training():
    frame = _frame()
    manifest = leave_one_factor_level_out(
        frame, factors=("DOD_pct",), min_train_cells=2,
    )
    assert manifest["split_name"].nunique() >= 1


def test_cell_in_two_conditions_is_rejected():
    frame = _frame()
    frame.loc[frame["cell"].eq("a1"), "condition"] = ["X", "Y"]
    with pytest.raises(ConditionSplitError, match="more than one condition"):
        leave_one_condition_out(frame, min_train_cells=1)


def test_split_id_is_content_addressed_and_stable():
    frame = _frame()
    first = leave_one_condition_out(frame, min_train_cells=2)
    second = leave_one_condition_out(frame.sample(frac=1.0, random_state=0),
                                     min_train_cells=2)
    assert first["split_id"].iloc[0] == second["split_id"].iloc[0], (
        "the manifest hash must not depend on input row order"
    )


def test_loco_refuses_to_exhaust_the_training_set():
    frame = _frame()
    with pytest.raises(ConditionSplitError, match="training cells"):
        leave_one_condition_out(frame, min_train_cells=99)


def test_iter_splits_rejects_a_hand_edited_overlapping_manifest():
    frame = _frame()
    manifest = leave_one_condition_out(frame, min_train_cells=2)
    target = manifest["split_name"].iloc[0]
    victim = manifest.index[
        manifest["split_name"].eq(target) & manifest["outer_role"].eq("train")
    ][0]
    manifest.loc[victim, "outer_role"] = "test"
    manifest = pd.concat([manifest, manifest.loc[[victim]].assign(outer_role="train")])
    with pytest.raises(ConditionSplitError, match="cell overlap"):
        list(iter_splits(manifest, frame))
