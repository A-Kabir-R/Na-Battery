"""Temporal-leakage audit assertions (Stage 1 diagnosis fix #6)."""
from __future__ import annotations

from src.pinn.dataset import (
    FORBIDDEN_COLUMNS, TARGETS, build_temporal_feature_audit,
)


def test_targets_are_forbidden() -> None:
    for column in TARGETS:
        assert column in FORBIDDEN_COLUMNS


def test_audit_rejects_forbidden_columns() -> None:
    audit = build_temporal_feature_audit(
        ["EFC_cum", "T_mean", "next_rpt_Q_Ah", "next_rpt_horizon_days"],
        preprocessing="p2",
    )
    row_efc = audit[audit["feature"] == "EFC_cum"].iloc[0]
    row_next = audit[audit["feature"] == "next_rpt_Q_Ah"].iloc[0]
    assert bool(row_efc["allowed_as_input"]) is True
    assert bool(row_next["allowed_as_input"]) is False
    assert row_next["rejection_reason"]


def test_audit_rejects_leaky_prefixes_and_suffixes() -> None:
    audit = build_temporal_feature_audit(
        [
            "future_capacity_50cycles",
            "planned_rpt_visit",
            "T_mean_ahead",
            "voltage_lookahead",
            "current_forward",
            "temperature_centered",
            "V_max_future",
            "centered_roll_mean_T",
        ],
        preprocessing="p2",
    )
    assert (audit["status"] == "rejected").all(), audit
    assert (~audit["allowed_as_input"]).all()


def test_audit_marks_unknown_columns_unaudited_and_rejects_by_default() -> None:
    audit = build_temporal_feature_audit(
        ["some_unregistered_feature", "T_mean"], preprocessing="p2",
    )
    unknown = audit[audit["feature"] == "some_unregistered_feature"].iloc[0]
    assert unknown["status"] == "unaudited"
    assert bool(unknown["allowed_as_input"]) is False
    assert "unaudited" in unknown["rejection_reason"]
    known = audit[audit["feature"] == "T_mean"].iloc[0]
    assert known["status"] == "allowed"
    assert bool(known["allowed_as_input"]) is True


def test_audit_allow_unaudited_opt_in() -> None:
    audit = build_temporal_feature_audit(
        ["some_unregistered_feature"], preprocessing="p2",
        allow_unaudited=True,
    )
    row = audit.iloc[0]
    assert row["status"] == "unaudited"
    assert bool(row["allowed_as_input"]) is True


def test_audit_allow_unaudited_does_not_override_rejections() -> None:
    audit = build_temporal_feature_audit(
        ["next_rpt_Q_Ah", "future_capacity"], preprocessing="p2",
        allow_unaudited=True,
    )
    assert (audit["status"] == "rejected").all()
    assert (~audit["allowed_as_input"]).all()
