"""Temporal-leakage audit assertions."""
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
