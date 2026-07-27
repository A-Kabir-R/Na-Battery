"""Development/holdout and inner/outer split leakage assertions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pinn.dataset import apply_split_manifest, fold_indices, inner_split_indices


def _dummy_frame(cells: list[str], anchors_per_cell: int = 3) -> pd.DataFrame:
    rows = []
    for i, cell in enumerate(cells):
        for a in range(anchors_per_cell):
            rows.append({"cell": cell, "condition": f"C{i % 2}"})
    return pd.DataFrame(rows)


def _dummy_manifest(cells: list[str]) -> pd.DataFrame:
    n_dev = len(cells) - 2
    rows = []
    for i, cell in enumerate(cells):
        role = "development" if i < n_dev else "holdout"
        rows.append({
            "cell": cell, "condition": f"C{i % 2}",
            "outer_role": role,
            "cv_fold": (i % 5) if role == "development" else pd.NA,
        })
    return pd.DataFrame(rows)


def test_apply_split_manifest_populates_roles() -> None:
    cells = [f"cell_{i}" for i in range(10)]
    frame = _dummy_frame(cells)
    manifest = _dummy_manifest(cells)
    attached = apply_split_manifest(frame, manifest)
    assert "outer_role" in attached.columns
    assert set(attached["outer_role"].unique()) <= {"development", "holdout"}


def test_no_train_val_cell_overlap() -> None:
    cells = [f"cell_{i}" for i in range(10)]
    manifest = _dummy_manifest(cells)
    frame = apply_split_manifest(_dummy_frame(cells), manifest)
    for fold in range(5):
        train_idx, val_idx = fold_indices(frame, fold)
        train_cells = set(frame.iloc[train_idx]["cell"].astype(str))
        val_cells = set(frame.iloc[val_idx]["cell"].astype(str))
        assert not (train_cells & val_cells)


def test_no_development_holdout_overlap() -> None:
    cells = [f"cell_{i}" for i in range(10)]
    manifest = _dummy_manifest(cells)
    frame = apply_split_manifest(_dummy_frame(cells), manifest)
    development = set(frame[frame["outer_role"] == "development"]["cell"].astype(str))
    holdout = set(frame[frame["outer_role"] == "holdout"]["cell"].astype(str))
    assert not (development & holdout)


def test_inner_split_disjoint() -> None:
    cells = [f"cell_{i}" for i in range(8)]
    frame = _dummy_frame(cells)
    train_idx, val_idx = inner_split_indices(frame, seed=1)
    train_cells = set(frame.iloc[train_idx]["cell"].astype(str))
    val_cells = set(frame.iloc[val_idx]["cell"].astype(str))
    assert not (train_cells & val_cells)
