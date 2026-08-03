"""Condition-disjoint and factor-level-disjoint evaluation splits.

Cell-grouped cross-validation answers *"a new cell under a condition we have
already seen"*. That is the weakest generalization claim available, because
three cells share each condition, so the model can learn the condition's
degradation rate from its siblings. Measured on this dataset, cell-grouped CV is
about 35% optimistic relative to leave-one-condition-out (MAE 0.00284 -> 0.00436).

Two stronger protocols live here:

``leave_one_condition_out``
    Hold out every cell of one complete operating condition
    ``(T, DOD, C_ch, C_dis)``. Tests generalization to an unseen *combination*.

``leave_one_factor_level_out``
    Hold out every cell whose temperature (or DOD, or charge/discharge rate)
    takes one value. Tests true *extrapolation* to an unseen level, which is
    strictly stronger: a condition can be interpolated when all of its factor
    levels appear elsewhere, but an omitted level cannot.

Both emit long-format manifests with the same columns as the locked split
manifest, so downstream code that already understands ``outer_role`` works
unchanged. Every manifest carries a ``split_id`` hash of its own content.
"""
from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pandas as pd

#: Columns whose tuple defines one operating condition.
FACTOR_COLUMNS: tuple[str, ...] = ("T_degC", "DOD_pct", "C_ch", "C_dis")

#: Manifest column order, matching build_locked_split_manifest.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "study", "split_name", "cell", "condition", "outer_role",
    "n_rows", "n_cells_held_out", "n_conditions_held_out",
    "held_out_value", "factor", "degenerate", "split_id",
)


class ConditionSplitError(RuntimeError):
    """Raised when a requested split cannot be built safely."""


def _split_id(frame: pd.DataFrame) -> str:
    """Content hash of a manifest, stable under row order."""
    payload = (
        frame.drop(columns=["split_id"], errors="ignore")
        .sort_values(["split_name", "cell"])
        .to_dict(orient="records")
    )
    return sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per cell carrying its condition and factor levels.

    A cell belonging to two conditions would silently corrupt every split, so
    that is an error rather than a warning.
    """
    required = ("cell", "condition", *FACTOR_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ConditionSplitError(f"frame is missing columns: {missing}")
    table = (
        frame[list(required)].drop_duplicates().reset_index(drop=True)
    )
    duplicated = table["cell"].duplicated()
    if duplicated.any():
        offenders = sorted(table.loc[duplicated, "cell"].unique())
        raise ConditionSplitError(
            f"cell(s) assigned to more than one condition: {offenders}"
        )
    counts = frame.groupby("cell").size().rename("n_rows")
    return table.merge(counts, left_on="cell", right_index=True, how="left")


def _assert_disjoint(manifest: pd.DataFrame) -> None:
    for split_name, group in manifest.groupby("split_name"):
        roles = group.groupby("cell")["outer_role"].nunique()
        clashing = sorted(roles[roles > 1].index)
        if clashing:
            raise ConditionSplitError(
                f"{split_name}: cell(s) in both train and test: {clashing}"
            )


def leave_one_condition_out(frame: pd.DataFrame, *,
                            min_train_cells: int = 4,
                            study: str = "loco") -> pd.DataFrame:
    """One split per operating condition, holding out all of its cells.

    Conditions represented by a single cell (six of fifteen here) are still
    emitted, flagged ``degenerate=True``: a single-cell, often single-anchor
    test set produces an estimate too noisy to interpret on its own, but
    silently dropping it would misreport the coverage of the study. Filter on
    the flag when reporting; do not filter it out of the manifest.
    """
    cells = _cell_table(frame)
    rows: list[dict[str, object]] = []
    for condition, group in cells.groupby("condition", sort=True):
        held_cells = set(group["cell"])
        n_train_cells = len(cells) - len(held_cells)
        if n_train_cells < min_train_cells:
            raise ConditionSplitError(
                f"holding out {condition} leaves only {n_train_cells} training "
                f"cells (minimum {min_train_cells})"
            )
        degenerate = len(held_cells) < 2 or int(group["n_rows"].sum()) < 2
        for _, cell_row in cells.iterrows():
            rows.append({
                "study": study,
                "split_name": f"loco__{condition}",
                "cell": cell_row["cell"],
                "condition": cell_row["condition"],
                "outer_role": (
                    "test" if cell_row["cell"] in held_cells else "train"
                ),
                "n_rows": int(cell_row["n_rows"]),
                "n_cells_held_out": len(held_cells),
                "n_conditions_held_out": 1,
                "held_out_value": condition,
                "factor": "condition",
                "degenerate": bool(degenerate),
            })
    if not rows:
        raise ConditionSplitError("no conditions available for LOCO")
    manifest = pd.DataFrame(rows)
    manifest["split_id"] = _split_id(manifest)
    _assert_disjoint(manifest)
    return manifest[list(MANIFEST_COLUMNS)]


def leave_one_factor_level_out(frame: pd.DataFrame, *,
                               factors: tuple[str, ...] = FACTOR_COLUMNS,
                               min_train_cells: int = 4,
                               study: str = "factor_holdout") -> pd.DataFrame:
    """One split per (factor, level), holding out every cell at that level.

    Levels whose removal would leave fewer than ``min_train_cells`` training
    cells are skipped and reported by :func:`describe_manifest`, rather than
    raising -- unlike LOCO, an un-runnable factor level is an expected feature of
    an unbalanced test matrix, not a configuration error.
    """
    cells = _cell_table(frame)
    rows: list[dict[str, object]] = []
    for factor in factors:
        if factor not in cells.columns:
            raise ConditionSplitError(f"unknown factor column: {factor}")
        for level, group in cells.groupby(factor, sort=True):
            held_cells = set(group["cell"])
            if len(cells) - len(held_cells) < min_train_cells:
                continue
            degenerate = len(held_cells) < 2 or int(group["n_rows"].sum()) < 2
            for _, cell_row in cells.iterrows():
                rows.append({
                    "study": study,
                    "split_name": f"{factor}__{level}",
                    "cell": cell_row["cell"],
                    "condition": cell_row["condition"],
                    "outer_role": (
                        "test" if cell_row["cell"] in held_cells else "train"
                    ),
                    "n_rows": int(cell_row["n_rows"]),
                    "n_cells_held_out": len(held_cells),
                    "n_conditions_held_out": int(group["condition"].nunique()),
                    "held_out_value": str(level),
                    "factor": factor,
                    "degenerate": bool(degenerate),
                })
    if not rows:
        raise ConditionSplitError(
            "no factor level could be held out without exhausting the training set"
        )
    manifest = pd.DataFrame(rows)
    manifest["split_id"] = _split_id(manifest)
    _assert_disjoint(manifest)
    return manifest[list(MANIFEST_COLUMNS)]


def iter_splits(manifest: pd.DataFrame, frame: pd.DataFrame):
    """Yield ``(split_name, train_idx, test_idx, info)`` positional indices.

    Indices address ``frame`` positionally, so the caller can use ``frame.iloc``
    directly. Cell disjointness is re-checked per split: a manifest can be
    edited by hand between generation and use.
    """
    cells = frame["cell"].astype(str).to_numpy()
    for split_name, group in manifest.groupby("split_name", sort=True):
        test_cells = set(group.loc[group["outer_role"].eq("test"), "cell"].astype(str))
        train_cells = set(group.loc[group["outer_role"].eq("train"), "cell"].astype(str))
        overlap = test_cells & train_cells
        if overlap:
            raise ConditionSplitError(f"{split_name}: cell overlap {sorted(overlap)}")
        test_idx = np.flatnonzero(np.isin(cells, list(test_cells)))
        train_idx = np.flatnonzero(np.isin(cells, list(train_cells)))
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        first = group.iloc[0]
        yield split_name, train_idx, test_idx, {
            "factor": first["factor"],
            "held_out_value": first["held_out_value"],
            "degenerate": bool(first["degenerate"]),
            "n_cells_held_out": int(first["n_cells_held_out"]),
            "split_id": first["split_id"],
        }


def describe_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-split summary: sizes, and whether the split is interpretable."""
    test = manifest[manifest["outer_role"].eq("test")]
    train = manifest[manifest["outer_role"].eq("train")]
    summary = (
        test.groupby(["study", "factor", "split_name", "held_out_value",
                      "degenerate"], dropna=False)
        .agg(test_cells=("cell", "nunique"), test_rows=("n_rows", "sum"))
        .reset_index()
    )
    train_sizes = (
        train.groupby("split_name")
        .agg(train_cells=("cell", "nunique"), train_rows=("n_rows", "sum"))
    )
    return summary.merge(train_sizes, on="split_name", how="left")
