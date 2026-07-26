"""GroupKFold by cell, stratified by condition when possible."""
from __future__ import annotations

from hashlib import sha256
import warnings

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold


def make_folds(df: pd.DataFrame, n_splits: int = 5,
               group_col: str = "cell", stratify_col: str | None = "condition",
               random_state: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    groups = df[group_col].to_numpy()
    if stratify_col and stratify_col in df.columns:
        y = df[stratify_col].astype("category").cat.codes.to_numpy()
        # StratifiedGroupKFold does not accept random_state prior to shuffle=True;
        # we shuffle groups outside instead.
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            yield from sgkf.split(np.zeros(len(df)), y, groups)
            return
        except Exception:
            pass
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(np.zeros(len(df)), groups=groups)


def assert_no_cell_overlap(train_idx: np.ndarray, test_idx: np.ndarray,
                            cells: pd.Series) -> None:
    tr = set(cells.iloc[train_idx].unique())
    te = set(cells.iloc[test_idx].unique())
    assert not (tr & te), f"cell overlap between train and test: {tr & te}"


def build_locked_split_manifest(df: pd.DataFrame, *, holdout_fraction: float = 0.25,
                                n_splits: int = 5, group_col: str = "cell",
                                stratify_col: str = "condition", random_state: int = 42,
                                holdout_candidates: int = 1024) -> pd.DataFrame:
    """Create one cell-level holdout and development-only CV assignment.

    Candidate grouped holdouts are scored for row-count and condition balance.
    The selected cells are then locked, and only development cells receive a
    five-fold validation assignment.
    """
    required = {group_col, stratify_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split data is missing columns: {sorted(missing)}")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")

    cell_conditions = df.groupby(group_col)[stratify_col].nunique(dropna=False)
    if (cell_conditions > 1).any():
        bad = cell_conditions[cell_conditions > 1].index.astype(str).tolist()
        raise ValueError(f"cells map to multiple conditions: {bad}")
    cells = (
        df.groupby(group_col, as_index=False)
        .agg(**{stratify_col: (stratify_col, "first")}, n_rows=(group_col, "size"))
        .sort_values(group_col)
        .reset_index(drop=True)
    )
    if len(cells) < n_splits + 1:
        raise ValueError(
            f"need at least {n_splits + 1} cells for a holdout plus {n_splits}-fold CV"
        )

    overall_condition = cells[stratify_col].value_counts(normalize=True)
    splitter = GroupShuffleSplit(
        n_splits=holdout_candidates,
        test_size=holdout_fraction,
        random_state=random_state,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for development_idx, holdout_idx in splitter.split(cells, groups=cells[group_col]):
        holdout = cells.iloc[holdout_idx]
        condition = holdout[stratify_col].value_counts(normalize=True)
        condition_error = float(
            overall_condition.subtract(condition, fill_value=0.0).abs().sum()
        )
        row_fraction = float(holdout["n_rows"].sum() / cells["n_rows"].sum())
        row_error = abs(row_fraction - holdout_fraction)
        cell_error = abs(len(holdout) / len(cells) - holdout_fraction)
        score = condition_error + 2.0 * row_error + 2.0 * cell_error
        if best is None or score < best[0]:
            best = (score, development_idx, holdout_idx)
    if best is None:
        raise RuntimeError("could not construct a grouped holdout")

    _, development_idx, holdout_idx = best
    cells["outer_role"] = "development"
    cells.loc[holdout_idx, "outer_role"] = "holdout"
    cells["cv_fold"] = pd.Series(pd.NA, index=cells.index, dtype="Int64")

    development_cells = set(cells.loc[development_idx, group_col])
    development_rows = df[df[group_col].isin(development_cells)].reset_index(drop=True)
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    assigned: dict[object, int] = {}
    y = development_rows[stratify_col].astype("category").cat.codes.to_numpy()
    groups = development_rows[group_col].to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The least populated class in y has only .* members"
        )
        for fold, (_, validation_idx) in enumerate(cv.split(
            np.zeros(len(development_rows)), y, groups
        )):
            for cell in development_rows.iloc[validation_idx][group_col].unique():
                if cell in assigned:
                    raise AssertionError(
                        f"development cell assigned to multiple CV folds: {cell}"
                    )
                assigned[cell] = fold
    if assigned.keys() != development_cells:
        missing_cells = sorted(str(cell) for cell in development_cells - assigned.keys())
        raise AssertionError(f"development cells missing CV assignments: {missing_cells}")
    cells.loc[cells["outer_role"] == "development", "cv_fold"] = (
        cells.loc[cells["outer_role"] == "development", group_col].map(assigned).astype("Int64")
    )
    cells["holdout_fraction"] = holdout_fraction
    cells["achieved_cell_holdout_fraction"] = float(
        cells["outer_role"].eq("holdout").mean()
    )
    cells["achieved_row_holdout_fraction"] = float(
        cells.loc[cells["outer_role"].eq("holdout"), "n_rows"].sum()
        / cells["n_rows"].sum()
    )
    cells["n_splits"] = n_splits
    cells["random_state"] = random_state
    development_condition_counts = cells.loc[
        cells["outer_role"].eq("development"), stratify_col
    ].value_counts()
    cells["cv_stratification_status"] = (
        "sparse_conditions_unavoidable"
        if (development_condition_counts < n_splits).any()
        else "fully_stratifiable"
    )
    result = cells.rename(columns={group_col: "cell", stratify_col: "condition"})[
        ["cell", "condition", "outer_role", "cv_fold", "n_rows",
         "holdout_fraction", "achieved_cell_holdout_fraction",
         "achieved_row_holdout_fraction", "n_splits", "random_state",
         "cv_stratification_status"]
    ]
    split_bytes = pd.util.hash_pandas_object(
        result[["cell", "condition", "outer_role", "cv_fold"]].sort_values("cell"),
        index=False,
    ).to_numpy().tobytes()
    result["split_id"] = sha256(split_bytes).hexdigest()
    return result


def validate_locked_split_manifest(df: pd.DataFrame, manifest: pd.DataFrame,
                                   *, n_splits: int = 5,
                                   holdout_fraction: float | None = None,
                                   random_state: int | None = None) -> None:
    required = {"cell", "condition", "outer_role", "cv_fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"split manifest is missing columns: {sorted(missing)}")
    if manifest["cell"].duplicated().any():
        raise ValueError("split manifest contains duplicate cells")
    data_cells = set(df["cell"].astype(str).unique())
    manifest_cells = set(manifest["cell"].astype(str).unique())
    if not data_cells <= manifest_cells:
        raise ValueError(f"unassigned cells: {sorted(data_cells - manifest_cells)}")
    current_conditions = (
        df.assign(cell=df["cell"].astype(str))
        .groupby("cell")["condition"].first().astype(str)
    )
    manifest_conditions = manifest.assign(cell=manifest["cell"].astype(str)).set_index(
        "cell"
    )["condition"].astype(str)
    mismatched = sorted(
        cell for cell in data_cells
        if manifest_conditions.get(cell) != current_conditions.get(cell)
    )
    if mismatched:
        raise ValueError(f"split manifest condition assignments changed: {mismatched}")
    roles = set(manifest["outer_role"].unique())
    if roles != {"development", "holdout"}:
        raise ValueError(f"split manifest has invalid outer roles: {sorted(roles)}")
    development = manifest[manifest["outer_role"] == "development"]
    holdout = manifest[manifest["outer_role"] == "holdout"]
    if holdout["cv_fold"].notna().any():
        raise ValueError("holdout cells must not have CV folds")
    folds = set(development["cv_fold"].dropna().astype(int).unique())
    if folds != set(range(n_splits)):
        raise ValueError(f"development CV folds are incomplete: {sorted(folds)}")
    if "n_splits" in manifest and set(manifest["n_splits"].astype(int)) != {n_splits}:
        raise ValueError("split manifest n_splits does not match current configuration")
    if holdout_fraction is not None and "holdout_fraction" in manifest:
        if not np.allclose(manifest["holdout_fraction"].astype(float), holdout_fraction):
            raise ValueError("split manifest holdout fraction does not match current configuration")
    if random_state is not None and "random_state" in manifest:
        if set(manifest["random_state"].astype(int)) != {random_state}:
            raise ValueError("split manifest random state does not match current configuration")
    if "split_id" in manifest:
        split_bytes = pd.util.hash_pandas_object(
            manifest[["cell", "condition", "outer_role", "cv_fold"]].sort_values("cell"),
            index=False,
        ).to_numpy().tobytes()
        expected_split_id = sha256(split_bytes).hexdigest()
        if set(manifest["split_id"].astype(str)) != {expected_split_id}:
            raise ValueError("split manifest checksum is invalid")
