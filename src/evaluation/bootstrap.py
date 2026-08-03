"""Cell-clustered paired bootstrap for comparing two models on paired predictions.

Resamples *complete cells* (cluster bootstrap), preserving all anchors belonging
to each resampled cell. This accounts for within-cell correlation that would
otherwise make the bootstrap overconfident.

Reference statistic: mean(|e_A| - |e_B|)
  negative → model A has lower MAE
  positive → model B has lower MAE
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_cell_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    cells: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict:
    """Cell-clustered paired bootstrap: mean(|e_A| - |e_B|) and CI.

    Parameters
    ----------
    errors_a, errors_b:
        Residuals (y_pred - y_true) for model A and B. Must be the same length
        and aligned row-by-row (same anchor, same fold for each index).
    cells:
        Cell label for each row (same length as errors_a/b).
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.
    ci_level:
        Confidence interval level (0.95 → 95% CI).

    Returns
    -------
    dict with keys:
        mean_diff       mean(|e_A| - |e_B|); negative = A better
        ci_low          lower CI bound
        ci_high         upper CI bound
        p_a_better      fraction of bootstrap draws where A has lower MAE
        n_cells         number of unique cells
        n_anchors       number of anchor rows
    """
    ae_a = np.abs(errors_a)
    ae_b = np.abs(errors_b)
    diff = ae_a - ae_b  # negative = A better, positive = B better

    unique_cells = np.unique(cells)
    n_cells = len(unique_cells)
    n_anchors = len(diff)

    # Build a fast index: cell → row indices
    cell_index: dict[object, np.ndarray] = {
        c: np.where(cells == c)[0] for c in unique_cells
    }

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_cells = rng.choice(unique_cells, size=n_cells, replace=True)
        indices = np.concatenate([cell_index[c] for c in sampled_cells])
        boot_means[i] = diff[indices].mean()

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_diff = float(diff.mean())
    p_a_better = float((boot_means < 0).mean())

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_a_better": p_a_better,
        "n_cells": n_cells,
        "n_anchors": n_anchors,
    }


def run_pairwise_bootstrap(
    predictions: pd.DataFrame,
    model_col: str = "model",
    error_col: str = "error",
    cell_col: str = "cell",
    reference: str = "NaPINN-Q",
    n_bootstrap: int = 10_000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Run paired bootstrap for each (reference, comparator) pair.

    Parameters
    ----------
    predictions:
        Long-format DataFrame with columns [model_col, error_col, cell_col,
        plus any join keys]. Must contain aligned rows (same anchor appears
        once per model after an inner join on the anchor key).
    reference:
        The model that is the "A" side (e.g. "NaPINN-Q"). Negative mean_diff
        means the reference has lower MAE.
    n_bootstrap, seed, ci_level:
        Passed to paired_cell_bootstrap.

    Returns
    -------
    DataFrame with one row per comparator, columns:
        reference, comparator, mean_diff, ci_low, ci_high, p_reference_better,
        n_cells, n_anchors, interpretation
    """
    rows = []
    ref_df = predictions[predictions[model_col].eq(reference)].set_index("_anchor_key")
    for comparator, comp_df in predictions.groupby(model_col):
        if comparator == reference:
            continue
        comp_df = comp_df.set_index("_anchor_key")
        shared = ref_df.index.intersection(comp_df.index)
        if len(shared) == 0:
            continue
        err_ref = ref_df.loc[shared, error_col].to_numpy(dtype=float)
        err_comp = comp_df.loc[shared, error_col].to_numpy(dtype=float)
        cells = ref_df.loc[shared, cell_col].to_numpy()
        result = paired_cell_bootstrap(
            err_ref, err_comp, cells,
            n_bootstrap=n_bootstrap, seed=seed, ci_level=ci_level,
        )
        if result["ci_high"] < 0:
            interp = f"{reference} significantly better"
        elif result["ci_low"] > 0:
            interp = f"{comparator} significantly better"
        else:
            interp = "unresolved"
        rows.append({
            "reference": reference,
            "comparator": comparator,
            "mean_diff": result["mean_diff"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_reference_better": result["p_a_better"],
            "n_cells": result["n_cells"],
            "n_anchors": result["n_anchors"],
            "interpretation": interp,
        })
    return pd.DataFrame(rows)
