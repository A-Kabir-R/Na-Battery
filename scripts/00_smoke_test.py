"""Non-destructive parser, RPT, split, and fold-preprocessing smoke test."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from sklearn.linear_model import Ridge

from src.evaluation.metrics import regression_metrics
from src.features.selection import FoldPreprocessor
from src.io.canonical import discover_standard_cycling_files
from src.io.ird_parser import parse_ird
from src.io.loaders import load_config
from src.preprocessing.protocol_segmentation import extract_reference_discharge, summarize_steps
from src.splits.group_kfold import assert_no_cell_overlap, make_folds


def main() -> None:
    started = time.time()
    cfg = load_config()
    files = discover_standard_cycling_files(cfg["paths"]["dataset_root"])
    if not files:
        raise RuntimeError("no Standard-cycling IRD files discovered")
    sample = files[0]
    parsed = parse_ird(sample)
    assert parsed.data is not None
    steps = summarize_steps(parsed.data)
    reference = extract_reference_discharge(parsed.data, steps=steps)
    print(
        f"[smoke] parsed {sample.name}: rows={len(parsed.data)} steps={len(steps)} "
        f"reference_status={reference.status}"
    )

    frame = pd.DataFrame({
        "condition": [f"condition-{index % 3}" for index in range(9)],
        "cell": [f"cell-{index}" for index in range(9)],
        "feature_a": [float(index) for index in range(9)],
        "feature_b": [float(index * index + 1) for index in range(9)],
        "next_rpt_Q_Ah": [1.2 - 0.01 * index for index in range(9)],
    })
    train_index, test_index = next(iter(make_folds(frame, n_splits=3)))
    assert_no_cell_overlap(train_index, test_index, frame["cell"])
    processor = FoldPreprocessor(scale=True)
    x_train = processor.fit_transform(frame.iloc[train_index], target="next_rpt_Q_Ah")
    x_test = processor.transform(frame.iloc[test_index])
    estimator = Ridge(alpha=1.0).fit(x_train, frame.iloc[train_index]["next_rpt_Q_Ah"])
    metrics = regression_metrics(
        frame.iloc[test_index]["next_rpt_Q_Ah"], estimator.predict(x_test)
    )
    print(
        f"[smoke] grouped fold: train_cells={len(train_index)} test_cells={len(test_index)} "
        f"features={len(processor.feature_names_ or [])} metrics={metrics}"
    )
    print(f"[smoke] completed in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
