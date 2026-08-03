#!/usr/bin/env python3
"""Nested selection of the PINN refit epoch count.

``train_fold`` early-stops on one cell-grouped inner split. With 26 development
cells and five outer folds that validation half holds four or five cells, so the
selected epoch moves when the split is re-drawn -- the number is not identified
by the data, and a single-split run is reporting a coin flip.

This script does proper nested selection per outer fold:

1. Run phase 1 once per inner fold (``--inner-folds``), refit disabled, so each
   run contributes only its validation curve.
2. Average the curves over their **common** epoch range and take the argmin.
3. Run once more with ``forced_best_epoch`` set to that consensus, so the
   two-phase refit trains on all outer-training cells for exactly that long.

Phase 1 itself is untouched: no training dynamics and no leakage boundary moves,
only the refit length changes. Trainer settings come from
:func:`src.pinn.config_builder.build_trainer_config`, the same builder the main
sweep uses, so the selection runs cannot silently differ from the runs they are
selecting for.

Writes, under ``artifacts/results/pinn_nested_selection/``:

    selection_per_fold.csv    consensus epoch and the per-fold argmin spread
    mean_curve_fold<k>.csv    the averaged inner curve, for the supplement

Report the spread. It is exactly how much a single-split run was leaving to
chance.

Usage::

    python3 scripts/15_run_nested_pinn_selection.py --dry-run
    python3 scripts/15_run_nested_pinn_selection.py --outer-folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from src.io.loaders import load_config  # noqa: E402
from src.pinn.nested_selection import (  # noqa: E402
    NestedSelectionError, load_epoch_logs, select_epoch, selection_report,
)


def _selection_run_dir(root: Path, architecture: str, preprocessing: str,
                       fold: int, seed: int, inner_fold: int) -> Path:
    """Selection runs live beside the main runs, never on top of them."""
    return (
        root / "nested_selection_runs" / architecture / preprocessing
        / f"fold_{fold}" / f"seed_{seed}" / f"inner_{inner_fold}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", default="NaPINN-Q")
    parser.add_argument("--preprocessing", default="unified")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epoch", type=int, default=0,
                        help="mirror the checkpoint-eligibility gate")
    parser.add_argument("--skip-refit", action="store_true",
                        help="select only; do not run the final refit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    pinn_cfg = cfg["pinn"]
    artifacts = Path(cfg["paths"]["artifacts"])
    features_dir = artifacts / "features"
    # Same resolution as scripts/06 `_results_dir`, so both runners agree on
    # where the fold artifacts live.
    results_root = artifacts / "results" / str(
        (cfg.get("pinn") or {}).get("results_subdir")
        or "pinn_preprocessing_study"
    )
    out = args.output_dir or (artifacts / "results" / "pinn_nested_selection")
    out.mkdir(parents=True, exist_ok=True)

    plan = [(fold, inner) for fold in args.outer_folds
            for inner in range(args.inner_folds)]
    print(f"[nested] {len(args.outer_folds)} outer x {args.inner_folds} inner "
          f"= {len(plan)} phase-1 runs"
          f"{'' if args.skip_refit else f', then {len(args.outer_folds)} refits'}")
    if args.dry_run:
        for fold, inner in plan:
            print(f"  fold {fold} inner {inner} -> " + str(_selection_run_dir(
                results_root, args.architecture, args.preprocessing,
                fold, args.seed, inner)))
        return 0

    # Imported late: these pull in torch, and --dry-run should not pay for it.
    from src.pinn.config_builder import (  # noqa: E402
        build_train_fold_kwargs, build_trainer_config,
    )
    from src.pinn.dataset import apply_split_manifest, build_anchor_dataset  # noqa: E402
    from src.pinn.trainer import FoldPaths, train_fold  # noqa: E402

    feature_file = features_dir / f"{args.preprocessing}.parquet"
    if not feature_file.exists():
        raise SystemExit(
            f"missing {feature_file}. Run scripts/01_build_features.py first, or "
            f"point SIB_ARTIFACTS at a built artifacts directory."
        )
    split_file = artifacts / "splits" / "split_manifest.parquet"
    if not split_file.exists():
        raise SystemExit(f"missing {split_file}. Run scripts/02 to build it.")

    frame = pd.read_parquet(feature_file)
    dataset = build_anchor_dataset(
        frame, preprocessing=args.preprocessing, stress_cfg=pinn_cfg["stress"],
        audit_cfg=pinn_cfg.get("audit"),
        audit_path=results_root / f"temporal_feature_audit_{args.preprocessing}.csv",
        features_dir=features_dir,
    )
    attached = apply_split_manifest(dataset.frame, pd.read_parquet(split_file))
    shared = build_train_fold_kwargs(
        pinn_cfg, cfg, preprocessing=args.preprocessing, features_dir=features_dir,
    )

    rows: list[pd.DataFrame] = []
    for fold in args.outer_folds:
        logs: dict[int, Path] = {}
        for inner in range(args.inner_folds):
            run_dir = _selection_run_dir(
                results_root, args.architecture, args.preprocessing,
                fold, args.seed, inner,
            )
            config = build_trainer_config(
                pinn_cfg, architecture=args.architecture,
                preprocessing=args.preprocessing, fold=fold, seed=args.seed,
                max_epochs=args.max_epochs, device=args.device,
                inner_fold=inner, n_inner_folds=args.inner_folds,
                # Phase 1 only: this run exists to produce a validation curve.
                two_phase_refit=False,
            )
            print(f"[nested] fold {fold} inner {inner}: phase 1")
            train_fold(dataset=dataset, frame=attached, config=config,
                       fold_paths=FoldPaths(root=run_dir), **shared)
            logs[inner] = run_dir / "logs" / "epoch_log.csv"

        try:
            selection = select_epoch(load_epoch_logs(logs), min_epoch=args.min_epoch)
        except NestedSelectionError as error:
            print(f"[nested] fold {fold}: selection failed -- {error}")
            continue

        selection.mean_curve.to_csv(out / f"mean_curve_fold{fold}.csv", index=False)
        report = selection_report(selection)
        report.insert(0, "outer_fold", fold)
        rows.append(report)
        print(f"[nested] fold {fold}: consensus epoch {selection.best_epoch} "
              f"(per-fold {selection.per_fold_best_epoch}, "
              f"spread {selection.spread()})")

        if args.skip_refit:
            continue
        final_dir = (
            results_root / "runs" / args.architecture / args.preprocessing
            / f"fold_{fold}" / f"seed_{args.seed}"
        )
        final_config = build_trainer_config(
            pinn_cfg, architecture=args.architecture,
            preprocessing=args.preprocessing, fold=fold, seed=args.seed,
            max_epochs=args.max_epochs, device=args.device,
            inner_fold=0, n_inner_folds=args.inner_folds,
            two_phase_refit=True,
            forced_best_epoch=selection.best_epoch,
        )
        print(f"[nested] fold {fold}: refit for {selection.refit_epochs} epochs")
        train_fold(dataset=dataset, frame=attached, config=final_config,
                   fold_paths=FoldPaths(root=final_dir), force=True, **shared)

    if not rows:
        print("[nested] no fold produced a selection")
        return 1

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out / "selection_per_fold.csv", index=False)
    print("\n" + summary.to_string(index=False))
    widest = int(summary["per_fold_spread"].max())
    if widest > 0:
        print(f"\n[nested] widest per-fold spread: {widest} epochs -- report it, "
              f"it is how much a single-split run left to chance.")
    print(f"[nested] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
