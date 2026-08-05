"""Inference-only PINN checkpoint loader.

Restores a trained NaPINN-Q from a completed run directory and exposes a
predict function suitable for robustness, uncertainty, and low-data studies.

The loader reads the canonical artifact layout written by train_fold:

    run_dir/
        experiment_manifest.json    -- trainer_config dict (architecture params)
        checkpoints/
            final_refit_model.pt    -- two-phase refit weights (preferred)
            best_model.pt           -- single-phase weights (fallback)
        final_refit_scaler.json     -- FoldScaler state (preferred)
        scaler.json                 -- inner-train scaler (fallback)
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import FoldScaler
from .models import NaPINNQ
from .trainer import FoldPaths, TrainerConfig


def prepare_anchor_frame(raw_rows: pd.DataFrame, dataset) -> pd.DataFrame:
    """Join unified.parquet rows to the anchor dataset to get PINN-required columns.

    ``raw_rows`` must have an ``anchor_id`` column (present in unified.parquet).
    Returns a frame with the same row order as ``raw_rows``, enriched with
    ``u_current``, ``stress_current``, ``stress_next``, ``stress_delta``,
    ``initial_rpt_Q_Ah``, and all feature columns the scaler requires.

    Raises ValueError if any anchor_id is absent from ``dataset.frame`` or if
    ``anchor_id`` is duplicated in ``dataset.frame`` (would signal a data bug).
    """
    if "anchor_id" not in raw_rows.columns:
        raise ValueError(
            "raw_rows must contain 'anchor_id'; "
            "unified.parquet rows always carry this column"
        )
    if "anchor_id" not in dataset.frame.columns:
        raise ValueError("dataset.frame must contain 'anchor_id'")

    merged = (
        raw_rows[["anchor_id"]]
        .reset_index(drop=True)
        .merge(dataset.frame, on="anchor_id", how="left", validate="many_to_one")
    )
    missing_mask = merged["u_current"].isna()
    if missing_mask.any():
        bad = raw_rows.loc[missing_mask.to_numpy(), "anchor_id"].head(5).tolist()
        raise ValueError(
            f"{int(missing_mask.sum())} anchor_id(s) in raw_rows not found in "
            f"dataset.frame: {bad}. "
            "This usually means some rows in the study frame are not anchor rows "
            "(is_prediction_anchor=False). Filter study frames to anchor rows first."
        )
    return merged.reset_index(drop=True)


def _feature_matrix(frame: pd.DataFrame, scaler: FoldScaler) -> np.ndarray:
    """Reproduce trainer._build_feature_matrix without needing AnchorDataset."""
    scaled = scaler.transform_features(frame, [])
    horizon = scaler.transform_horizon(frame["stress_delta"].to_numpy(dtype=float))
    u_current = frame["u_current"].to_numpy(dtype=float)
    return np.concatenate([horizon[:, None], u_current[:, None], scaled], axis=1)


def _rebuild_model(cfg: TrainerConfig, scaler: FoldScaler, device: str) -> NaPINNQ:
    cols = list(scaler.feature_columns_used or [])
    n_features = len(cols)
    feature_dim = n_features + 2  # horizon + u_current prepended by _feature_matrix

    def _idx(name: str) -> int:
        return 2 + cols.index(name) if name in cols else 0

    hybrid_temp_idx = _idx(cfg.hybrid_temperature_feature) if cfg.use_hybrid_rate else 0
    hybrid_dod_idx = _idx(cfg.hybrid_dod_feature) if cfg.use_hybrid_rate else 0
    hybrid_crate_idx = (
        _idx(cfg.hybrid_c_rate_feature)
        if cfg.use_hybrid_rate and cfg.hybrid_c_rate_feature and cfg.hybrid_c_rate_feature in cols
        else hybrid_dod_idx
    )
    model = NaPINNQ(
        feature_dim=feature_dim,
        solution_hidden_dims=cfg.solution_hidden_dims,
        rate_hidden_dims=cfg.rate_hidden_dims,
        solution_activation=cfg.solution_activation,
        rate_activation=cfg.rate_activation,
        solution_dropout=cfg.solution_dropout,
        rate_dropout=cfg.rate_dropout,
        rate_uses_u_hat=cfg.rate_uses_u_hat,
        predict_delta_u=cfg.predict_delta_u,
        rate_floor=cfg.rate_floor,
        use_hybrid_rate=cfg.use_hybrid_rate,
        hybrid_temperature_index=hybrid_temp_idx,
        hybrid_dod_index=hybrid_dod_idx,
        hybrid_c_rate_index=hybrid_crate_idx,
        hybrid_enable_cold_regime=cfg.hybrid_enable_cold_regime,
        hybrid_fit_c_rate_exponent=cfg.hybrid_fit_c_rate_exponent,
    )
    model.to(device)
    model.eval()
    return model


def _load_config_from_manifest(manifest: dict[str, Any]) -> TrainerConfig:
    trainer_cfg_dict = manifest.get("trainer_config", {})
    valid_fields = {f.name for f in fields(TrainerConfig)}
    filtered: dict[str, Any] = {k: v for k, v in trainer_cfg_dict.items() if k in valid_fields}
    for tuple_key in ("solution_hidden_dims", "rate_hidden_dims"):
        if tuple_key in filtered and isinstance(filtered[tuple_key], list):
            filtered[tuple_key] = tuple(int(h) for h in filtered[tuple_key])
    if filtered:
        return TrainerConfig(**filtered)
    return TrainerConfig(architecture="NaPINN-Q", preprocessing="unified", fold=0, seed=42)


class PINNInferenceSession:
    """Load a trained NaPINN-Q from a completed run directory for inference.

    Checkpoint priority:
        1. checkpoints/final_refit_model.pt + final_refit_scaler.json  (two-phase)
        2. checkpoints/best_model.pt         + scaler.json              (fallback)

    Parameters
    ----------
    run_dir:
        The fold-seed directory, e.g.
        ``artifacts/results/pinn_unified_study/runs/NaPINN-Q/unified/fold_0/seed_42``
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, run_dir: Path | str, device: str = "cpu") -> None:
        self.run_dir = Path(run_dir)
        self.device = device
        paths = FoldPaths(root=self.run_dir)

        # --- scaler ----------------------------------------------------------
        if paths.final_scaler.exists():
            scaler_state = json.loads(paths.final_scaler.read_text())
        elif paths.scaler_path.exists():
            scaler_state = json.loads(paths.scaler_path.read_text())
        else:
            raise FileNotFoundError(
                f"no scaler found in {self.run_dir}; "
                "run scripts/06_run_pinn_experiments.py first"
            )
        self.scaler: FoldScaler = FoldScaler.from_state_dict(scaler_state)

        # --- trainer config (for architecture) -------------------------------
        if not paths.manifest_path.exists():
            raise FileNotFoundError(
                f"experiment_manifest.json missing in {self.run_dir}; "
                "the run may not have completed"
            )
        manifest = json.loads(paths.manifest_path.read_text())
        self.config: TrainerConfig = _load_config_from_manifest(manifest)

        # --- model -----------------------------------------------------------
        self.model: NaPINNQ = _rebuild_model(self.config, self.scaler, device)

        if paths.final_model.exists():
            ckpt = torch.load(paths.final_model, map_location="cpu", weights_only=False)
        elif paths.best_model.exists():
            ckpt = torch.load(paths.best_model, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(
                f"no model checkpoint found in {self.run_dir}"
            )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict next-RPT normalized capacity (u_next) for each row.

        ``frame`` must contain: ``stress_next``, ``stress_current``,
        ``stress_delta``, ``u_current``, and all feature columns the
        scaler was fitted on.

        Returns
        -------
        np.ndarray of shape (n_rows,)
        """
        features_np = _feature_matrix(frame, self.scaler)
        stress_next_np = self.scaler.transform_stress(
            frame["stress_next"].to_numpy(dtype=float)
        )
        stress_current_np = self.scaler.transform_stress(
            frame["stress_current"].to_numpy(dtype=float)
        )
        u_current_np = frame["u_current"].to_numpy(dtype=float)

        features = torch.as_tensor(features_np, dtype=torch.float32, device=self.device)
        stress_next = torch.as_tensor(stress_next_np, dtype=torch.float32, device=self.device)
        stress_current = torch.as_tensor(stress_current_np, dtype=torch.float32, device=self.device)
        u_current = torch.as_tensor(u_current_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            u_next = self.model.solution(stress_next, features, u_current, stress_current)
        return u_next.cpu().numpy()

    def predict_delta_q_ah(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict capacity change in Ah: (u_next - u_current) * initial_rpt_Q_Ah.

        ``frame`` must be an anchor-dataset frame (from ``prepare_anchor_frame``
        or ``dataset.frame`` directly). This is the authoritative method for
        producing predictions on the same scale as ``delta_next_rpt_Q_Ah``.

        Returns
        -------
        np.ndarray of shape (n_rows,) in Ah.
        """
        u_next = self.predict(frame)
        u_current = frame["u_current"].to_numpy(dtype=float)
        q0 = frame["initial_rpt_Q_Ah"].to_numpy(dtype=float)
        return ((u_next - u_current) * q0).astype(float)

    def predict_delta(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict normalised capacity change u_next - u_current (dimensionless).

        Use ``predict_delta_q_ah`` to get predictions in Ah on the same scale
        as ``delta_next_rpt_Q_Ah``. This method is kept for backward
        compatibility; it returns the raw normalised delta, not Ah.
        """
        u_current_np = frame["u_current"].to_numpy(dtype=float)
        u_next_np = self.predict(frame)
        return (u_next_np - u_current_np).astype(float)


def find_run_dir(
    pinn_results_dir: Path,
    architecture: str = "NaPINN-Q",
    preprocessing: str = "unified",
    fold: int = 0,
    seed: int = 42,
) -> Path:
    """Resolve the canonical run directory for a given config."""
    return pinn_results_dir / "runs" / architecture / preprocessing / f"fold_{fold}" / f"seed_{seed}"


def load_fold_predictions(
    pinn_results_dir: Path,
    architecture: str = "NaPINN-Q",
    preprocessing: str = "unified",
    seed: int = 42,
) -> pd.DataFrame:
    """Load and concatenate outer-val predictions from all folds for one seed.

    Reads ``predictions.parquet`` from each fold directory and returns a
    DataFrame with a ``fold`` column added. Useful for cross-fold conformal
    calibration where each fold's outer-val predictions are calibration data
    for every other fold.
    """
    runs_root = pinn_results_dir / "runs" / architecture / preprocessing
    frames: list[pd.DataFrame] = []
    for fold_dir in sorted(runs_root.glob("fold_*")):
        seed_dir = fold_dir / f"seed_{seed}"
        pred_path = seed_dir / "predictions.parquet"
        if not pred_path.exists():
            continue
        df = pd.read_parquet(pred_path)
        outer_val = df[df.get("evaluation_role", pd.Series(["outer_validation"] * len(df)))
                       .eq("outer_validation")] if "evaluation_role" in df.columns else df
        fold_num = int(fold_dir.name.replace("fold_", ""))
        outer_val = outer_val.copy()
        outer_val["fold"] = fold_num
        frames.append(outer_val)
    if not frames:
        raise FileNotFoundError(
            f"no completed fold predictions found under {runs_root} for seed {seed}"
        )
    return pd.concat(frames, ignore_index=True)
