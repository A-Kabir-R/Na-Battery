"""Fold-level trainer for NaPINN-Q.

Implements the composite cell-balanced loss, physics-weight curriculum,
autograd-based governing residual, unlabeled collocation points, checkpointing
and resume, plus per-epoch physics + validation metrics.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .batching import CellBalancedSampler
from .collocation import sample_collocation_points
from .dataset import (
    UNIFIED_PREPROCESSING, AnchorDataset, FoldScaler, fold_indices,
    inner_split_indices,
)
from .logging_utils import (
    log_epoch_summary, log_event, log_failure, log_fold_curriculum_state,
    log_fold_summary,
)
from .losses import (
    BALANCED_COMPONENTS, CurriculumSchedule, GradientBalancer, LossWeights,
    bounds_loss, cell_balanced_reduce, data_loss,
    discrete_state_transition_loss, initial_condition_loss,
    integral_consistency_loss, monotonicity_loss, observed_degradation_rate,
    pairwise_delta_loss, pde_loss, rate_data_loss, rate_regularization_loss,
)
from .degradation_laws import HybridRateModel, parameter_prior_penalty, residual_share_penalty
from .models import NaPINNQ, count_parameters
from .physics import (
    autograd_du_ds, discrete_state_transition_residual, integral_transition,
    pde_residual,
)
from .utils import (
    atomic_write_csv, atomic_write_json, atomic_write_parquet, atomic_write_torch,
    configure_torch_backend, finite_or_raise, gpu_memory_mb, resolve_device,
    run_fingerprint, set_seeds,
)


# Backend flags used to be set here at import time (TF32 on, cudnn.benchmark on)
# while set_seeds() set the opposite (deterministic on, benchmark off) when it
# ran. Whichever executed last silently won, so reproducibility depended on
# import-versus-call order. Both now route through configure_torch_backend(),
# called explicitly per run below.


class FoldTrainingError(RuntimeError):
    """Raised when a fold cannot be completed. Prevents aggregation contamination."""


@dataclass
class TrainerConfig:
    architecture: str
    preprocessing: str
    fold: int
    seed: int
    max_epochs: int = 2000
    min_epochs: int = 200
    early_stopping_patience: int = 200
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 60
    minimum_learning_rate: float = 1.0e-6
    gradient_clip_norm: float = 1.0
    use_amp: bool = False
    physics_float32: bool = True
    #: Publication mode. True disables TF32 and cuDNN autotuning so a run is
    #: reproducible on fixed hardware; False trades that for throughput. The
    #: resolved backend state is recorded in the fold status artifact.
    deterministic: bool = True
    #: Number of cell-grouped inner folds, and which of them this run uses for
    #: early stopping. Running every ``inner_fold`` in ``range(n_inner_folds)``
    #: and selecting the epoch count on the mean inner curve is nested selection;
    #: a single fold is the cheaper, higher-variance approximation.
    n_inner_folds: int = 3
    inner_fold: int = 0
    #: Epoch count for the full-outer-training refit, supplied from outside.
    #: When None the refit uses this run's own ``best_epoch``, i.e. selection on
    #: a single inner split. Nested selection sets it to the argmin of the mean
    #: validation curve across all inner folds -- see
    #: :mod:`src.pinn.nested_selection` and scripts/15. Phase 1 still runs
    #: normally; only the refit length changes.
    forced_best_epoch: int | None = None
    curriculum_warmup_fraction: float = 0.10
    curriculum_ramp_end_fraction: float = 0.30
    huber_delta: float = 1.0
    collocation_points: int = 8
    quadrature_method: str = "trapezoidal"
    quadrature_nodes: int = 8
    solution_hidden_dims: tuple[int, ...] = (16, 16)
    rate_hidden_dims: tuple[int, ...] = (8, 8)
    solution_activation: str = "tanh"
    rate_activation: str = "tanh"
    solution_dropout: float = 0.2
    rate_dropout: float = 0.2
    rate_uses_u_hat: bool = True
    predict_delta_u: bool = True
    # Floor subtracted from the Softplus rate head: r_hat = softplus(.) - rate_floor,
    # so r_hat (and therefore du/ds = -r_hat) can go slightly negative instead of
    # being hard-clamped at exactly zero. 0.0 reproduces the original hard
    # nonnegative-rate behaviour. Non-zero lets the model express the apparent
    # capacity regeneration seen in ~31% of real RPT-to-RPT transitions, which a
    # hard-nonnegative rate head can never fit -- that mismatch is what drives
    # delta_next_rpt_* R2 negative even though the absolute next_rpt_* targets are
    # fine. Only implemented for the plain RateNet (use_hybrid_rate=False).
    rate_floor: float = 0.0
    # Cell-balanced mini-batching. batch_size <= 0 preserves the legacy
    # full-batch behaviour; batch_mode selects the sampler (see batching.py).
    batch_size: int = 32
    batch_mode: str = "cell_balanced"
    maximum_parameters: int = 50000
    inner_split_seed: int = 20240117
    log_every_epochs: int = 1
    save_checkpoint_every_epochs: int = 100
    log_gpu_memory: bool = True
    log_gradient_norms: bool = True
    device: str = "cuda"
    # Two-phase refit: after inner train/val selects best_epoch, discard the
    # temporary model, refit statistics on ALL outer-training cells, and
    # retrain a fresh model for exactly best_epoch+1 epochs without early
    # stopping. Improves comparability with classical models trained on the
    # full outer-training fold.
    two_phase_refit: bool = True
    # PDE-gradient gate — after curriculum warm-up, if the physics loss is
    # active but its gradient is either non-finite or below
    # ``pde_gradient_min_norm`` for ``pde_gradient_zero_patience`` consecutive
    # epochs, raise FoldTrainingError instead of silently reporting a PINN
    # that isn't actually learning physics.
    pde_gradient_min_norm: float = 1.0e-8
    pde_gradient_zero_patience: int = 5
    # Gradient-norm balancing across every active loss component (see
    # losses.GradientBalancer). Multipliers are clipped and logged.
    use_gradient_balance: bool = True
    gradient_balance_ema: float = 0.95
    gradient_balance_min_multiplier: float = 0.1
    gradient_balance_max_multiplier: float = 10.0
    # When True, each balanced component's target multiplier is divided by the
    # number of balanced components active this step (see
    # GradientBalancer.multiplier). Without this, a config with more
    # simultaneously-active physics terms (e.g. the full headline model) sums
    # to a larger worst-case physics-vs-data gradient ratio than a
    # leave-one-out ablation, purely as an artefact of the balancing scheme --
    # not because any individual term's weight changed. False reproduces the
    # original per-component-independent behaviour exactly.
    gradient_balance_normalize_active: bool = False
    # Checkpoint eligibility: when True, no checkpoint may be selected as the
    # fold's best model before the curriculum factor reaches 1.0. Prevents
    # reporting a partially-physics-active model as the full PINN.
    checkpoint_after_physics_active: bool = True
    # Direct supervision of the rate head on the observed degradation rate.
    rate_data_weight: float = 0.0
    # Cross-cell relative-degradation regulariser. 0.0 disables it.
    pairwise_weight: float = 0.0
    pairwise_max_pairs: int = 256
    # Hybrid rate model. When True, NaPINNQ uses HybridRateModel instead of
    # RateNet and the named rate components (cycling/cold/diagnostic/residual)
    # are available for decomposition analysis and the residual-share penalty.
    use_hybrid_rate: bool = False
    hybrid_temperature_feature: str = "T_degC"
    hybrid_dod_feature: str = "DOD_pct"
    hybrid_c_rate_feature: str = ""  # empty → use DOD index; C-rate frozen at 0
    hybrid_enable_cold_regime: bool = True
    hybrid_fit_c_rate_exponent: bool = False
    # Upper bound on the neural residual's share of the total rate. Only
    # applied when use_hybrid_rate=True and residual_share weight > 0.
    hybrid_residual_share_limit: float = 0.5
    # Hidden dims of the small neural correction inside HybridRateModel (the
    # r_residual term). Previously hardcoded in NaPINNQ's signature with no
    # config.yaml entry or wiring here -- same default value, now visible and
    # tunable like every other architecture setting.
    hybrid_residual_hidden_dims: tuple[int, ...] = (8, 8)


@dataclass
class FoldPaths:
    root: Path

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def epoch_log_csv(self) -> Path:
        return self.logs_dir / "epoch_log.csv"

    @property
    def epoch_log_parquet(self) -> Path:
        return self.logs_dir / "epoch_log.parquet"

    @property
    def best_model(self) -> Path:
        return self.checkpoint_dir / "best_model.pt"

    @property
    def last_model(self) -> Path:
        return self.checkpoint_dir / "last_model.pt"

    @property
    def optimizer_state(self) -> Path:
        return self.checkpoint_dir / "optimizer.pt"

    @property
    def scheduler_state(self) -> Path:
        return self.checkpoint_dir / "scheduler.pt"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "experiment_manifest.json"

    @property
    def scaler_path(self) -> Path:
        return self.root / "scaler.json"

    @property
    def final_model(self) -> Path:
        return self.checkpoint_dir / "final_refit_model.pt"

    @property
    def final_scaler(self) -> Path:
        return self.root / "final_refit_scaler.json"

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config_snapshot.json"

    @property
    def audit_subset(self) -> Path:
        return self.root / "temporal_audit_subset.csv"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.parquet"

    @property
    def fingerprint_path(self) -> Path:
        return self.root / "run_fingerprint.json"


def _tensor(values, device: str, dtype=torch.float32,
            requires_grad: bool = False) -> torch.Tensor:
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _cell_index(cell_series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    codes, uniques = pd.factorize(cell_series.astype(str), sort=True)
    mapping = {str(cell): int(i) for i, cell in enumerate(uniques)}
    return codes.astype(np.int64), mapping


def _build_feature_matrix(dataset: AnchorDataset, frame: pd.DataFrame,
                          scaler: FoldScaler) -> np.ndarray:
    scaled_features = scaler.transform_features(frame, dataset.feature_columns)
    horizon = scaler.transform_horizon(frame["stress_delta"].to_numpy(dtype=float))
    u_current = frame["u_current"].to_numpy(dtype=float)
    return np.concatenate(
        [horizon[:, None], u_current[:, None], scaled_features], axis=1
    )


def _pde_gradient_norm(component_loss: torch.Tensor,
                       parameters: list[nn.Parameter],
                       logger: logging.Logger | None = None) -> float:
    """Return ‖∂(PDE component)/∂θ‖₂ via ``torch.autograd.grad`` (issue #7).

    Uses ``retain_graph=True`` so the caller's subsequent ``total.backward()``
    still succeeds. Returns 0.0 only when the component is not differentiable
    w.r.t. the model (e.g. when curriculum has scaled the PDE weight to zero).
    Autograd runtime errors are logged and re-raised — silently swallowing
    them would mask real PINN faults that the enforcement gate must catch.
    """
    if not component_loss.requires_grad:
        return 0.0
    try:
        grads = torch.autograd.grad(
            component_loss,
            tuple(parameters),
            retain_graph=True,
            allow_unused=True,
        )
    except RuntimeError as exc:
        if logger is not None:
            log_event(logger, logging.ERROR, "pde_gradient_autograd_failed",
                      error=str(exc))
        raise
    # Sum squared norms on GPU; single .item() sync at the end. Previously did
    # one .item() per parameter (~15 syncs per batch), which serialized the
    # training step on modest MLPs.
    total_sq: torch.Tensor | None = None
    for g in grads:
        if g is None:
            continue
        piece = g.detach().pow(2).sum()
        total_sq = piece if total_sq is None else total_sq + piece
    if total_sq is None:
        return 0.0
    return float(total_sq.sqrt().item())


def _init_optimizer(model: nn.Module, config: TrainerConfig
                    ) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay)


def _init_scheduler(optimizer: torch.optim.Optimizer, config: TrainerConfig):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.minimum_learning_rate,
    )


def _load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler, paths: FoldPaths) -> dict[str, Any]:
    state: dict[str, Any] = {"next_epoch": 0, "best_val": float("inf"),
                              "best_epoch": 0, "patience": 0}
    if paths.last_model.exists():
        payload = torch.load(paths.last_model, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        for key in list(state.keys()):
            if key in payload:
                state[key] = payload[key]
        # Backward-compat: earlier checkpoints stored ``epoch`` (== last-run
        # epoch); translate to ``next_epoch`` so resume never trains it twice.
        if "next_epoch" not in payload and "epoch" in payload:
            state["next_epoch"] = int(payload["epoch"]) + 1
    if paths.optimizer_state.exists():
        optimizer.load_state_dict(torch.load(paths.optimizer_state, map_location="cpu",
                                             weights_only=False))
    if paths.scheduler_state.exists() and scheduler is not None:
        scheduler.load_state_dict(torch.load(paths.scheduler_state, map_location="cpu",
                                             weights_only=False))
    return state


def _file_sha256(path: Path | None) -> str:
    """Return a short hex hash of a file for fingerprinting; '' if missing."""
    if path is None:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compute_run_fingerprint(*, architecture: str, preprocessing: str,
                              fold: int, seed: int,
                              feature_columns: list[str],
                              trainer_config: TrainerConfig,
                              loss_weights: LossWeights,
                              feature_file: Path | None = None,
                              split_manifest_file: Path | None = None,
                              git_commit_hash: str = "",
                              target_builder_version: str = "") -> str:
    """Fingerprint used for completed-run reuse.

    Includes not just config/columns but also hashes of the feature parquet
    and split manifest — so if the dataset changes under the same column
    names, the fingerprint invalidates and the run is retrained.
    """
    payload = {
        "architecture": architecture,
        "preprocessing": preprocessing,
        "fold": fold,
        "seed": seed,
        "feature_columns": list(feature_columns),
        "trainer_config": asdict(trainer_config),
        "loss_weights": asdict(loss_weights),
        "feature_file_sha256": _file_sha256(feature_file),
        "split_manifest_sha256": _file_sha256(split_manifest_file),
        "git_commit": git_commit_hash,
        "target_builder_version": target_builder_version,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class FoldResult:
    architecture: str
    preprocessing: str
    fold: int
    seed: int
    trainable_parameters: int
    best_epoch: int
    best_validation_mae: float
    predictions_path: Path
    physics_metrics: dict[str, float]
    status: str = "completed"
    error_message: str | None = None
    ablation_name: str | None = None
    epoch_log: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""


def _prediction_frame(architecture: str, preprocessing: str, fold: int, seed: int,
                      evaluation_role: str, indices: np.ndarray,
                      dataset: AnchorDataset, frame: pd.DataFrame,
                      predicted_u_next: np.ndarray, predicted_r_norm: np.ndarray,
                      du_ds_norm: np.ndarray, pde_res_norm: np.ndarray,
                      ic_err: np.ndarray, integral_err: np.ndarray,
                      mono_viol: np.ndarray, lower_viol: np.ndarray,
                      upper_viol: np.ndarray, *,
                      stress_std: float,
                      ablation_name: str | None = None) -> pd.DataFrame:
    sub = frame.iloc[indices].reset_index(drop=True)
    q0 = sub[dataset.q0_column].to_numpy(dtype=float)
    q_current = sub[dataset.q_current_column].to_numpy(dtype=float)
    predicted_q_next = predicted_u_next * q0
    predicted_soh_next = 100.0 * predicted_u_next
    predicted_delta_q = predicted_q_next - q_current
    predicted_delta_soh = predicted_soh_next - (100.0 * q_current / q0)

    stress_std = float(stress_std) if stress_std else 1.0
    du_ds_physical = du_ds_norm / stress_std
    pde_res_physical = pde_res_norm / stress_std
    predicted_r_physical = predicted_r_norm / stress_std

    frame_out = pd.DataFrame({
        "run_id": [run_fingerprint(architecture=architecture,
                                    preprocessing=preprocessing,
                                    target=None,
                                    fold=fold, seed=seed,
                                    extra={"role": evaluation_role,
                                            "ablation": ablation_name or ""})] * len(sub),
        "architecture": architecture,
        "preprocessing": preprocessing,
        "fold": fold,
        "seed": seed,
        "evaluation_role": evaluation_role,
        "ablation": ablation_name or "",
        "cell_id": sub["cell"].astype(str).to_numpy(),
        "anchor_id": sub["anchor_id"].to_numpy() if "anchor_id" in sub.columns else np.full(len(sub), np.nan),
        "condition_id": sub.get("condition", pd.Series([""] * len(sub))).astype(str).to_numpy(),
        "current_visit": sub.get("visit", pd.Series([np.nan] * len(sub))).to_numpy(),
        "target_visit": sub.get("next_rpt_visit", pd.Series([np.nan] * len(sub))).to_numpy(),
        "stress_current": sub["stress_current"].to_numpy(dtype=float),
        "stress_target": sub["stress_next"].to_numpy(dtype=float),
        "stress_increment": sub["stress_delta"].to_numpy(dtype=float),
        "Q_reference_Ah": q0,
        "Q_current_Ah": q_current,
        "SOH_current_pct": 100.0 * q_current / q0,
        "true_next_Q_Ah": sub["next_rpt_Q_Ah"].to_numpy(dtype=float),
        "predicted_next_Q_Ah": predicted_q_next,
        "true_next_SOH_pct": sub["next_rpt_SOH_pct"].to_numpy(dtype=float)
            if "next_rpt_SOH_pct" in sub.columns else np.full(len(sub), np.nan),
        "predicted_next_SOH_pct": predicted_soh_next,
        "true_delta_Q_Ah": sub.get("delta_next_rpt_Q_Ah", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "predicted_delta_Q_Ah": predicted_delta_q,
        "true_delta_SOH_pct": sub.get("delta_next_rpt_SOH_pct", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "predicted_delta_SOH_pct": predicted_delta_soh,
        "normalized_capacity_true": sub["u_true_next"].to_numpy(dtype=float),
        "normalized_capacity_predicted": predicted_u_next,
        # Rate: both coordinates.
        "predicted_degradation_rate_normalized": predicted_r_norm,
        "predicted_degradation_rate_physical": predicted_r_physical,
        # Derivative: both coordinates.
        "du_dstress_normalized": du_ds_norm,
        "du_dstress_physical": du_ds_physical,
        # PDE residual: both coordinates.
        "pde_residual_normalized": pde_res_norm,
        "pde_residual_physical": pde_res_physical,
        "initial_condition_error": ic_err,
        "integral_consistency_error": integral_err,
        "monotonicity_violation": mono_viol,
        "lower_bound_violation": lower_viol,
        "upper_bound_violation": upper_viol,
        "horizon_days": sub.get("next_rpt_horizon_days",
                                 pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "EFC": sub.get("EFC_cum", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "Ah_throughput": sub.get("cumulative_Ah_throughput",
                                  pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "temperature": sub.get("T_mean", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "DOD": sub.get("DOD_pct", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "charge_C_rate": sub.get("C_ch", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
        "discharge_C_rate": sub.get("C_dis", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float),
    })
    return frame_out


def _evaluate(model, dataset: AnchorDataset, frame: pd.DataFrame,
               indices: np.ndarray, scaler: FoldScaler, device: str,
               config: TrainerConfig, quadrature_method: str,
               quadrature_nodes: int) -> dict[str, np.ndarray]:
    """Compute predictions + physics diagnostics for one split."""
    if indices.size == 0:
        return {}
    frame_sub = frame.iloc[indices].reset_index(drop=True)
    features_np = _build_feature_matrix(dataset, frame_sub, scaler)
    stress_next_np = scaler.transform_stress(frame_sub["stress_next"].to_numpy(dtype=float))
    stress_current_np = scaler.transform_stress(frame_sub["stress_current"].to_numpy(dtype=float))
    delta_np = scaler.transform_horizon(frame_sub["stress_delta"].to_numpy(dtype=float))
    u_current_np = frame_sub["u_current"].to_numpy(dtype=float)
    u_true_np = frame_sub["u_true_next"].to_numpy(dtype=float)

    features = _tensor(features_np, device)
    stress_next = _tensor(stress_next_np, device, requires_grad=True)
    stress_current = _tensor(stress_current_np, device, requires_grad=True)
    delta = _tensor(delta_np, device)
    u_current = _tensor(u_current_np, device)

    model.eval()
    with torch.enable_grad():
        u_next = model.solution(stress_next, features, u_current, stress_current)
        r_next = model.rate(stress_next, features, u_next)
        u_anchor = model.solution(
            stress_current, features, u_current, stress_current
        )
        du_next = autograd_du_ds(u_next, stress_next)
        residual = pde_residual(du_next, r_next)
        u_integrated = integral_transition(
            model.solution, model.rate, stress_current, delta, features,
            u_current, method=quadrature_method, n_nodes=quadrature_nodes,
        )

    u_next_np = u_next.detach().cpu().numpy()
    r_next_np = r_next.detach().cpu().numpy()
    r_anchor_np = model.rate(
        stress_current, features, u_anchor
    ).detach().cpu().numpy()
    du_ds_np = du_next.detach().cpu().numpy()
    residual_np = residual.detach().cpu().numpy()
    ic_err_np = (u_anchor.detach().cpu().numpy() - u_current_np)
    integral_err_np = (u_next_np - u_integrated.detach().cpu().numpy())
    mono_viol_np = np.maximum(u_next_np - u_current_np - scaler.epsilon_rec, 0.0)
    lower_viol_np = np.maximum(0.0 - u_next_np, 0.0)
    upper_viol_np = np.maximum(u_next_np - scaler.u_max, 0.0)

    # Observed degradation rate for rate-head diagnostics, in the same
    # normalized-stress coordinate the model is trained in.
    safe_delta = np.maximum(delta_np, 1.0e-6)
    observed_rate_np = np.maximum((u_current_np - u_true_np) / safe_delta, 0.0)
    rate_valid = (
        (delta_np > 1.0e-6)
        & np.isfinite(observed_rate_np)
        & np.isfinite(r_anchor_np)
    )

    return {
        "u_next": u_next_np,
        "r_next": r_next_np,
        "r_anchor": r_anchor_np,
        "du_ds": du_ds_np,
        "pde_residual": residual_np,
        "ic_error": ic_err_np,
        "integral_error": integral_err_np,
        "mono_violation": mono_viol_np,
        "lower_violation": lower_viol_np,
        "upper_violation": upper_viol_np,
        "u_true": u_true_np,
        "observed_rate": observed_rate_np,
        "rate_valid": rate_valid,
        "stress_delta": delta_np,
    }


def _try_reuse_completed_run(paths: FoldPaths, fingerprint: str,
                              architecture: str, preprocessing: str,
                              fold: int, seed: int,
                              ablation_name: str | None) -> FoldResult | None:
    """Return an existing FoldResult only when every artifact is present and
    the fingerprint matches. Otherwise return None to force retraining.
    """
    if not paths.status_path.exists():
        return None
    try:
        status = json.loads(paths.status_path.read_text())
    except Exception:
        return None
    if status.get("status") != "completed":
        return None
    if not paths.best_model.exists() or not paths.predictions_path.exists():
        return None
    if not paths.fingerprint_path.exists():
        return None
    try:
        recorded = json.loads(paths.fingerprint_path.read_text()).get("fingerprint")
    except Exception:
        return None
    if recorded != fingerprint:
        return None
    return FoldResult(
        architecture=architecture, preprocessing=preprocessing,
        fold=fold, seed=seed,
        trainable_parameters=int(status.get("trainable_parameters", 0)),
        best_epoch=int(status.get("best_epoch", -1)),
        best_validation_mae=float(status.get("best_validation_mae", np.nan)),
        predictions_path=paths.predictions_path,
        physics_metrics=status.get("physics_metrics", {}),
        status="reused", ablation_name=ablation_name,
        fingerprint=fingerprint,
    )


def train_fold(*, dataset: AnchorDataset, frame: pd.DataFrame,
               config: TrainerConfig, fold_paths: FoldPaths,
               loss_weights: LossWeights, u_max_source: str,
               u_max_constant: float, u_max_margin: float,
               tolerance_source: str, tolerance_constant: float,
               tolerance_quantile: float,
               resume: bool = False,
               force: bool = False,
               ablation_name: str | None = None,
               include_discrete_transition: bool = False,
               dry_run: bool = False,
               feature_file: Path | None = None,
               split_manifest_file: Path | None = None,
               git_commit_hash: str = "",
               target_builder_version: str = "",
               logger: logging.Logger | None = None) -> FoldResult:
    """Train one fold + seed for the requested architecture.

    Raises :class:`FoldTrainingError` on any training failure so that
    aggregation cannot consume partial predictions from a failed run.
    """
    fold_paths.root.mkdir(parents=True, exist_ok=True)
    fold_paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_paths.logs_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if config.device == "cuda" and device != "cuda":
        msg = ("[trainer] WARNING: cuda requested but torch.cuda.is_available() is False; "
               "training will run on CPU. Check your PyTorch installation and CUDA drivers.")
        print(msg)
        if logger is not None:
            log_event(logger, logging.WARNING, "cuda_unavailable_cpu_fallback",
                      requested=config.device, actual=device)

    # AMP is configured but not implemented — training runs in fp32. Warn
    # loudly if the caller still asks for AMP so downstream metrics aren't
    # attributed to a mixed-precision run that never happened.
    if config.use_amp and logger is not None:
        log_event(logger, logging.WARNING, "amp_not_implemented",
                  message="use_amp=True was requested but AMP is not implemented; "
                          "training will run in fp32.")

    # Determine fingerprint early so completed-run reuse can validate it.
    fingerprint = _compute_run_fingerprint(
        architecture=config.architecture, preprocessing=config.preprocessing,
        fold=config.fold, seed=config.seed,
        feature_columns=dataset.feature_columns,
        trainer_config=config, loss_weights=loss_weights,
        feature_file=feature_file, split_manifest_file=split_manifest_file,
        git_commit_hash=git_commit_hash,
        target_builder_version=target_builder_version,
    )
    if not force:
        reused = _try_reuse_completed_run(
            fold_paths, fingerprint,
            architecture=config.architecture, preprocessing=config.preprocessing,
            fold=config.fold, seed=config.seed, ablation_name=ablation_name,
        )
        if reused is not None:
            return reused

    set_seeds(config.seed)
    backend_state = configure_torch_backend(deterministic=config.deterministic)

    outer_train_idx, outer_val_idx = fold_indices(frame, config.fold)
    outer_train_frame = frame.iloc[outer_train_idx].reset_index(drop=True)
    inner_train_local, inner_val_local = inner_split_indices(
        outer_train_frame, n_inner=config.n_inner_folds,
        seed=config.inner_split_seed, inner_fold=config.inner_fold,
    )
    inner_train_idx = outer_train_idx[inner_train_local]
    inner_val_idx = outer_train_idx[inner_val_local]

    # Fit scaler on INNER TRAIN ONLY (issue #12).
    inner_train_frame = frame.iloc[inner_train_idx].reset_index(drop=True)
    use_unified = config.preprocessing == UNIFIED_PREPROCESSING
    scaler = FoldScaler().fit(
        inner_train_frame, dataset.feature_columns,
        u_max_source=u_max_source, u_max_constant=u_max_constant,
        u_max_margin=u_max_margin, tolerance_source=tolerance_source,
        tolerance_constant=tolerance_constant, tolerance_quantile=tolerance_quantile,
        use_unified=use_unified,
    )

    if logger is not None:
        log_fold_summary(
            logger,
            architecture=config.architecture, preprocessing=config.preprocessing,
            fold=config.fold, seed=config.seed,
            outer_train_cells=int(outer_train_frame["cell"].nunique()),
            outer_val_cells=int(frame.iloc[outer_val_idx]["cell"].nunique()),
            inner_train_cells=int(frame.iloc[inner_train_idx]["cell"].nunique()),
            inner_val_cells=int(frame.iloc[inner_val_idx]["cell"].nunique()),
            n_rows_train=int(inner_train_idx.size),
            n_rows_val=int(inner_val_idx.size),
            trainable_parameters=0,
            u_max=float(scaler.u_max),
            epsilon_rec=float(scaler.epsilon_rec),
            device=device,
        )

    if config.architecture != "NaPINN-Q":
        raise FoldTrainingError(
            f"unsupported architecture: {config.architecture} (only NaPINN-Q is supported)"
        )
    feature_dim = 2 + len(scaler.feature_columns_used or [])

    # Resolve hybrid-rate feature indices if requested. The feature matrix is
    # [horizon, u_current, *scaled_features], so each registry column sits at
    # index 2 + position_in_feature_columns_used.
    hybrid_temp_idx: int | None = None
    hybrid_dod_idx: int | None = None
    hybrid_crate_idx: int | None = None
    if config.use_hybrid_rate:
        cols = list(scaler.feature_columns_used or [])
        if config.hybrid_temperature_feature not in cols:
            raise FoldTrainingError(
                f"Hybrid rate: temperature feature "
                f"'{config.hybrid_temperature_feature}' not found in feature "
                f"columns {cols[:10]}…"
            )
        if config.hybrid_dod_feature not in cols:
            raise FoldTrainingError(
                f"Hybrid rate: DOD feature '{config.hybrid_dod_feature}' not "
                f"found in feature columns {cols[:10]}…"
            )
        hybrid_temp_idx = 2 + cols.index(config.hybrid_temperature_feature)
        hybrid_dod_idx = 2 + cols.index(config.hybrid_dod_feature)
        if config.hybrid_c_rate_feature and config.hybrid_c_rate_feature in cols:
            hybrid_crate_idx = 2 + cols.index(config.hybrid_c_rate_feature)
        else:
            hybrid_crate_idx = hybrid_dod_idx

    model = NaPINNQ(
        feature_dim=feature_dim,
        solution_hidden_dims=config.solution_hidden_dims,
        rate_hidden_dims=config.rate_hidden_dims,
        solution_activation=config.solution_activation,
        rate_activation=config.rate_activation,
        solution_dropout=config.solution_dropout,
        rate_dropout=config.rate_dropout,
        rate_uses_u_hat=config.rate_uses_u_hat,
        predict_delta_u=config.predict_delta_u,
        rate_floor=config.rate_floor,
        use_hybrid_rate=config.use_hybrid_rate,
        hybrid_temperature_index=hybrid_temp_idx,
        hybrid_dod_index=hybrid_dod_idx,
        hybrid_c_rate_index=hybrid_crate_idx,
        hybrid_enable_cold_regime=config.hybrid_enable_cold_regime,
        hybrid_fit_c_rate_exponent=config.hybrid_fit_c_rate_exponent,
        hybrid_residual_hidden_dims=config.hybrid_residual_hidden_dims,
    )

    trainable_parameters = count_parameters(model)
    if trainable_parameters > config.maximum_parameters:
        raise FoldTrainingError(
            f"model has {trainable_parameters} params exceeding the cap "
            f"{config.maximum_parameters}"
        )
    model.to(device)
    if logger is not None:
        log_event(logger, logging.INFO, "model_built",
                  architecture=config.architecture,
                  preprocessing=config.preprocessing,
                  fold=config.fold, seed=config.seed,
                  trainable_parameters=trainable_parameters,
                  feature_dim=feature_dim,
                  device=device)

    optimizer = _init_optimizer(model, config)
    scheduler = _init_scheduler(optimizer, config)
    checkpoint_state = _load_checkpoint(model, optimizer, scheduler, fold_paths) if resume else {
        "next_epoch": 0, "best_val": float("inf"), "best_epoch": 0, "patience": 0
    }
    curriculum = CurriculumSchedule(
        max_epochs=config.max_epochs,
        warmup_fraction=config.curriculum_warmup_fraction,
        ramp_end_fraction=config.curriculum_ramp_end_fraction,
    )

    # Build training tensors on the outer-training rows (inner train/val are
    # boolean sub-selections).
    training_frame = frame.iloc[outer_train_idx].reset_index(drop=True)
    features_np = _build_feature_matrix(dataset, training_frame, scaler)
    stress_current_np = scaler.transform_stress(training_frame["stress_current"].to_numpy(dtype=float))
    stress_next_np = scaler.transform_stress(training_frame["stress_next"].to_numpy(dtype=float))
    delta_np = scaler.transform_horizon(training_frame["stress_delta"].to_numpy(dtype=float))
    u_current_np = training_frame["u_current"].to_numpy(dtype=float)
    u_true_np = training_frame["u_true_next"].to_numpy(dtype=float)
    cell_codes, cell_map = _cell_index(training_frame["cell"])

    features = _tensor(features_np, device)
    delta = _tensor(delta_np, device)
    u_current = _tensor(u_current_np, device)
    u_true = _tensor(u_true_np, device)
    cell_index = torch.as_tensor(cell_codes, device=device, dtype=torch.long)
    # Cache stress tensors on device once — used both as autograd leaves in
    # each batch (via detach + requires_grad_) and as read-only inputs to the
    # validation forward pass. Previously the batch loop re-uploaded these
    # from numpy on every iteration.
    stress_current_dev = _tensor(stress_current_np, device)
    stress_next_dev = _tensor(stress_next_np, device)

    inner_train_mask = np.isin(outer_train_idx, inner_train_idx)
    inner_val_mask = np.isin(outer_train_idx, inner_val_idx)
    train_selector = torch.as_tensor(inner_train_mask, device=device, dtype=torch.bool)
    # Precompute val-side selectors/targets once. Previously the epoch loop
    # rebuilt ``val_selector`` and ``val_stress_next`` from numpy every epoch
    # (two H2D copies per epoch) and re-sliced ``val_true_tensor`` twice for
    # R² statistics.
    val_selector = torch.as_tensor(inner_val_mask, device=device, dtype=torch.bool)
    val_true_tensor_cached = u_true[val_selector]
    val_n = int(val_true_tensor_cached.numel())
    if val_n:
        val_true_mean_cached = val_true_tensor_cached.mean()
        val_ss_tot_cached = float(
            ((val_true_tensor_cached - val_true_mean_cached) ** 2).sum().item()
        )
    else:
        val_ss_tot_cached = 0.0

    epoch_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    manifest = {
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "fold": config.fold,
        "seed": config.seed,
        "device": device,
        "feature_dim": feature_dim,
        "feature_columns": scaler.feature_columns_used,
        "dropped_all_nan_feature_columns": scaler.dropped_all_nan_columns,
        "cell_map": cell_map,
        "trainable_parameters": trainable_parameters,
        "loss_weights": asdict(loss_weights),
        "ablation": ablation_name or "",
        "trainer_config": asdict(config),
        "fingerprint": fingerprint,
    }
    atomic_write_json(manifest, fold_paths.manifest_path)
    atomic_write_json(scaler.state_dict(), fold_paths.scaler_path)
    dataset.audit.to_csv(fold_paths.audit_subset, index=False)
    atomic_write_json({"seed": config.seed, "architecture": config.architecture,
                       "preprocessing": config.preprocessing}, fold_paths.config_snapshot)
    atomic_write_json({"fingerprint": fingerprint}, fold_paths.fingerprint_path)

    # Clear stale status while training so a crash doesn't leave a claim of
    # "completed" behind (issue #8, defensive).
    if fold_paths.status_path.exists():
        try:
            fold_paths.status_path.unlink()
        except OSError:
            pass

    if dry_run:
        return FoldResult(
            architecture=config.architecture, preprocessing=config.preprocessing,
            fold=config.fold, seed=config.seed,
            trainable_parameters=trainable_parameters,
            best_epoch=-1, best_validation_mae=float("nan"),
            predictions_path=fold_paths.predictions_path,
            physics_metrics={}, status="dry_run", ablation_name=ablation_name,
            fingerprint=fingerprint,
        )

    best_val = float(checkpoint_state.get("best_val", float("inf")))
    best_epoch = int(checkpoint_state.get("best_epoch", 0))
    patience = int(checkpoint_state.get("patience", 0))
    start_epoch = int(checkpoint_state.get("next_epoch", 0))
    consecutive_zero_pde_gradients = 0
    warmup_epochs = curriculum.warmup_epochs
    # First epoch at which the curriculum factor is exactly 1.0. Checkpoint
    # selection and early-stopping patience both start here, so a fold can
    # never report a partially-physics-active model as its best full PINN.
    physics_full_epoch = curriculum.physics_full_epoch
    checkpoint_eligibility_epoch = (
        physics_full_epoch if config.checkpoint_after_physics_active else 0
    )
    balancer = GradientBalancer(
        ema=config.gradient_balance_ema,
        min_multiplier=config.gradient_balance_min_multiplier,
        max_multiplier=config.gradient_balance_max_multiplier,
    )

    tag = f"[{config.architecture} | {config.preprocessing} | fold {config.fold} | seed {config.seed}]"
    bar = tqdm(range(start_epoch, config.max_epochs), desc=tag, unit="ep",
               initial=start_epoch, total=config.max_epochs, leave=False)

    # Cell-balanced mini-batching over the inner-training rows. ``batch_mode``
    # is now honoured at runtime (it was previously dead configuration), and the
    # whole forward pass is restricted to the sampled rows rather than being
    # computed over the full frame and then indexed.
    train_row_positions = torch.nonzero(train_selector, as_tuple=False).view(-1)
    n_train_rows = int(train_row_positions.numel())
    sampler = CellBalancedSampler(
        row_positions=train_row_positions,
        cell_index=cell_index,
        batch_size=int(config.batch_size),
        mode=str(config.batch_mode),
    )
    n_batches = sampler.n_batches()

    delta_full = torch.as_tensor(delta_np, dtype=torch.float32, device=device)

    last_epoch_seen = start_epoch - 1
    try:
        for epoch in bar:
            model.train()
            last_epoch_seen = epoch
            factor = curriculum.factor(epoch)
            effective_weights = loss_weights.effective(factor)

            # Resample this epoch's batches. Cell-balanced mode draws cells
            # uniformly so high-anchor-count cells cannot dominate, and keeps
            # many distinct cells per batch for the pairwise loss.
            epoch_batches = sampler.epoch_batches(generator)

            # Accumulators for epoch-level logging (weighted by batch rows).
            # ``acc_total`` and ``grad_total_epoch`` stay as GPU tensors so we
            # only pay a single CPU-GPU sync per epoch (at the end) instead of
            # one per batch. Row-mean scalars from the loss module come back
            # as Python floats already, so those accumulators can stay float.
            acc_total = torch.zeros((), device=device)
            acc_data = 0.0; acc_pde = 0.0
            acc_pde_a = 0.0; acc_pde_c = 0.0
            acc_ic = 0.0; acc_int = 0.0
            acc_mono = 0.0; acc_bounds = 0.0
            acc_rate = 0.0; acc_discrete = 0.0
            acc_rate_data = 0.0; acc_pairwise = 0.0
            rate_coverage_epoch = 0.0
            pair_count_epoch = 0
            n_rows_seen = 0
            grad_pde_epoch = 0.0
            grad_total_epoch = torch.zeros((), device=device)
            # Physics diagnostics accumulated over the whole epoch, exactly like
            # the losses above. They used to be read off the final mini-batch's
            # leftover tensors after the loop, which made them a ~32-row sample
            # taken under dropout while every neighbouring column in the same log
            # row was a batch-weighted epoch mean. Kept on-device; synced once.
            diag_res_abs = torch.zeros((), device=device)
            diag_res_sq = torch.zeros((), device=device)
            diag_res_n = 0
            diag_pos_deriv = torch.zeros((), device=device)
            diag_deriv_n = 0
            diag_mono_viol = torch.zeros((), device=device)
            diag_lower_viol = torch.zeros((), device=device)
            diag_upper_viol = torch.zeros((), device=device)
            diag_state_n = 0

            for b, batch_positions in enumerate(epoch_batches):
                if batch_positions.numel() == 0:
                    continue

                # The forward pass is restricted to the sampled rows. Previously
                # it ran over the whole outer-training frame and only the losses
                # were indexed, so every "mini-batch" step paid a full-frame
                # forward and physics collocation was generated for rows that
                # took no part in the update.
                optimizer.zero_grad(set_to_none=True)

                batch_features = features[batch_positions]
                batch_u_current = u_current[batch_positions]
                batch_u_true = u_true[batch_positions]
                batch_cells = cell_index[batch_positions]
                batch_delta = delta_full[batch_positions]

                # Fresh autograd leaves for the batch's stress coordinates.
                stress_current_tensor = (
                    stress_current_dev[batch_positions].detach().requires_grad_(True)
                )
                stress_next_tensor = (
                    stress_next_dev[batch_positions].detach().requires_grad_(True)
                )

                u_next_hat = model.solution(
                    stress_next_tensor, batch_features, batch_u_current,
                    stress_current_tensor,
                )
                r_next_hat = model.rate(stress_next_tensor, batch_features, u_next_hat)
                # The anchor coordinate must be detached. Passing the live leaf as
                # both the evaluation point and the IC anchor makes
                # u = u_c + (s - s_c) f(s) differentiate to f*(1-1) + 0*df/ds == 0,
                # so the anchor residual would collapse to mean(r^2). Detaching
                # leaves the IC value untouched (local_stress is still 0) while
                # recovering du/ds = f_theta(s_c). Mirrors the collocation path below.
                u_anchor_hat = model.solution(
                    stress_current_tensor, batch_features, batch_u_current,
                    stress_current_tensor.detach(),
                )
                r_anchor_hat = model.rate(
                    stress_current_tensor, batch_features, u_anchor_hat
                )
                du_next = autograd_du_ds(u_next_hat, stress_next_tensor)
                du_current = autograd_du_ds(u_anchor_hat, stress_current_tensor)
                residual_anchor = pde_residual(du_current, r_anchor_hat)
                residual_next = pde_residual(du_next, r_next_hat)
                collocation = sample_collocation_points(
                    stress_current=stress_current_tensor.detach(),
                    stress_delta=batch_delta,
                    features=batch_features,
                    cell_index=batch_cells,
                    points_per_transition=config.collocation_points,
                    generator=generator,
                )
                collo_stress = collocation.stress.clone().detach().requires_grad_(True)
                collo_u_current = batch_u_current[collocation.row_index]
                collo_stress_current = (
                    stress_current_tensor.detach()[collocation.row_index]
                )
                u_collo = model.solution(
                    collo_stress, collocation.features, collo_u_current,
                    collo_stress_current,
                )
                r_collo = model.rate(collo_stress, collocation.features, u_collo)
                du_collo = autograd_du_ds(u_collo, collo_stress)
                residual_collo = pde_residual(du_collo, r_collo)
                u_integrated = integral_transition(
                    model.solution, model.rate,
                    stress_current_tensor, batch_delta,
                    batch_features, batch_u_current,
                    method=config.quadrature_method, n_nodes=config.quadrature_nodes,
                )

                train_data_loss, data_row = data_loss(
                    u_next_hat, batch_u_true, batch_cells, delta=config.huber_delta,
                )

                pde_anchor_loss, pde_anchor_row = pde_loss(residual_anchor, batch_cells)
                pde_collo_loss = residual_collo.new_zeros(())
                pde_collo_row = 0.0
                if residual_collo.numel():
                    pde_collo_loss, pde_collo_row = pde_loss(
                        residual_collo, collocation.cell_index,
                    )
                pde_row_mean = 0.5 * (pde_anchor_row + pde_collo_row)
                # With the hard initial condition this residual is identically
                # zero; it is still computed and logged as a runtime assertion
                # that the parameterisation is in force.
                ic_loss, ic_row = initial_condition_loss(
                    u_anchor_hat, batch_u_current, batch_cells,
                )
                integral_loss, integral_row = integral_consistency_loss(
                    u_next_hat, u_integrated, batch_cells,
                )
                mono_loss_val, mono_row = monotonicity_loss(
                    u_next_hat, batch_u_current, batch_cells,
                    epsilon_rec=scaler.epsilon_rec,
                )
                bounds_val, bounds_row = bounds_loss(
                    u_next_hat, batch_cells, u_min=0.0, u_max=scaler.u_max,
                )
                rate_val, rate_row = rate_regularization_loss(r_next_hat, batch_cells)

                # Direct rate supervision: the rate head otherwise only learns
                # through the PDE coupling.
                rate_data_val = u_next_hat.new_zeros(())
                rate_data_row = float("nan")
                rate_coverage = 0.0
                if effective_weights.rate_data > 0.0:
                    observed_rate, rate_mask = observed_degradation_rate(
                        batch_u_current, batch_u_true, batch_delta,
                        epsilon=1.0e-6,
                        regeneration_tolerance=scaler.epsilon_rec,
                        rate_floor=config.rate_floor,
                    )
                    rate_data_val, rate_data_row, rate_coverage = rate_data_loss(
                        r_anchor_hat, observed_rate, batch_cells, rate_mask,
                        delta=config.huber_delta,
                    )

                # Optional cross-cell relative-degradation regulariser.
                pairwise_val = u_next_hat.new_zeros(())
                pairwise_row = float("nan")
                n_pairs = 0
                if effective_weights.pairwise > 0.0:
                    pairwise_val, pairwise_row, n_pairs = pairwise_delta_loss(
                        u_next_hat, batch_u_current, batch_u_true, batch_cells,
                        generator=generator,
                        max_pairs=config.pairwise_max_pairs,
                        delta=config.huber_delta,
                    )

                discrete_val = u_next_hat.new_zeros(())
                discrete_row = 0.0
                if include_discrete_transition:
                    discrete_residual = discrete_state_transition_residual(
                        u_next_hat, batch_u_current, r_anchor_hat, batch_delta,
                    )
                    discrete_val, discrete_row = discrete_state_transition_loss(
                        discrete_residual, batch_cells,
                    )

                pde_anchor_and_collo = pde_anchor_loss + pde_collo_loss
                physics_active = effective_weights.pde > 0

                # ---- gradient-norm balancing across all active components.
                # Probe each component's gradient norm on the live graph before
                # the composite backward, feed the EMA, and read back a clipped
                # multiplier. ``data`` is the reference and is never rescaled.
                parameters = list(model.parameters())
                grad_pde = 0.0
                multipliers = {name: 1.0 for name in BALANCED_COMPONENTS}
                if physics_active:
                    grad_pde = _pde_gradient_norm(
                        effective_weights.pde * pde_anchor_and_collo,
                        parameters, logger=logger,
                    )
                    if epoch >= warmup_epochs and not np.isfinite(grad_pde):
                        raise FoldTrainingError(
                            f"non-finite PDE gradient at epoch {epoch}: {grad_pde}"
                        )

                if config.use_gradient_balance and epoch >= warmup_epochs:
                    probes: dict[str, torch.Tensor] = {
                        "data": effective_weights.data * train_data_loss,
                        "integral": effective_weights.integral * integral_loss,
                        "monotonicity": effective_weights.monotonicity * mono_loss_val,
                        "bounds": effective_weights.bounds * bounds_val,
                        "rate_data": effective_weights.rate_data * rate_data_val,
                        "pairwise": effective_weights.pairwise * pairwise_val,
                    }
                    balancer.observe("data", _pde_gradient_norm(
                        probes["data"], parameters, logger=None,
                    ))
                    if physics_active and grad_pde > 0:
                        balancer.observe("pde", grad_pde)
                    active_components = set()
                    for name in ("integral", "monotonicity", "bounds",
                                 "rate_data", "pairwise"):
                        weight = getattr(effective_weights, name)
                        if weight <= 0.0:
                            continue
                        active_components.add(name)
                        balancer.observe(name, _pde_gradient_norm(
                            probes[name], parameters, logger=None,
                        ))
                    if physics_active and grad_pde > 0:
                        active_components.add("pde")
                    n_active = (
                        len(active_components)
                        if config.gradient_balance_normalize_active
                        else 1
                    )
                    multipliers = {
                        name: balancer.multiplier(name, n_active=n_active)
                        for name in BALANCED_COMPONENTS
                    }

                total_loss = (
                    effective_weights.data * train_data_loss
                    + effective_weights.ode * multipliers["pde"] * pde_anchor_and_collo
                    + effective_weights.initial_condition * ic_loss
                    + effective_weights.integral * multipliers["integral"] * integral_loss
                    + effective_weights.monotonicity * multipliers["monotonicity"] * mono_loss_val
                    + effective_weights.bounds * multipliers["bounds"] * bounds_val
                    + effective_weights.rate_regularization * rate_val
                    + effective_weights.rate_data * multipliers["rate_data"] * rate_data_val
                    + effective_weights.pairwise * multipliers["pairwise"] * pairwise_val
                    + effective_weights.discrete_state_transition * discrete_val
                )
                if (effective_weights.residual_share > 0.0
                        and isinstance(model.rate, HybridRateModel)):
                    comps = model.rate.components(
                        stress_next_tensor, batch_features, u_next_hat.detach()
                    )
                    rshare_pen = residual_share_penalty(
                        comps, limit=config.hybrid_residual_share_limit,
                    )
                    total_loss = total_loss + effective_weights.residual_share * rshare_pen
                    if config.use_hybrid_rate:
                        total_loss = total_loss + parameter_prior_penalty(model.rate)

                if not torch.isfinite(total_loss):
                    raise FoldTrainingError(
                        f"non-finite total loss at epoch {epoch} batch {b}: {total_loss.item()}"
                    )

                total_loss.backward()
                # ``clip_grad_norm_`` already returns the pre-clip total gradient
                # norm as a GPU scalar tensor. Reusing it drops the previous
                # per-parameter ``.pow(2).sum().item()`` sweep (one CUDA sync per
                # parameter — the largest single sync source in the training
                # step).
                grad_total_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm,
                )
                optimizer.step()

                # Batch-weighted accumulators (weight by batch row count).
                bw = float(batch_positions.numel())
                acc_total.add_(total_loss.detach() * bw)
                acc_data += float(data_row) * bw
                acc_pde += float(pde_row_mean) * bw
                acc_pde_a += float(pde_anchor_row) * bw
                acc_pde_c += float(pde_collo_row) * bw
                acc_ic += float(ic_row) * bw
                acc_int += float(integral_row) * bw
                acc_mono += float(mono_row) * bw
                acc_bounds += float(bounds_row) * bw
                acc_rate += float(rate_row) * bw
                acc_discrete += float(discrete_row) * bw
                if np.isfinite(rate_data_row):
                    acc_rate_data += float(rate_data_row) * bw
                if np.isfinite(pairwise_row):
                    acc_pairwise += float(pairwise_row) * bw
                rate_coverage_epoch = max(rate_coverage_epoch, float(rate_coverage))
                pair_count_epoch += int(n_pairs)

                # Physics diagnostics over every row of the epoch. Sums stay on
                # device; the single CPU sync happens after the loop.
                with torch.no_grad():
                    if residual_next.numel():
                        res_d = residual_next.detach()
                        diag_res_abs += res_d.abs().sum()
                        diag_res_sq += (res_d * res_d).sum()
                        diag_res_n += res_d.numel()
                    if du_next.numel():
                        diag_pos_deriv += (du_next.detach() > 0).sum()
                        diag_deriv_n += du_next.numel()
                    if u_next_hat.numel():
                        u_d = u_next_hat.detach()
                        diag_mono_viol += (
                            (u_d - batch_u_current - scaler.epsilon_rec) > 0
                        ).sum()
                        diag_lower_viol += (u_d < 0).sum()
                        diag_upper_viol += (u_d > scaler.u_max).sum()
                        diag_state_n += u_d.numel()
                grad_pde_epoch = max(grad_pde_epoch, grad_pde)
                # Keep grad_total on-device; take the running max as a tensor
                # op and sync only at epoch end.
                grad_total_epoch = torch.maximum(
                    grad_total_epoch, grad_total_tensor.detach(),
                )
                n_rows_seen += int(bw)

            # ---- end of batch loop; PDE-gradient enforcement is epoch-level.
            physics_active = effective_weights.pde > 0
            if physics_active and epoch >= warmup_epochs:
                if grad_pde_epoch <= config.pde_gradient_min_norm:
                    consecutive_zero_pde_gradients += 1
                else:
                    consecutive_zero_pde_gradients = 0
                if consecutive_zero_pde_gradients >= config.pde_gradient_zero_patience:
                    raise FoldTrainingError(
                        f"PDE loss produced negligible gradient (<= "
                        f"{config.pde_gradient_min_norm}) for "
                        f"{consecutive_zero_pde_gradients} consecutive "
                        f"physics-active epochs (last epoch={epoch})"
                    )
            else:
                consecutive_zero_pde_gradients = 0

            # Reduce accumulators to per-row means for logging. The two
            # remaining GPU tensors are synced once per epoch here.
            denom = max(n_rows_seen, 1)
            data_row = acc_data / denom
            pde_row_mean = acc_pde / denom
            pde_anchor_row = acc_pde_a / denom
            pde_collo_row = acc_pde_c / denom
            ic_row = acc_ic / denom
            integral_row = acc_int / denom
            mono_row = acc_mono / denom
            bounds_row = acc_bounds / denom
            rate_row = acc_rate / denom
            discrete_row = acc_discrete / denom
            rate_data_row = acc_rate_data / denom
            pairwise_row = acc_pairwise / denom
            grad_pde = grad_pde_epoch
            grad_total = float(grad_total_epoch.item())
            total_loss_scalar = float((acc_total / denom).item())

            # Validation on inner-validation rows. Switch model to eval() so
            # dropout is disabled; then restore train() for the next epoch.
            # Reuse cached ``stress_next_dev`` (no per-epoch H2D transfer) and
            # cached ``val_true_tensor_cached`` / ``val_ss_tot_cached``.
            model.eval()
            with torch.no_grad():
                val_u = model.solution(
                    stress_next_dev, features, u_current, stress_current_dev,
                )
                val_pred = val_u[val_selector]
                if val_n:
                    val_error_t = val_pred - val_true_tensor_cached
                    val_mae_t = val_error_t.abs().mean()
                    val_ss_res_t = (val_error_t * val_error_t).sum()
                    val_rmse_t = (val_ss_res_t / val_n).sqrt()
                    # One CPU sync for all three stats.
                    val_stats = torch.stack(
                        (val_mae_t, val_rmse_t, val_ss_res_t)
                    ).detach().cpu().numpy()
                    val_mae = float(val_stats[0])
                    val_rmse = float(val_stats[1])
                    ss_res = float(val_stats[2])
                    if val_n >= 2 and val_ss_tot_cached > 0:
                        val_r2 = 1.0 - ss_res / val_ss_tot_cached
                    else:
                        val_r2 = float("nan")
                    val_data_mse = ss_res / val_n
                else:
                    val_mae = float("inf")
                    val_rmse = float("inf")
                    val_r2 = float("nan")
                    val_data_mse = float("nan")

            scheduler.step(val_mae)
            lr_current = float(optimizer.param_groups[0]["lr"])

            # Per-epoch rate-head validation: MAE and R² vs observed degradation
            # rate on the inner-validation rows. Logged to epoch_log so training
            # curves for rate quality are available for post-hoc plotting without
            # re-running the model.
            with torch.no_grad():
                val_r_full = model.rate(stress_next_dev, features, val_u)
                val_r_sel = val_r_full[val_selector].detach()
                val_obs_rate, val_rate_mask = observed_degradation_rate(
                    u_current[val_selector], u_true[val_selector],
                    delta_full[val_selector],
                    epsilon=1.0e-6,
                    regeneration_tolerance=scaler.epsilon_rec,
                    rate_floor=config.rate_floor,
                )
                if val_rate_mask.any():
                    _rp = val_r_sel[val_rate_mask].cpu().numpy()
                    _ro = val_obs_rate[val_rate_mask].cpu().numpy()
                    val_rate_mae = float(np.mean(np.abs(_rp - _ro)))
                    _ss_r = float(np.sum((_rp - _ro) ** 2))
                    _ro_mean = float(_ro.mean())
                    _ss_t = float(np.sum((_ro - _ro_mean) ** 2))
                    val_rate_r2 = 1.0 - _ss_r / _ss_t if _ss_t > 0 else float("nan")
                    val_rate_coverage = float(int(val_rate_mask.sum()) / max(int(val_rate_mask.numel()), 1))
                else:
                    val_rate_mae = float("nan")
                    val_rate_r2 = float("nan")
                    val_rate_coverage = 0.0

            # Reduce the epoch-wide physics accumulators in a single CPU sync.
            # These describe every row the epoch trained on, not the last batch.
            nan_t = torch.full((), float("nan"), device=device)
            if diag_res_n:
                pde_mae_t = diag_res_abs / diag_res_n
                pde_rmse_t = (diag_res_sq / diag_res_n).sqrt()
            else:
                pde_mae_t = nan_t
                pde_rmse_t = nan_t
            pos_deriv_t = (
                diag_pos_deriv / diag_deriv_n if diag_deriv_n else nan_t
            )
            if diag_state_n:
                mono_frac_t = diag_mono_viol / diag_state_n
                lower_viol_frac_t = diag_lower_viol / diag_state_n
                upper_viol_frac_t = diag_upper_viol / diag_state_n
            else:
                mono_frac_t = nan_t
                lower_viol_frac_t = nan_t
                upper_viol_frac_t = nan_t
            diag_np = torch.stack([
                torch.as_tensor(value, device=device, dtype=torch.float32)
                for value in (
                    pde_mae_t, pde_rmse_t, pos_deriv_t,
                    mono_frac_t, lower_viol_frac_t, upper_viol_frac_t,
                )
            ]).cpu().numpy()
            pde_residual_mae = float(diag_np[0])
            pde_residual_rmse = float(diag_np[1])
            positive_derivative_fraction = float(diag_np[2])
            monotonicity_violation_fraction = float(diag_np[3])
            lower_bound_violation_fraction = float(diag_np[4])
            upper_bound_violation_fraction = float(diag_np[5])

            allocated_mb, reserved_mb = gpu_memory_mb()
            epoch_row = {
                "run_id": run_fingerprint(architecture=config.architecture,
                                           preprocessing=config.preprocessing,
                                           target="normalized_capacity",
                                           fold=config.fold, seed=config.seed,
                                           extra={"epoch": epoch,
                                                   "ablation": ablation_name or ""}),
                "architecture": config.architecture,
                "preprocessing": config.preprocessing,
                "fold": config.fold,
                "seed": config.seed,
                "epoch": epoch,
                "train_total_loss": float(total_loss_scalar),
                "train_data_loss": data_row,
                "train_pde_loss": pde_row_mean,
                "train_pde_anchor_loss": pde_anchor_row,
                "train_pde_collocation_loss": pde_collo_row,
                "train_initial_condition_loss": ic_row,
                "train_integral_loss": integral_row,
                "train_monotonicity_loss": mono_row,
                "train_bounds_loss": bounds_row,
                "train_rate_loss": rate_row,
                "train_discrete_state_transition_loss": discrete_row,
                "validation_data_loss": val_data_mse,
                "validation_MAE": val_mae,
                "validation_RMSE": val_rmse,
                "validation_R2": val_r2,
                "pde_residual_MAE": pde_residual_mae,
                "pde_residual_RMSE": pde_residual_rmse,
                "positive_derivative_fraction": positive_derivative_fraction,
                "monotonicity_violation_fraction": monotonicity_violation_fraction,
                "lower_bound_violation_fraction": lower_bound_violation_fraction,
                "upper_bound_violation_fraction": upper_bound_violation_fraction,
                "effective_lambda_data": float(effective_weights.data),
                "effective_lambda_pde": float(effective_weights.pde),
                "effective_lambda_initial_condition": float(effective_weights.initial_condition),
                "effective_lambda_integral": float(effective_weights.integral),
                "effective_lambda_monotonicity": float(effective_weights.monotonicity),
                "effective_lambda_bounds": float(effective_weights.bounds),
                "effective_lambda_rate": float(effective_weights.rate_regularization),
                "effective_lambda_discrete_transition": float(effective_weights.discrete_state_transition),
                "effective_lambda_rate_data": float(effective_weights.rate_data),
                "effective_lambda_pairwise": float(effective_weights.pairwise),
                "loss_rate_data": rate_data_row,
                "loss_pairwise": pairwise_row,
                "rate_target_coverage": rate_coverage_epoch,
                "validation_rate_MAE": val_rate_mae,
                "validation_rate_R2": val_rate_r2,
                "validation_rate_coverage": val_rate_coverage,
                "pairwise_pair_count": pair_count_epoch,
                "learning_rate": lr_current,
                "gradient_norm_total": grad_total,
                "gradient_norm_pde": grad_pde,
                "GPU_memory_allocated_MB": allocated_mb,
                "GPU_memory_reserved_MB": reserved_mb,
                "best_validation_MAE": min(best_val, val_mae),
                "best_epoch": best_epoch,
                "early_stopping_counter": patience,
                "curriculum_factor": float(factor),
                "physics_full_epoch": physics_full_epoch,
                "checkpoint_eligibility_epoch": checkpoint_eligibility_epoch,
                "checkpoint_eligible": bool(epoch >= checkpoint_eligibility_epoch),
                "n_batches": int(n_batches),
                "batch_mode": str(config.batch_mode),
                "ablation": ablation_name or "",
                # Nominal (pre-curriculum, pre-balance) weights plus the
                # adaptive multipliers actually applied, so the logged effective
                # weight is reproducible. The previous log recorded only the
                # curriculum-scaled weight and silently omitted the balance
                # scale that multiplied the PDE term.
                **{f"nominal_lambda_{k}": float(v)
                   for k, v in loss_weights.as_dict().items()},
                **balancer.state(),
            }
            epoch_rows.append(epoch_row)
            if logger is not None and (epoch % max(1, config.log_every_epochs) == 0
                                        or epoch == config.max_epochs - 1):
                log_epoch_summary(logger, epoch_row)

            # Checkpoint eligibility. A checkpoint may only become the fold's
            # best model once the curriculum factor has reached 1.0, i.e. from
            # ``checkpoint_eligibility_epoch`` onwards. The previous gate used
            # the *warm-up* epoch, which with warmup=0.02 and ramp_end=0.10 let
            # epochs 16..79 win the selection while physics was still ramping —
            # so the reported "full PINN" could be a partially-physics model.
            # Validation metrics are still logged before eligibility begins.
            eligible = epoch >= checkpoint_eligibility_epoch
            if eligible and val_mae + 1e-9 < best_val:
                best_val = val_mae
                best_epoch = epoch
                patience = 0
                atomic_write_torch(
                    {"model": model.state_dict(),
                     "next_epoch": epoch + 1,
                     "best_val": best_val, "best_epoch": best_epoch,
                     "patience": patience},
                    fold_paths.best_model,
                )
            elif eligible:
                patience += 1

            bar.set_postfix({
                "total": f"{total_loss_scalar:.4f}",
                "data": f"{data_row:.4f}",
                "pde": f"{pde_row_mean:.4f}",
                "val_MAE": f"{val_mae:.4f}",
                "best": f"{best_val:.4f}",
                "pat": f"{patience}/{config.early_stopping_patience}",
                "lr": f"{lr_current:.2e}",
            })

            if (epoch + 1) % max(1, config.save_checkpoint_every_epochs) == 0 or epoch == config.max_epochs - 1:
                atomic_write_torch(
                    {"model": model.state_dict(),
                     "next_epoch": epoch + 1,
                     "best_val": best_val, "best_epoch": best_epoch,
                     "patience": patience},
                    fold_paths.last_model,
                )
                atomic_write_torch(optimizer.state_dict(), fold_paths.optimizer_state)
                atomic_write_torch(scheduler.state_dict(), fold_paths.scheduler_state)

            # Early stopping also requires eligibility: patience only starts
            # accumulating once physics is fully active, so a fold cannot stop
            # before it has trained at least one fully-physics-active epoch.
            if (epoch >= checkpoint_eligibility_epoch
                    and patience >= config.early_stopping_patience
                    and epoch >= config.min_epochs):
                break
    except FoldTrainingError:
        bar.close()
        _write_failure(fold_paths, config, ablation_name, fingerprint,
                        trainable_parameters, best_epoch, best_val, logger,
                        epoch_rows)
        raise
    except Exception as exc:  # pragma: no cover — propagate as FoldTrainingError
        bar.close()
        _write_failure(fold_paths, config, ablation_name, fingerprint,
                        trainable_parameters, best_epoch, best_val, logger,
                        epoch_rows,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc())
        raise FoldTrainingError(f"{type(exc).__name__}: {exc}") from exc

    bar.close()

    epoch_log = pd.DataFrame(epoch_rows)
    if not epoch_log.empty:
        atomic_write_csv(epoch_log, fold_paths.epoch_log_csv)
        try:
            atomic_write_parquet(epoch_log, fold_paths.epoch_log_parquet)
        except Exception as _parquet_exc:
            if logger is not None:
                log_event(logger, logging.WARNING, "epoch_log_parquet_write_failed",
                          path=str(fold_paths.epoch_log_parquet),
                          error=str(_parquet_exc))
            print(f"[trainer] WARNING: epoch_log.parquet write failed "
                  f"({type(_parquet_exc).__name__}: {_parquet_exc}); "
                  f"CSV fallback written to {fold_paths.epoch_log_csv}")

    # Selection provenance: fail loudly rather than reporting a checkpoint that
    # was chosen while the physics curriculum was still ramping.
    selected_curriculum_factor = curriculum.factor(best_epoch) if best_epoch >= 0 else 0.0
    fully_active_epochs = max(0, int(last_epoch_seen) - physics_full_epoch + 1)
    curriculum_provenance = {
        "physics_full_epoch": physics_full_epoch,
        "checkpoint_eligibility_epoch": checkpoint_eligibility_epoch,
        "selected_epoch": int(best_epoch),
        "curriculum_factor_at_selected_epoch": float(selected_curriculum_factor),
        "fully_physics_active_epochs": fully_active_epochs,
        "effective_physics_weights_at_selected_epoch": (
            loss_weights.effective(selected_curriculum_factor).as_dict()
        ),
        "batch_mode": str(config.batch_mode),
        "n_batches_per_epoch": int(n_batches),
        "n_training_cells": int(sampler.n_cells),
    }
    manifest["curriculum"] = curriculum_provenance
    manifest["effective_settings"] = asdict(config)
    atomic_write_json(manifest, fold_paths.manifest_path)

    if (config.checkpoint_after_physics_active and best_epoch >= 0
            and selected_curriculum_factor < 1.0 - 1e-9):
        raise FoldTrainingError(
            f"selected checkpoint at epoch {best_epoch} has curriculum factor "
            f"{selected_curriculum_factor:.4f} < 1.0 (physics_full_epoch="
            f"{physics_full_epoch}); refusing to report a partially "
            f"physics-active model as the full PINN"
        )

    # Restore best model weights for evaluation.
    if fold_paths.best_model.exists():
        best_state = torch.load(fold_paths.best_model, map_location=device,
                                 weights_only=False)
        model.load_state_dict(best_state["model"])

    # Inner-train and inner-validation predictions always come from the
    # inner-training model with the inner-training scaler — that is the model
    # that produced early-stopping decisions.
    prediction_frames = []
    for indices, split_name in (
        (inner_train_idx, "inner_train"),
        (inner_val_idx, "inner_validation"),
    ):
        eval_data = _evaluate(model, dataset, frame, indices, scaler, device,
                               config, config.quadrature_method,
                               config.quadrature_nodes)
        if not eval_data:
            continue
        prediction_frames.append(_prediction_frame(
            config.architecture, config.preprocessing, config.fold, config.seed,
            split_name, indices, dataset, frame,
            eval_data["u_next"], eval_data["r_next"], eval_data["du_ds"],
            eval_data["pde_residual"], eval_data["ic_error"],
            eval_data["integral_error"], eval_data["mono_violation"],
            eval_data["lower_violation"], eval_data["upper_violation"],
            stress_std=scaler.stress_std,
            ablation_name=ablation_name,
        ))

    # Two-phase outer-training refit: rebuild statistics and model on ALL
    # outer-training cells, train for best_epoch+1 epochs without early
    # stopping, and produce the outer-validation predictions from the refit
    # model. This mirrors the classical model, which is trained on the full
    # outer-training fold, and stops discarding the inner-validation cells
    # from the final production model.
    used_refit = False
    refit_model = None
    refit_scaler = scaler
    # Where the refit length comes from. Selecting it on this run's own single
    # inner split is high variance when that split holds four or five cells;
    # nested selection supplies the argmin of the mean curve across all inner
    # folds instead. Phase 1 is unchanged either way.
    selected_epoch = (
        int(config.forced_best_epoch)
        if config.forced_best_epoch is not None else best_epoch
    )
    epoch_selection_source = (
        "nested_inner_folds" if config.forced_best_epoch is not None
        else "single_inner_split"
    )
    if (config.two_phase_refit and np.isfinite(best_val)
            and selected_epoch >= 0):
        try:
            refit_model, refit_scaler = _refit_full_outer_train(
                dataset=dataset, frame=frame, outer_train_idx=outer_train_idx,
                original_scaler=scaler, best_epoch=selected_epoch,
                config=config, loss_weights=loss_weights,
                u_max_source=u_max_source, u_max_constant=u_max_constant,
                u_max_margin=u_max_margin,
                tolerance_source=tolerance_source,
                tolerance_constant=tolerance_constant,
                tolerance_quantile=tolerance_quantile,
                include_discrete_transition=include_discrete_transition,
                device=device, logger=logger, ablation_name=ablation_name,
            )
            used_refit = True
            atomic_write_torch(
                {
                    "model": refit_model.state_dict(),
                    "architecture": config.architecture,
                    "preprocessing": config.preprocessing,
                    "fold": config.fold,
                    "seed": config.seed,
                    "refit_epochs": selected_epoch + 1,
                    "fingerprint": fingerprint,
                },
                fold_paths.final_model,
            )
            atomic_write_json(refit_scaler.state_dict(), fold_paths.final_scaler)
            manifest["selection_checkpoint"] = "checkpoints/best_model.pt"
            manifest["final_checkpoint"] = "checkpoints/final_refit_model.pt"
            manifest["final_scaler"] = "final_refit_scaler.json"
            atomic_write_json(manifest, fold_paths.manifest_path)
            if logger is not None:
                log_event(logger, logging.INFO, "two_phase_refit_completed",
                          architecture=config.architecture,
                          preprocessing=config.preprocessing,
                          fold=config.fold, seed=config.seed,
                          refit_epochs=selected_epoch + 1)
        except FoldTrainingError:
            # Refit failure is a hard failure — do not silently fall back to
            # the inner-only model, since the reported outer-val MAE would
            # then be under-trained relative to the classical baseline.
            raise
        except Exception as exc:
            raise FoldTrainingError(
                f"two-phase refit failed: {type(exc).__name__}: {exc}"
            ) from exc

    outer_eval_model = refit_model if used_refit else model
    outer_eval_scaler = refit_scaler
    outer_eval_data = _evaluate(
        outer_eval_model, dataset, frame, outer_val_idx, outer_eval_scaler,
        device, config, config.quadrature_method,
        config.quadrature_nodes,
    )
    if outer_eval_data:
        prediction_frames.append(_prediction_frame(
            config.architecture, config.preprocessing, config.fold, config.seed,
            "outer_validation", outer_val_idx, dataset, frame,
            outer_eval_data["u_next"], outer_eval_data["r_next"],
            outer_eval_data["du_ds"], outer_eval_data["pde_residual"],
            outer_eval_data["ic_error"], outer_eval_data["integral_error"],
            outer_eval_data["mono_violation"], outer_eval_data["lower_violation"],
            outer_eval_data["upper_violation"],
            stress_std=outer_eval_scaler.stress_std,
            ablation_name=ablation_name,
        ))

    predictions_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame()
    )
    if not predictions_frame.empty:
        atomic_write_parquet(predictions_frame, fold_paths.predictions_path)

    physics_summary = _summarize_outer_physics(
        outer_eval_data,
        stress_std=float(refit_scaler.stress_std),
        curriculum=curriculum_provenance,
    )

    atomic_write_json({
        "status": "completed",
        "error_message": None,
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "fold": config.fold,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_val,
        "trainable_parameters": trainable_parameters,
        "physics_metrics": physics_summary,
        "ablation": ablation_name or "",
        "fingerprint": fingerprint,
        "inner_stress_std": float(scaler.stress_std),
        "final_refit_stress_std": float(outer_eval_scaler.stress_std),
        "stress_std": float(outer_eval_scaler.stress_std),
        "two_phase_refit_used": bool(used_refit),
        "refit_epochs": int(selected_epoch + 1) if used_refit else 0,
        "refit_selected_epoch": int(selected_epoch),
        "refit_epoch_selection_source": epoch_selection_source,
        "inner_fold": int(config.inner_fold),
        "n_inner_folds": int(config.n_inner_folds),
        # Resolved backend flags. Recorded because a reproducibility claim that
        # is not written down cannot be checked after the fact.
        **backend_state,
    }, fold_paths.status_path)

    if logger is not None:
        log_event(logger, logging.INFO, "fold_completed",
                  architecture=config.architecture,
                  preprocessing=config.preprocessing,
                  fold=config.fold, seed=config.seed,
                  status="completed",
                  best_epoch=best_epoch,
                  best_validation_mae=best_val,
                  physics_metrics=physics_summary,
                  n_epoch_rows=len(epoch_rows),
                  fingerprint=fingerprint)

    return FoldResult(
        architecture=config.architecture, preprocessing=config.preprocessing,
        fold=config.fold, seed=config.seed,
        trainable_parameters=trainable_parameters,
        best_epoch=best_epoch, best_validation_mae=best_val,
        predictions_path=fold_paths.predictions_path,
        physics_metrics=physics_summary,
        status="completed",
        error_message=None,
        ablation_name=ablation_name,
        epoch_log=epoch_rows,
        fingerprint=fingerprint,
    )


def _refit_full_outer_train(
    *, dataset: AnchorDataset, frame: pd.DataFrame,
    outer_train_idx: np.ndarray, original_scaler: FoldScaler,
    best_epoch: int, config: TrainerConfig, loss_weights: LossWeights,
    u_max_source: str, u_max_constant: float, u_max_margin: float,
    tolerance_source: str, tolerance_constant: float, tolerance_quantile: float,
    include_discrete_transition: bool,
    device: str, logger: logging.Logger | None, ablation_name: str | None,
) -> tuple[nn.Module, FoldScaler]:
    """Fit a fresh model on ALL outer-training rows for ``best_epoch + 1``
    epochs, with no early stopping and no scheduler patience.

    Returns ``(refit_model, refit_scaler)``. Raises FoldTrainingError if the
    refit scaler drops features that the inner-training scaler kept (which
    would silently change model architecture).
    """
    outer_train_frame = frame.iloc[outer_train_idx].reset_index(drop=True)
    # Fit on the exact columns selected during inner training so that the
    # refit scaler's statistical arrays (mean, std, medians) and its
    # feature_columns_used are always built together and always aligned.
    # Fitting on the full candidate set and then overwriting feature_columns_used
    # is unsafe: the statistical arrays would still index refit_cols, not
    # original_cols, causing broadcasting errors or silent column mismatches
    # for sparse P3 waveform features that appear only on the larger dataset.
    original_cols = list(original_scaler.feature_columns_used or [])
    if not original_cols:
        raise FoldTrainingError(
            "inner-training scaler retained no features"
        )
    refit_scaler = FoldScaler().fit(
        outer_train_frame, original_cols,
        u_max_source=u_max_source, u_max_constant=u_max_constant,
        u_max_margin=u_max_margin, tolerance_source=tolerance_source,
        tolerance_constant=tolerance_constant,
        tolerance_quantile=tolerance_quantile,
        use_unified=config.preprocessing == UNIFIED_PREPROCESSING,
    )
    if refit_scaler.feature_columns_used != original_cols:
        raise FoldTrainingError(
            "full outer-training refit changed the selected feature schema"
        )

    set_seeds(config.seed)
    backend_state = configure_torch_backend(deterministic=config.deterministic)
    feature_dim = 2 + len(original_cols)

    refit_hybrid_temp_idx: int | None = None
    refit_hybrid_dod_idx: int | None = None
    refit_hybrid_crate_idx: int | None = None
    if config.use_hybrid_rate:
        rcols = list(original_cols)
        refit_hybrid_temp_idx = 2 + rcols.index(config.hybrid_temperature_feature)
        refit_hybrid_dod_idx = 2 + rcols.index(config.hybrid_dod_feature)
        if config.hybrid_c_rate_feature and config.hybrid_c_rate_feature in rcols:
            refit_hybrid_crate_idx = 2 + rcols.index(config.hybrid_c_rate_feature)
        else:
            refit_hybrid_crate_idx = refit_hybrid_dod_idx

    refit_model = NaPINNQ(
        feature_dim=feature_dim,
        solution_hidden_dims=config.solution_hidden_dims,
        rate_hidden_dims=config.rate_hidden_dims,
        solution_activation=config.solution_activation,
        rate_activation=config.rate_activation,
        solution_dropout=config.solution_dropout,
        rate_dropout=config.rate_dropout,
        rate_uses_u_hat=config.rate_uses_u_hat,
        predict_delta_u=config.predict_delta_u,
        rate_floor=config.rate_floor,
        use_hybrid_rate=config.use_hybrid_rate,
        hybrid_temperature_index=refit_hybrid_temp_idx,
        hybrid_dod_index=refit_hybrid_dod_idx,
        hybrid_c_rate_index=refit_hybrid_crate_idx,
        hybrid_enable_cold_regime=config.hybrid_enable_cold_regime,
        hybrid_fit_c_rate_exponent=config.hybrid_fit_c_rate_exponent,
        hybrid_residual_hidden_dims=config.hybrid_residual_hidden_dims,
    )
    refit_model.to(device)
    optimizer = _init_optimizer(refit_model, config)

    # Build training tensors on the FULL outer-training frame.
    features_np = _build_feature_matrix(dataset, outer_train_frame, refit_scaler)
    stress_current_np = refit_scaler.transform_stress(
        outer_train_frame["stress_current"].to_numpy(dtype=float))
    stress_next_np = refit_scaler.transform_stress(
        outer_train_frame["stress_next"].to_numpy(dtype=float))
    delta_np = refit_scaler.transform_horizon(
        outer_train_frame["stress_delta"].to_numpy(dtype=float))
    u_current_np = outer_train_frame["u_current"].to_numpy(dtype=float)
    u_true_np = outer_train_frame["u_true_next"].to_numpy(dtype=float)
    cell_codes, _ = _cell_index(outer_train_frame["cell"])

    features = _tensor(features_np, device)
    delta = _tensor(delta_np, device)
    u_current = _tensor(u_current_np, device)
    u_true = _tensor(u_true_np, device)
    cell_index = torch.as_tensor(cell_codes, device=device, dtype=torch.long)
    # Cached stress tensors on device — same rationale as the main training
    # loop: we don't want to re-upload numpy → GPU on every batch.
    stress_current_dev = _tensor(stress_current_np, device)
    stress_next_dev = _tensor(stress_next_np, device)

    # The curriculum must be built from the ACTUAL refit duration. Reusing
    # ``config.max_epochs`` (the inner-training budget) meant a refit lasting
    # fewer than ramp_end_fraction * max_epochs epochs never reached curriculum
    # factor 1.0 — so the model producing the reported outer-validation
    # predictions was trained under weaker physics than the model whose epoch
    # was selected. Deriving the schedule from ``refit_epochs`` guarantees the
    # refit reaches full physics.
    refit_epochs = max(1, best_epoch + 1)
    curriculum = CurriculumSchedule(
        max_epochs=refit_epochs,
        warmup_fraction=config.curriculum_warmup_fraction,
        ramp_end_fraction=config.curriculum_ramp_end_fraction,
    )
    refit_physics_full_epoch = curriculum.physics_full_epoch
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    # Mirror the inner-training batching exactly, including cell balancing.
    n_rows = int(u_current.numel())
    delta_full = torch.as_tensor(delta_np, dtype=torch.float32, device=device)
    all_row_positions = torch.arange(n_rows, device=device)
    sampler = CellBalancedSampler(
        row_positions=all_row_positions,
        cell_index=cell_index,
        batch_size=int(config.batch_size),
        mode=str(config.batch_mode),
    )
    balancer = GradientBalancer(
        ema=config.gradient_balance_ema,
        min_multiplier=config.gradient_balance_min_multiplier,
        max_multiplier=config.gradient_balance_max_multiplier,
    )
    warmup_epochs = curriculum.warmup_epochs
    n_full_physics_epochs = 0

    for epoch in range(refit_epochs):
        refit_model.train()
        factor = curriculum.factor(epoch)
        effective_weights = loss_weights.effective(factor)
        if factor >= 1.0 - 1e-9:
            n_full_physics_epochs += 1

        for b, batch_positions in enumerate(sampler.epoch_batches(generator)):
            if batch_positions.numel() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            batch_features = features[batch_positions]
            batch_u_current = u_current[batch_positions]
            batch_u_true = u_true[batch_positions]
            batch_cells = cell_index[batch_positions]
            batch_delta = delta_full[batch_positions]

            stress_current_tensor = (
                stress_current_dev[batch_positions].detach().requires_grad_(True)
            )
            stress_next_tensor = (
                stress_next_dev[batch_positions].detach().requires_grad_(True)
            )

            u_next_hat = refit_model.solution(
                stress_next_tensor, batch_features, batch_u_current,
                stress_current_tensor,
            )
            r_next_hat = refit_model.rate(
                stress_next_tensor, batch_features, u_next_hat
            )
            # Anchor coordinate detached: see the note in the phase-1 loop.
            u_anchor_hat = refit_model.solution(
                stress_current_tensor, batch_features, batch_u_current,
                stress_current_tensor.detach(),
            )
            r_anchor_hat = refit_model.rate(
                stress_current_tensor, batch_features, u_anchor_hat
            )
            du_next = autograd_du_ds(u_next_hat, stress_next_tensor)
            du_current = autograd_du_ds(u_anchor_hat, stress_current_tensor)
            residual_anchor = pde_residual(du_current, r_anchor_hat)
            collocation = sample_collocation_points(
                stress_current=stress_current_tensor.detach(),
                stress_delta=batch_delta,
                features=batch_features,
                cell_index=batch_cells,
                points_per_transition=config.collocation_points,
                generator=generator,
            )
            collo_stress = collocation.stress.clone().detach().requires_grad_(True)
            collo_u_current = batch_u_current[collocation.row_index]
            collo_stress_current = (
                stress_current_tensor.detach()[collocation.row_index]
            )
            u_collo = refit_model.solution(
                collo_stress, collocation.features, collo_u_current,
                collo_stress_current,
            )
            r_collo = refit_model.rate(collo_stress, collocation.features, u_collo)
            du_collo = autograd_du_ds(u_collo, collo_stress)
            residual_collo = pde_residual(du_collo, r_collo)
            u_integrated = integral_transition(
                refit_model.solution, refit_model.rate,
                stress_current_tensor, batch_delta,
                batch_features, batch_u_current,
                method=config.quadrature_method, n_nodes=config.quadrature_nodes,
            )

            data_l, _ = data_loss(
                u_next_hat, batch_u_true, batch_cells, delta=config.huber_delta,
            )
            pde_anchor_l, _ = pde_loss(residual_anchor, batch_cells)
            pde_collo_l = residual_collo.new_zeros(())
            if residual_collo.numel():
                pde_collo_l, _ = pde_loss(residual_collo, collocation.cell_index)
            ic_l, _ = initial_condition_loss(
                u_anchor_hat, batch_u_current, batch_cells,
            )
            integral_l, _ = integral_consistency_loss(
                u_next_hat, u_integrated, batch_cells,
            )
            mono_l, _ = monotonicity_loss(
                u_next_hat, batch_u_current, batch_cells,
                epsilon_rec=refit_scaler.epsilon_rec,
            )
            bounds_l, _ = bounds_loss(
                u_next_hat, batch_cells, u_min=0.0, u_max=refit_scaler.u_max,
            )
            rate_l, _ = rate_regularization_loss(r_next_hat, batch_cells)

            rate_data_l = u_next_hat.new_zeros(())
            if effective_weights.rate_data > 0.0:
                observed_rate, rate_mask = observed_degradation_rate(
                    batch_u_current, batch_u_true, batch_delta,
                    epsilon=1.0e-6,
                    regeneration_tolerance=refit_scaler.epsilon_rec,
                    rate_floor=config.rate_floor,
                )
                rate_data_l, _, _ = rate_data_loss(
                    r_anchor_hat, observed_rate, batch_cells, rate_mask,
                    delta=config.huber_delta,
                )

            pairwise_l = u_next_hat.new_zeros(())
            if effective_weights.pairwise > 0.0:
                pairwise_l, _, _ = pairwise_delta_loss(
                    u_next_hat, batch_u_current, batch_u_true, batch_cells,
                    generator=generator,
                    max_pairs=config.pairwise_max_pairs,
                    delta=config.huber_delta,
                )

            discrete_l = u_next_hat.new_zeros(())
            if include_discrete_transition:
                discrete_residual = discrete_state_transition_residual(
                    u_next_hat, batch_u_current, r_anchor_hat, batch_delta,
                )
                discrete_l, _ = discrete_state_transition_loss(
                    discrete_residual, batch_cells,
                )

            pde_total = pde_anchor_l + pde_collo_l
            parameters = list(refit_model.parameters())
            multipliers = {name: 1.0 for name in BALANCED_COMPONENTS}
            if config.use_gradient_balance and epoch >= warmup_epochs:
                probes = {
                    "data": effective_weights.data * data_l,
                    "pde": effective_weights.pde * pde_total,
                    "integral": effective_weights.integral * integral_l,
                    "monotonicity": effective_weights.monotonicity * mono_l,
                    "bounds": effective_weights.bounds * bounds_l,
                    "rate_data": effective_weights.rate_data * rate_data_l,
                    "pairwise": effective_weights.pairwise * pairwise_l,
                }
                balancer.observe("data", _pde_gradient_norm(
                    probes["data"], parameters, logger=None,
                ))
                active_components = set()
                for name in ("pde", "integral", "monotonicity", "bounds",
                             "rate_data", "pairwise"):
                    if getattr(effective_weights, name) <= 0.0:
                        continue
                    active_components.add(name)
                    balancer.observe(name, _pde_gradient_norm(
                        probes[name], parameters, logger=None,
                    ))
                n_active = (
                    len(active_components)
                    if config.gradient_balance_normalize_active
                    else 1
                )
                multipliers = {
                    name: balancer.multiplier(name, n_active=n_active)
                    for name in BALANCED_COMPONENTS
                }

            total = (
                effective_weights.data * data_l
                + effective_weights.ode * multipliers["pde"] * pde_total
                + effective_weights.initial_condition * ic_l
                + effective_weights.integral * multipliers["integral"] * integral_l
                + effective_weights.monotonicity * multipliers["monotonicity"] * mono_l
                + effective_weights.bounds * multipliers["bounds"] * bounds_l
                + effective_weights.rate_regularization * rate_l
                + effective_weights.rate_data * multipliers["rate_data"] * rate_data_l
                + effective_weights.pairwise * multipliers["pairwise"] * pairwise_l
                + effective_weights.discrete_state_transition * discrete_l
            )
            if (effective_weights.residual_share > 0.0
                    and isinstance(refit_model.rate, HybridRateModel)):
                refit_comps = refit_model.rate.components(
                    stress_next_tensor, batch_features, u_next_hat.detach()
                )
                total = total + effective_weights.residual_share * residual_share_penalty(
                    refit_comps, limit=config.hybrid_residual_share_limit,
                )
                total = total + parameter_prior_penalty(refit_model.rate)
            if not torch.isfinite(total):
                raise FoldTrainingError(
                    f"non-finite refit loss at refit epoch {epoch} batch {b}: {total.item()}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(refit_model.parameters(), config.gradient_clip_norm)
            optimizer.step()

        if logger is not None and (epoch % max(1, config.log_every_epochs) == 0
                                    or epoch == refit_epochs - 1):
            log_fold_curriculum_state(
                logger,
                epoch=epoch,
                curriculum_factor=factor,
                physics_full_epoch=refit_physics_full_epoch,
                effective_weights=effective_weights.as_dict(),
                balancer_state=balancer.state() if config.use_gradient_balance else {},
                is_refit=True,
            )

    if n_full_physics_epochs <= 0:
        raise FoldTrainingError(
            f"two-phase refit ran {refit_epochs} epochs but never reached "
            f"curriculum factor 1.0 (physics_full_epoch={refit_physics_full_epoch}); "
            f"the refit model would be weaker in physics than the selected one"
        )
    if logger is not None:
        log_event(logger, logging.INFO, "refit_curriculum",
                  architecture=config.architecture,
                  preprocessing=config.preprocessing,
                  fold=config.fold, seed=config.seed,
                  refit_epochs=refit_epochs,
                  refit_physics_full_epoch=refit_physics_full_epoch,
                  full_physics_epochs=n_full_physics_epochs,
                  refit_feature_hash=refit_scaler.feature_hash()
                  if hasattr(refit_scaler, "feature_hash") else None,
                  final_effective_weights=effective_weights.as_dict())

    return refit_model, refit_scaler


def _summarize_outer_physics(outer_eval: dict[str, np.ndarray], *,
                             stress_std: float = 1.0,
                             curriculum: dict[str, Any] | None = None
                             ) -> dict[str, float]:
    if not outer_eval:
        return {}
    residual = outer_eval["pde_residual"]
    summary = {
        # Normalized-coordinate residual (the coordinate the PDE is solved in).
        "pde_residual_MAE": float(np.mean(np.abs(residual))),
        "pde_residual_RMSE": float(np.sqrt(np.mean(residual ** 2))),
        # Physical units: du/ds carries 1/stress, so converting back to the raw
        # stress coordinate divides by the fold's stress scale.
        "pde_residual_MAE_physical": float(
            np.mean(np.abs(residual)) / stress_std
        ) if stress_std else float("nan"),
        "positive_derivative_fraction": float((outer_eval["du_ds"] > 0).mean()),
        "monotonicity_violation_fraction": float((outer_eval["mono_violation"] > 0).mean()),
        "mean_monotonicity_violation": float(outer_eval["mono_violation"].mean()),
        "max_monotonicity_violation": float(outer_eval["mono_violation"].max())
            if outer_eval["mono_violation"].size else float("nan"),
        "initial_condition_MAE": float(np.mean(np.abs(outer_eval["ic_error"]))),
        "integral_consistency_MAE": float(np.mean(np.abs(outer_eval["integral_error"]))),
        "lower_bound_violation_fraction": float((outer_eval["lower_violation"] > 0).mean()),
        "upper_bound_violation_fraction": float((outer_eval["upper_violation"] > 0).mean()),
        "negative_rate_fraction": float((outer_eval["r_next"] < 0).mean()),
        "mean_r": float(outer_eval["r_next"].mean()),
        "std_r": float(outer_eval["r_next"].std()),
    }

    # Rate-head diagnostics against the observed degradation rate.
    observed = outer_eval.get("observed_rate")
    predicted = outer_eval.get("r_anchor")
    valid = outer_eval.get("rate_valid")
    if observed is not None and predicted is not None and valid is not None:
        n_total = int(valid.size)
        if valid.any():
            summary.update({
                "rate_data_MAE": float(
                    np.mean(np.abs(predicted[valid] - observed[valid]))
                ),
                "rate_target_coverage": float(int(valid.sum()) / max(n_total, 1)),
                "observed_rate_mean": float(observed[valid].mean()),
                "observed_rate_std": float(observed[valid].std()),
                "predicted_rate_mean": float(predicted[valid].mean()),
                "predicted_rate_std": float(predicted[valid].std()),
            })
        else:
            summary.update({
                "rate_data_MAE": float("nan"),
                "rate_target_coverage": 0.0,
                "observed_rate_mean": float("nan"),
                "observed_rate_std": float("nan"),
                "predicted_rate_mean": float("nan"),
                "predicted_rate_std": float("nan"),
            })

    if curriculum:
        summary["selected_epoch"] = float(curriculum.get("selected_epoch", -1))
        summary["curriculum_factor_at_selected_epoch"] = float(
            curriculum.get("curriculum_factor_at_selected_epoch", float("nan"))
        )
        summary["fully_physics_active_epochs"] = float(
            curriculum.get("fully_physics_active_epochs", 0)
        )
        for name, value in (
            curriculum.get("effective_physics_weights_at_selected_epoch") or {}
        ).items():
            summary[f"selected_effective_lambda_{name}"] = float(value)
    return summary


def _write_failure(fold_paths: FoldPaths, config: TrainerConfig,
                    ablation_name: str | None, fingerprint: str,
                    trainable_parameters: int, best_epoch: int, best_val: float,
                    logger: logging.Logger | None,
                    epoch_rows: list[dict[str, Any]],
                    *, error: str | None = None,
                    traceback_text: str | None = None) -> None:
    """Persist a status=failed marker without predictions.

    Aggregator scripts must key on this status when deciding what to include.
    """
    if epoch_rows:
        try:
            df = pd.DataFrame(epoch_rows)
            atomic_write_csv(df, fold_paths.epoch_log_csv)
            atomic_write_parquet(df, fold_paths.epoch_log_parquet)
        except Exception:
            pass
    atomic_write_json({
        "status": "failed",
        "error_message": error,
        "traceback": traceback_text,
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "fold": config.fold,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_val,
        "trainable_parameters": trainable_parameters,
        "physics_metrics": {},
        "ablation": ablation_name or "",
        "fingerprint": fingerprint,
    }, fold_paths.status_path)
    # Delete stale predictions from a previous completed run so the aggregator
    # cannot re-consume them under a new (failed) fingerprint.
    if fold_paths.predictions_path.exists():
        try:
            fold_paths.predictions_path.unlink()
        except OSError:
            pass
    if logger is not None:
        log_event(logger, logging.WARNING, "fold_marked_failed",
                  architecture=config.architecture,
                  preprocessing=config.preprocessing,
                  fold=config.fold, seed=config.seed,
                  error=error, fingerprint=fingerprint)
