"""Fold-level trainer for DNN-Q and NaPINN-Q.

Implements the composite cell-balanced loss, physics-weight curriculum,
autograd-based governing residual, unlabeled collocation points, checkpointing
and resume, plus per-epoch physics + validation metrics.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .collocation import sample_collocation_points
from .dataset import AnchorDataset, FoldScaler, fold_indices, inner_split_indices
from .logging_utils import (
    log_epoch_summary, log_event, log_failure, log_fold_summary,
)
from .losses import (
    CurriculumSchedule, LossWeights, bounds_loss, cell_balanced_reduce,
    data_loss, discrete_state_transition_loss, initial_condition_loss,
    integral_consistency_loss, monotonicity_loss, pde_loss, rate_regularization_loss,
)
from .models import DNNQ, NaPINNQ, count_parameters
from .physics import (
    autograd_du_ds, discrete_state_transition_residual, integral_transition,
    pde_residual,
)
from .utils import (
    atomic_write_csv, atomic_write_json, atomic_write_parquet, atomic_write_torch,
    finite_or_raise, gpu_memory_mb, resolve_device, run_fingerprint, set_seeds,
)


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
    use_amp: bool = True
    physics_float32: bool = True
    curriculum_warmup_fraction: float = 0.10
    curriculum_ramp_end_fraction: float = 0.30
    huber_delta: float = 1.0
    collocation_points: int = 8
    quadrature_method: str = "trapezoidal"
    quadrature_nodes: int = 8
    solution_hidden_dims: tuple[int, ...] = (64, 64, 32)
    rate_hidden_dims: tuple[int, ...] = (32, 16)
    solution_activation: str = "tanh"
    rate_activation: str = "tanh"
    maximum_parameters: int = 50000
    log_every_epochs: int = 1
    save_checkpoint_every_epochs: int = 100
    log_gpu_memory: bool = True
    log_gradient_norms: bool = True
    device: str = "cuda"


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
    def config_snapshot(self) -> Path:
        return self.root / "config_snapshot.json"

    @property
    def audit_subset(self) -> Path:
        return self.root / "temporal_audit_subset.csv"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.parquet"

    @property
    def collocation_predictions_path(self) -> Path:
        return self.root / "collocation_predictions.parquet"


def _tensor(values, device: str, dtype=torch.float32,
            requires_grad: bool = False) -> torch.Tensor:
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
    # Feature vector: [horizon, u_current, scaled auxiliary features]
    return np.concatenate(
        [horizon[:, None], u_current[:, None], scaled_features], axis=1
    )


def _forward_pinn(model: NaPINNQ, stress: torch.Tensor, features: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    u_hat = model.solution(stress, features)
    r_hat = model.rate(stress, features, u_hat)
    return u_hat, r_hat


def _gradient_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


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
    state: dict[str, Any] = {"epoch": 0, "best_val": float("inf"),
                              "best_epoch": 0, "patience": 0}
    if paths.last_model.exists():
        payload = torch.load(paths.last_model, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        state.update({k: payload.get(k, state[k]) for k in state})
    if paths.optimizer_state.exists():
        optimizer.load_state_dict(torch.load(paths.optimizer_state, map_location="cpu",
                                             weights_only=False))
    if paths.scheduler_state.exists() and scheduler is not None:
        scheduler.load_state_dict(torch.load(paths.scheduler_state, map_location="cpu",
                                             weights_only=False))
    return state


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


def _prediction_frame(architecture: str, preprocessing: str, fold: int, seed: int,
                      evaluation_role: str, indices: np.ndarray,
                      dataset: AnchorDataset, frame: pd.DataFrame,
                      predicted_u_next: np.ndarray, predicted_r: np.ndarray,
                      du_ds: np.ndarray, pde_res: np.ndarray, ic_err: np.ndarray,
                      integral_err: np.ndarray, mono_viol: np.ndarray,
                      lower_viol: np.ndarray, upper_viol: np.ndarray,
                      ablation_name: str | None = None) -> pd.DataFrame:
    sub = frame.iloc[indices].reset_index(drop=True)
    q0 = sub[dataset.q0_column].to_numpy(dtype=float)
    q_current = sub[dataset.q_current_column].to_numpy(dtype=float)
    predicted_q_next = predicted_u_next * q0
    predicted_soh_next = 100.0 * predicted_u_next
    predicted_delta_q = predicted_q_next - q_current
    predicted_delta_soh = predicted_soh_next - (100.0 * q_current / q0)
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
        "predicted_degradation_rate": predicted_r,
        "du_ds": du_ds,
        "pde_residual": pde_res,
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
               is_pinn: bool, config: TrainerConfig, quadrature_method: str,
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
        if is_pinn:
            u_next, r_next = _forward_pinn(model, stress_next, features)
            u_anchor = model.solution(stress_current, features)
            r_anchor = model.rate(stress_current, features, u_anchor)
            du_next = autograd_du_ds(u_next, stress_next)
            residual = pde_residual(du_next, r_next)
            u_integrated = integral_transition(
                model.solution, model.rate, stress_current, delta, features,
                u_current, method=quadrature_method, n_nodes=quadrature_nodes,
            )
        else:
            u_next = model(stress_next, features)
            u_anchor = model(stress_current, features)
            r_next = torch.zeros_like(u_next)
            r_anchor = torch.zeros_like(u_anchor)
            du_next = autograd_du_ds(u_next, stress_next)
            residual = torch.zeros_like(u_next)
            u_integrated = torch.zeros_like(u_next)

    u_next_np = u_next.detach().cpu().numpy()
    r_next_np = r_next.detach().cpu().numpy()
    du_ds_np = du_next.detach().cpu().numpy()
    residual_np = residual.detach().cpu().numpy()
    ic_err_np = (u_anchor.detach().cpu().numpy() - u_current_np)
    integral_err_np = (u_next_np - u_integrated.detach().cpu().numpy())
    mono_viol_np = np.maximum(u_next_np - u_current_np - scaler.epsilon_rec, 0.0)
    lower_viol_np = np.maximum(0.0 - u_next_np, 0.0)
    upper_viol_np = np.maximum(u_next_np - scaler.u_max, 0.0)
    return {
        "u_next": u_next_np,
        "r_next": r_next_np,
        "du_ds": du_ds_np,
        "pde_residual": residual_np,
        "ic_error": ic_err_np,
        "integral_error": integral_err_np,
        "mono_violation": mono_viol_np,
        "lower_violation": lower_viol_np,
        "upper_violation": upper_viol_np,
        "u_true": u_true_np,
    }


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
               logger: logging.Logger | None = None) -> FoldResult:
    """Train one fold + seed for the requested architecture.

    ``ablation_name`` is stored on prediction rows and status files for later
    reporting. When ``include_discrete_transition`` is true, the A5 discrete
    residual is added to the total loss.
    """
    fold_paths.root.mkdir(parents=True, exist_ok=True)
    fold_paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_paths.logs_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    # Skip completed runs unless --force.
    if not force and fold_paths.status_path.exists():
        try:
            status = json.loads(fold_paths.status_path.read_text())
            if status.get("status") == "completed":
                return FoldResult(
                    architecture=config.architecture,
                    preprocessing=config.preprocessing,
                    fold=config.fold, seed=config.seed,
                    trainable_parameters=int(status.get("trainable_parameters", 0)),
                    best_epoch=int(status.get("best_epoch", -1)),
                    best_validation_mae=float(status.get("best_validation_mae", np.nan)),
                    predictions_path=fold_paths.predictions_path,
                    physics_metrics=status.get("physics_metrics", {}),
                    status="reused", ablation_name=ablation_name,
                )
        except Exception:
            pass

    set_seeds(config.seed)

    outer_train_idx, outer_val_idx = fold_indices(frame, config.fold)
    outer_train_frame = frame.iloc[outer_train_idx].reset_index(drop=True)
    inner_train_local, inner_val_local = inner_split_indices(
        outer_train_frame, seed=config.seed
    )
    inner_train_idx = outer_train_idx[inner_train_local]
    inner_val_idx = outer_train_idx[inner_val_local]

    scaler = FoldScaler().fit(
        outer_train_frame, dataset.feature_columns,
        u_max_source=u_max_source, u_max_constant=u_max_constant,
        u_max_margin=u_max_margin, tolerance_source=tolerance_source,
        tolerance_constant=tolerance_constant, tolerance_quantile=tolerance_quantile,
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
            trainable_parameters=0,  # updated after model construction
            u_max=float(scaler.u_max),
            epsilon_rec=float(scaler.epsilon_rec),
            device=device,
        )

    is_pinn = config.architecture == "NaPINN-Q"
    feature_dim = 2 + len(dataset.feature_columns)  # horizon + u_current + aux
    if is_pinn:
        model = NaPINNQ(
            feature_dim=feature_dim,
            solution_hidden_dims=config.solution_hidden_dims,
            rate_hidden_dims=config.rate_hidden_dims,
            solution_activation=config.solution_activation,
            rate_activation=config.rate_activation,
        )
    elif config.architecture == "DNN-Q":
        model = DNNQ(feature_dim=feature_dim,
                     solution_hidden_dims=config.solution_hidden_dims,
                     solution_activation=config.solution_activation)
    else:
        raise ValueError(f"unknown architecture: {config.architecture}")

    trainable_parameters = count_parameters(model)
    if trainable_parameters > config.maximum_parameters:
        raise ValueError(
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
        "epoch": 0, "best_val": float("inf"), "best_epoch": 0, "patience": 0
    }
    curriculum = CurriculumSchedule(
        max_epochs=config.max_epochs,
        warmup_fraction=config.curriculum_warmup_fraction,
        ramp_end_fraction=config.curriculum_ramp_end_fraction,
    )

    # Build training tensors once (all outer-training rows). Inner train/val
    # are sub-selections by boolean masks.
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

    inner_train_mask = np.isin(outer_train_idx, inner_train_idx)
    inner_val_mask = np.isin(outer_train_idx, inner_val_idx)
    train_selector = torch.as_tensor(inner_train_mask, device=device, dtype=torch.bool)

    epoch_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=device if device == "cpu" else "cpu")
    generator.manual_seed(config.seed)

    if dry_run:
        return FoldResult(
            architecture=config.architecture, preprocessing=config.preprocessing,
            fold=config.fold, seed=config.seed,
            trainable_parameters=trainable_parameters,
            best_epoch=-1, best_validation_mae=float("nan"),
            predictions_path=fold_paths.predictions_path,
            physics_metrics={}, status="dry_run", ablation_name=ablation_name,
        )

    best_val = float(checkpoint_state.get("best_val", float("inf")))
    best_epoch = int(checkpoint_state.get("best_epoch", 0))
    patience = int(checkpoint_state.get("patience", 0))
    start_epoch = int(checkpoint_state.get("epoch", 0))

    manifest = {
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "fold": config.fold,
        "seed": config.seed,
        "device": device,
        "feature_dim": feature_dim,
        "feature_columns": dataset.feature_columns,
        "cell_map": cell_map,
        "trainable_parameters": trainable_parameters,
        "loss_weights": asdict(loss_weights),
        "ablation": ablation_name or "",
        "trainer_config": asdict(config),
    }
    atomic_write_json(manifest, fold_paths.manifest_path)
    atomic_write_json(scaler.state_dict(), fold_paths.scaler_path)
    dataset.audit.to_csv(fold_paths.audit_subset, index=False)
    atomic_write_json({"seed": config.seed, "architecture": config.architecture,
                       "preprocessing": config.preprocessing}, fold_paths.config_snapshot)

    tag = f"[{config.architecture} | {config.preprocessing} | fold {config.fold} | seed {config.seed}]"
    bar = tqdm(range(start_epoch, config.max_epochs), desc=tag, unit="ep",
               initial=start_epoch, total=config.max_epochs, leave=False)
    error_message: str | None = None

    try:
        for epoch in bar:
            model.train()
            optimizer.zero_grad()
            factor = curriculum.factor(epoch)
            effective_weights = loss_weights.effective(factor)

            stress_current_tensor = torch.as_tensor(
                stress_current_np, dtype=torch.float32, device=device,
            ).clone().requires_grad_(True)
            stress_next_tensor = torch.as_tensor(
                stress_next_np, dtype=torch.float32, device=device,
            ).clone().requires_grad_(True)

            if is_pinn:
                u_next_hat = model.solution(stress_next_tensor, features)
                r_next_hat = model.rate(stress_next_tensor, features, u_next_hat)
                u_anchor_hat = model.solution(stress_current_tensor, features)
                r_anchor_hat = model.rate(stress_current_tensor, features, u_anchor_hat)
                du_next = autograd_du_ds(u_next_hat, stress_next_tensor)
                du_current = autograd_du_ds(u_anchor_hat, stress_current_tensor)
                residual_anchor = pde_residual(du_current, r_anchor_hat)
                residual_next = pde_residual(du_next, r_next_hat)
                # Collocation
                collocation = sample_collocation_points(
                    stress_current=stress_current_tensor.detach(),
                    stress_delta=torch.as_tensor(delta_np, dtype=torch.float32, device=device),
                    features=features,
                    cell_index=cell_index,
                    points_per_transition=config.collocation_points,
                    generator=generator if device == "cpu" else None,
                )
                collo_stress = collocation.stress.clone().detach().requires_grad_(True)
                u_collo = model.solution(collo_stress, collocation.features)
                r_collo = model.rate(collo_stress, collocation.features, u_collo)
                du_collo = autograd_du_ds(u_collo, collo_stress)
                residual_collo = pde_residual(du_collo, r_collo)
                # Integral consistency
                u_integrated = integral_transition(
                    model.solution, model.rate,
                    stress_current_tensor, torch.as_tensor(delta_np, dtype=torch.float32, device=device),
                    features, u_current,
                    method=config.quadrature_method, n_nodes=config.quadrature_nodes,
                )
            else:
                u_next_hat = model(stress_next_tensor, features)
                u_anchor_hat = model(stress_current_tensor, features)
                r_next_hat = torch.zeros_like(u_next_hat)
                r_anchor_hat = torch.zeros_like(u_anchor_hat)
                du_next = autograd_du_ds(u_next_hat, stress_next_tensor)
                du_current = autograd_du_ds(u_anchor_hat, stress_current_tensor)
                residual_anchor = torch.zeros_like(u_anchor_hat)
                residual_next = torch.zeros_like(u_next_hat)
                residual_collo = torch.zeros_like(u_next_hat)
                u_integrated = torch.zeros_like(u_next_hat)
                collocation = None

            # Mask to inner-train rows only.
            train_pred = u_next_hat[train_selector]
            train_true = u_true[train_selector]
            train_cells = cell_index[train_selector]
            train_data_loss, data_row = data_loss(
                train_pred, train_true, train_cells, delta=config.huber_delta,
            )

            pde_anchor_loss = residual_anchor.new_zeros(())
            pde_collo_loss = residual_collo.new_zeros(())
            pde_row_mean = 0.0
            pde_anchor_row = 0.0
            pde_collo_row = 0.0
            ic_loss = residual_anchor.new_zeros(())
            ic_row = 0.0
            integral_loss = residual_anchor.new_zeros(())
            integral_row = 0.0
            mono_loss_val = residual_anchor.new_zeros(())
            mono_row = 0.0
            bounds_val = residual_anchor.new_zeros(())
            bounds_row = 0.0
            rate_val = residual_anchor.new_zeros(())
            rate_row = 0.0
            discrete_val = residual_anchor.new_zeros(())
            discrete_row = 0.0

            if is_pinn:
                pde_anchor_loss, pde_anchor_row = pde_loss(residual_anchor[train_selector], train_cells)
                if collocation is not None and residual_collo.numel():
                    collo_cells = collocation.cell_index
                    train_selector_collo = torch.isin(
                        collocation.row_index,
                        torch.nonzero(train_selector).view(-1),
                    )
                    pde_collo_loss, pde_collo_row = pde_loss(
                        residual_collo[train_selector_collo],
                        collo_cells[train_selector_collo],
                    )
                pde_row_mean = 0.5 * (pde_anchor_row + pde_collo_row)
                ic_loss, ic_row = initial_condition_loss(
                    u_anchor_hat[train_selector], u_current[train_selector], train_cells,
                )
                integral_loss, integral_row = integral_consistency_loss(
                    u_next_hat[train_selector], u_integrated[train_selector], train_cells,
                )
                mono_loss_val, mono_row = monotonicity_loss(
                    u_next_hat[train_selector], u_current[train_selector], train_cells,
                    epsilon_rec=scaler.epsilon_rec,
                )
                bounds_val, bounds_row = bounds_loss(
                    u_next_hat[train_selector], train_cells,
                    u_min=0.0, u_max=scaler.u_max,
                )
                rate_val, rate_row = rate_regularization_loss(
                    r_next_hat[train_selector], train_cells,
                )
                if include_discrete_transition:
                    discrete_residual = discrete_state_transition_residual(
                        u_next_hat, u_current, r_anchor_hat,
                        torch.as_tensor(delta_np, dtype=torch.float32, device=device),
                    )
                    discrete_val, discrete_row = discrete_state_transition_loss(
                        discrete_residual[train_selector], train_cells,
                    )

            total_loss = (
                effective_weights.data * train_data_loss
                + effective_weights.pde * (pde_anchor_loss + pde_collo_loss)
                + effective_weights.initial_condition * ic_loss
                + effective_weights.integral * integral_loss
                + effective_weights.monotonicity * mono_loss_val
                + effective_weights.bounds * bounds_val
                + effective_weights.rate_regularization * rate_val
                + effective_weights.discrete_state_transition * discrete_val
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"non-finite total loss at epoch {epoch}: {total_loss.item()}"
                )
            total_loss.backward()
            grad_total = _gradient_norm(model.parameters())
            grad_pde = 0.0
            if is_pinn and effective_weights.pde > 0 and (
                pde_anchor_loss.requires_grad or pde_collo_loss.requires_grad
            ):
                # Approximate physics gradient contribution via a scaled second pass.
                pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            # Validation on inner-validation rows.
            val_selector = torch.as_tensor(inner_val_mask, device=device, dtype=torch.bool)
            with torch.no_grad():
                if is_pinn:
                    val_u = model.solution(stress_next_tensor, features).detach()
                else:
                    val_u = model(stress_next_tensor, features).detach()
                val_true_tensor = u_true[val_selector]
                val_pred = val_u[val_selector]
                val_error = (val_pred - val_true_tensor).cpu().numpy()
                val_mae = float(np.mean(np.abs(val_error))) if val_error.size else float("inf")
                val_rmse = float(np.sqrt(np.mean(val_error ** 2))) if val_error.size else float("inf")
                if val_error.size >= 2:
                    ss_res = float(np.sum((val_error) ** 2))
                    ss_tot = float(np.sum((val_true_tensor.cpu().numpy() - val_true_tensor.cpu().numpy().mean()) ** 2))
                    val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                else:
                    val_r2 = float("nan")

            scheduler.step(val_mae)
            lr_current = float(optimizer.param_groups[0]["lr"])

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
                "train_total_loss": float(total_loss.detach().item()),
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
                "validation_data_loss": float(np.mean(val_error ** 2)) if val_error.size else float("nan"),
                "validation_MAE": val_mae,
                "validation_RMSE": val_rmse,
                "validation_R2": val_r2,
                "pde_residual_MAE": float(np.mean(np.abs(residual_next.detach().cpu().numpy())))
                    if residual_next.numel() else float("nan"),
                "pde_residual_RMSE": float(np.sqrt(np.mean(residual_next.detach().cpu().numpy() ** 2)))
                    if residual_next.numel() else float("nan"),
                "positive_derivative_fraction": float((du_next.detach().cpu().numpy() > 0).mean())
                    if du_next.numel() else float("nan"),
                "monotonicity_violation_fraction": float(
                    ((u_next_hat.detach().cpu().numpy() - u_current_np - scaler.epsilon_rec) > 0).mean()
                ) if u_next_hat.numel() else float("nan"),
                "lower_bound_violation_fraction": float((u_next_hat.detach().cpu().numpy() < 0).mean())
                    if u_next_hat.numel() else float("nan"),
                "upper_bound_violation_fraction": float((u_next_hat.detach().cpu().numpy() > scaler.u_max).mean())
                    if u_next_hat.numel() else float("nan"),
                "effective_lambda_data": float(effective_weights.data),
                "effective_lambda_pde": float(effective_weights.pde),
                "effective_lambda_initial_condition": float(effective_weights.initial_condition),
                "effective_lambda_integral": float(effective_weights.integral),
                "effective_lambda_monotonicity": float(effective_weights.monotonicity),
                "effective_lambda_bounds": float(effective_weights.bounds),
                "effective_lambda_rate": float(effective_weights.rate_regularization),
                "effective_lambda_discrete_transition": float(effective_weights.discrete_state_transition),
                "learning_rate": lr_current,
                "gradient_norm_total": grad_total,
                "gradient_norm_pde": grad_pde,
                "GPU_memory_allocated_MB": allocated_mb,
                "GPU_memory_reserved_MB": reserved_mb,
                "best_validation_MAE": min(best_val, val_mae),
                "best_epoch": best_epoch,
                "early_stopping_counter": patience,
                "ablation": ablation_name or "",
            }
            epoch_rows.append(epoch_row)
            if logger is not None and (epoch % max(1, config.log_every_epochs) == 0
                                        or epoch == config.max_epochs - 1):
                log_epoch_summary(logger, epoch_row)

            if val_mae + 1e-9 < best_val:
                best_val = val_mae
                best_epoch = epoch
                patience = 0
                atomic_write_torch(
                    {"model": model.state_dict(), "epoch": epoch,
                     "best_val": best_val, "best_epoch": best_epoch,
                     "patience": patience},
                    fold_paths.best_model,
                )
            else:
                patience += 1

            bar.set_postfix({
                "total": f"{float(total_loss.detach().item()):.4f}",
                "data": f"{data_row:.4f}",
                "pde": f"{pde_row_mean:.4f}",
                "val_MAE": f"{val_mae:.4f}",
                "best": f"{best_val:.4f}",
                "pat": f"{patience}/{config.early_stopping_patience}",
                "lr": f"{lr_current:.2e}",
            })

            if (epoch + 1) % max(1, config.save_checkpoint_every_epochs) == 0 or epoch == config.max_epochs - 1:
                atomic_write_torch(
                    {"model": model.state_dict(), "epoch": epoch,
                     "best_val": best_val, "best_epoch": best_epoch,
                     "patience": patience},
                    fold_paths.last_model,
                )
                atomic_write_torch(optimizer.state_dict(), fold_paths.optimizer_state)
                atomic_write_torch(scheduler.state_dict(), fold_paths.scheduler_state)

            if patience >= config.early_stopping_patience and epoch >= config.min_epochs:
                break
    except Exception as exc:  # pragma: no cover
        error_message = f"{type(exc).__name__}: {exc}"
        if logger is not None:
            log_failure(logger, "fold_training_failed", exc,
                        architecture=config.architecture,
                        preprocessing=config.preprocessing,
                        fold=config.fold, seed=config.seed)

    bar.close()

    epoch_log = pd.DataFrame(epoch_rows)
    if not epoch_log.empty:
        atomic_write_csv(epoch_log, fold_paths.epoch_log_csv)
        try:
            atomic_write_parquet(epoch_log, fold_paths.epoch_log_parquet)
        except Exception:
            pass

    # Restore best model weights for evaluation.
    if fold_paths.best_model.exists():
        best_state = torch.load(fold_paths.best_model, map_location=device,
                                 weights_only=False)
        model.load_state_dict(best_state["model"])

    # Evaluate on inner-validation, outer-validation.
    inner_eval = _evaluate(model, dataset, frame, inner_val_idx, scaler, device,
                            is_pinn, config, config.quadrature_method,
                            config.quadrature_nodes)
    outer_eval = _evaluate(model, dataset, frame, outer_val_idx, scaler, device,
                            is_pinn, config, config.quadrature_method,
                            config.quadrature_nodes)

    prediction_frames = []
    for indices, split_name, eval_data in (
        (inner_train_idx, "inner_train", None),
        (inner_val_idx, "inner_validation", inner_eval),
        (outer_val_idx, "outer_validation", outer_eval),
    ):
        if eval_data is None:
            eval_data = _evaluate(model, dataset, frame, indices, scaler, device,
                                   is_pinn, config, config.quadrature_method,
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
            ablation_name=ablation_name,
        ))

    predictions_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame()
    )
    if not predictions_frame.empty:
        atomic_write_parquet(predictions_frame, fold_paths.predictions_path)

    physics_summary = {}
    if outer_eval:
        physics_summary = {
            "pde_residual_MAE": float(np.mean(np.abs(outer_eval["pde_residual"]))),
            "pde_residual_RMSE": float(np.sqrt(np.mean(outer_eval["pde_residual"] ** 2))),
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

    atomic_write_json({
        "status": "failed" if error_message else "completed",
        "error_message": error_message,
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "fold": config.fold,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_val,
        "trainable_parameters": trainable_parameters,
        "physics_metrics": physics_summary,
        "ablation": ablation_name or "",
    }, fold_paths.status_path)

    if logger is not None:
        log_event(logger, logging.INFO, "fold_completed",
                  architecture=config.architecture,
                  preprocessing=config.preprocessing,
                  fold=config.fold, seed=config.seed,
                  status="failed" if error_message else "completed",
                  best_epoch=best_epoch,
                  best_validation_mae=best_val,
                  physics_metrics=physics_summary,
                  n_epoch_rows=len(epoch_rows))

    return FoldResult(
        architecture=config.architecture, preprocessing=config.preprocessing,
        fold=config.fold, seed=config.seed,
        trainable_parameters=trainable_parameters,
        best_epoch=best_epoch, best_validation_mae=best_val,
        predictions_path=fold_paths.predictions_path,
        physics_metrics=physics_summary,
        status="failed" if error_message else "completed",
        error_message=error_message,
        ablation_name=ablation_name,
        epoch_log=epoch_rows,
    )
