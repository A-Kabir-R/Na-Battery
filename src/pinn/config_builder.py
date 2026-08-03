"""Build a :class:`TrainerConfig` from the ``pinn`` config block.

Extracted so that every entry point -- the main sweep (scripts/06), the ablation
runner (scripts/10) and nested selection (scripts/15) -- constructs the trainer
from one place. A duplicated 60-line mapping would drift, and a selection run
whose hyperparameters silently differ from the run it is selecting *for* would
invalidate the whole nesting argument without failing anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .losses import LossWeights
from .trainer import TrainerConfig


def build_trainer_config(pinn_cfg: dict[str, Any], *, architecture: str,
                         preprocessing: str, fold: int, seed: int,
                         max_epochs: int | None = None,
                         device: str | None = None,
                         **overrides: Any) -> TrainerConfig:
    """Map the ``pinn`` config block onto a :class:`TrainerConfig`.

    ``overrides`` sets fields after the config is applied, which is how the
    nested-selection runner switches ``inner_fold``, disables the refit for its
    phase-1 passes and injects ``forced_best_epoch`` for the final one.
    """
    training = pinn_cfg["training"]
    model = pinn_cfg["model"]
    losses = pinn_cfg["losses"]
    logging_cfg = pinn_cfg["logging"]

    config = TrainerConfig(
        architecture=architecture,
        preprocessing=preprocessing,
        fold=fold,
        seed=seed,
        max_epochs=int(max_epochs if max_epochs is not None
                       else training["maximum_epochs"]),
        min_epochs=int(training["minimum_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        scheduler_factor=float(training["scheduler_factor"]),
        scheduler_patience=int(training["scheduler_patience"]),
        minimum_learning_rate=float(training["minimum_learning_rate"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        use_amp=bool(training["use_amp"]),
        physics_float32=bool(training["physics_float32"]),
        curriculum_warmup_fraction=float(pinn_cfg["curriculum"]["warmup_fraction"]),
        curriculum_ramp_end_fraction=float(pinn_cfg["curriculum"]["ramp_end_fraction"]),
        huber_delta=float(losses["huber_delta"]),
        collocation_points=int(pinn_cfg["collocation"]["points_per_transition"]),
        quadrature_method=str(pinn_cfg["quadrature"]["method"]),
        quadrature_nodes=int(pinn_cfg["quadrature"]["nodes"]),
        solution_hidden_dims=tuple(int(h) for h in model["solution_hidden_dims"]),
        rate_hidden_dims=tuple(int(h) for h in model["rate_hidden_dims"]),
        solution_activation=str(model["solution_activation"]),
        rate_activation=str(model["rate_activation"]),
        solution_dropout=float(model.get("solution_dropout", 0.0)),
        rate_dropout=float(model.get("rate_dropout", 0.0)),
        rate_uses_u_hat=bool(model.get("rate_uses_u_hat", True)),
        predict_delta_u=bool(model.get("predict_delta_u", False)),
        batch_size=int(training.get("batch_size", 0)),
        batch_mode=str(training.get("batch_mode", "cell_balanced")),
        maximum_parameters=int(model["maximum_parameters"]),
        inner_split_seed=int(
            pinn_cfg.get("audit", {}).get("inner_split_seed", 20240117)
        ),
        two_phase_refit=bool(training.get("two_phase_refit", True)),
        pde_gradient_min_norm=float(training.get("pde_gradient_min_norm", 1.0e-8)),
        pde_gradient_zero_patience=int(
            training.get("pde_gradient_zero_patience", 5)
        ),
        use_gradient_balance=bool(training.get("use_gradient_balance", True)),
        gradient_balance_ema=float(training.get("gradient_balance_ema", 0.95)),
        gradient_balance_min_multiplier=float(
            training.get("gradient_balance_min_multiplier", 0.1)
        ),
        gradient_balance_max_multiplier=float(
            training.get("gradient_balance_max_multiplier", 10.0)
        ),
        checkpoint_after_physics_active=bool(
            training.get("checkpoint_after_physics_active", True)
        ),
        rate_data_weight=float(losses.get("rate_data", 0.0)),
        pairwise_weight=float(losses.get("pairwise", 0.0)),
        pairwise_max_pairs=int(training.get("pairwise_max_pairs", 256)),
        log_every_epochs=int(logging_cfg["log_every_epochs"]),
        save_checkpoint_every_epochs=int(logging_cfg["save_checkpoint_every_epochs"]),
        log_gpu_memory=bool(logging_cfg["log_gpu_memory"]),
        log_gradient_norms=bool(logging_cfg["log_gradient_norms"]),
        device=device,
        use_hybrid_rate=bool(model.get("use_hybrid_rate", False)),
        hybrid_temperature_feature=str(
            model.get("hybrid_temperature_feature", "T_degC")
        ),
        hybrid_dod_feature=str(model.get("hybrid_dod_feature", "DOD_pct")),
        hybrid_c_rate_feature=str(model.get("hybrid_c_rate_feature", "")),
        hybrid_enable_cold_regime=bool(
            model.get("hybrid_enable_cold_regime", True)
        ),
        hybrid_fit_c_rate_exponent=bool(
            model.get("hybrid_fit_c_rate_exponent", False)
        ),
        hybrid_residual_share_limit=float(
            model.get("hybrid_residual_share_limit", 0.5)
        ),
    )
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"TrainerConfig has no field {key!r}")
        setattr(config, key, value)
    return config


def build_train_fold_kwargs(pinn_cfg: dict[str, Any], cfg: dict[str, Any], *,
                            preprocessing: str,
                            features_dir: Path) -> dict[str, Any]:
    """Bounds, tolerances and fingerprint feeds that ``train_fold`` requires.

    Kept beside the config builder so the two cannot drift apart either.
    """
    return {
        "loss_weights": LossWeights.from_mapping(pinn_cfg["losses"]),
        "u_max_source": str(pinn_cfg["bounds"]["u_max_source"]),
        "u_max_constant": float(pinn_cfg["bounds"]["u_max_constant"]),
        "u_max_margin": float(pinn_cfg["bounds"]["u_max_margin"]),
        "tolerance_source": str(pinn_cfg["monotonicity"]["tolerance_source"]),
        "tolerance_constant": float(pinn_cfg["monotonicity"]["tolerance_constant"]),
        "tolerance_quantile": float(pinn_cfg["monotonicity"]["tolerance_quantile"]),
        "feature_file": features_dir / f"{preprocessing}.parquet",
        "split_manifest_file": (
            Path(cfg["paths"]["artifacts"]) / "splits" / "split_manifest.parquet"
        ),
    }
