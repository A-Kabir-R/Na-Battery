"""NaPINN-Q: autograd-based physics-informed neural network for sodium-ion
capacity degradation forecasting.

Public entry points:
    - build_anchor_dataset (dataset.py): construct leakage-audited anchor rows
    - SolutionNet, RateNet (models.py): the two Tanh/Softplus networks
    - autograd_du_ds, pde_residual (physics.py): governing residual
    - sample_collocation_points (collocation.py)
    - PhysicsLossBundle, huber_data_loss (losses.py)
    - train_fold (trainer.py): one architecture x preprocessing x fold x seed fit
    - evaluate_fold, physics_metrics (evaluation.py)
"""
from __future__ import annotations

__all__ = [
    "MODEL_NAME",
]

MODEL_NAME = "NaPINN-Q"
