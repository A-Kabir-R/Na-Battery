"""Checkpoint round-trip and physics-gate tests for the trainer."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from src.pinn.losses import CurriculumSchedule
from src.pinn.models import NaPINNQ
from src.pinn.utils import atomic_write_torch

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_atomic_torch_roundtrip(tmp_path: Path) -> None:
    model = NaPINNQ(feature_dim=3, solution_hidden_dims=(8,), rate_hidden_dims=(4,))
    ckpt = tmp_path / "best_model.pt"
    atomic_write_torch({"model": model.state_dict(), "epoch": 42}, ckpt)
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 42
    new_model = NaPINNQ(feature_dim=3, solution_hidden_dims=(8,), rate_hidden_dims=(4,))
    new_model.load_state_dict(payload["model"])
    for (a, va), (b, vb) in zip(model.state_dict().items(),
                                 new_model.state_dict().items()):
        assert a == b
        assert torch.allclose(va, vb)


def test_physics_full_epoch_gates_checkpoint_eligibility() -> None:
    """Epoch physics_full_epoch-1 is still ramping; physics_full_epoch is fully active."""
    curriculum = CurriculumSchedule(max_epochs=500, warmup_fraction=0.05,
                                    ramp_end_fraction=0.20)
    pfe = curriculum.physics_full_epoch
    assert pfe == math.ceil(0.20 * 500)
    assert curriculum.factor(pfe - 1) < 1.0
    assert curriculum.factor(pfe) == 1.0


def test_checkpoint_eligibility_with_configured_curriculum() -> None:
    """Configured curriculum reaches full physics before minimum_epochs."""
    pinn = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["pinn"]
    schedule = CurriculumSchedule(
        max_epochs=int(pinn["training"]["maximum_epochs"]),
        warmup_fraction=float(pinn["curriculum"]["warmup_fraction"]),
        ramp_end_fraction=float(pinn["curriculum"]["ramp_end_fraction"]),
    )
    assert schedule.physics_full_epoch < int(pinn["training"]["minimum_epochs"])
    assert schedule.factor(schedule.physics_full_epoch) == 1.0
    assert schedule.factor(schedule.physics_full_epoch - 1) < 1.0


def test_refit_schedule_derives_from_best_epoch_not_max_epochs() -> None:
    """A refit built from best_epoch+1 reaches full physics on its own schedule."""
    warmup, ramp_end = 0.05, 0.20
    inner = CurriculumSchedule(max_epochs=500, warmup_fraction=warmup,
                               ramp_end_fraction=ramp_end)
    refit = CurriculumSchedule(max_epochs=50, warmup_fraction=warmup,
                               ramp_end_fraction=ramp_end)
    assert refit.physics_full_epoch == math.ceil(ramp_end * 50)
    assert refit.factor(refit.physics_full_epoch) == 1.0
    # With the inner schedule, the same epoch is deep inside the warm-up ramp.
    assert inner.factor(refit.physics_full_epoch) < 1.0
