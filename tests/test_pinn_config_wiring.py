"""Every declared PINN setting must have a runtime effect.

The failure mode this guards against is a configuration key that reads as
enabled but is never passed through to the object that would honour it.
"""
from __future__ import annotations

import inspect
import math

import pytest
import yaml

torch = pytest.importorskip("torch")

from pathlib import Path

from src.pinn.losses import CurriculumSchedule, GradientBalancer, LossWeights
from src.pinn.models import NaPINNQ
from src.pinn.trainer import TrainerConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _pinn_config() -> dict:
    # Read the raw YAML: env-var interpolation is irrelevant to the pinn block.
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["pinn"]


# ---------------------------------------------------------------------------
# TrainerConfig surface
# ---------------------------------------------------------------------------
REQUIRED_TRAINER_FIELDS = (
    "solution_dropout", "rate_dropout", "predict_delta_u", "rate_uses_u_hat",
    "batch_size", "batch_mode", "use_gradient_balance", "gradient_balance_ema",
    "gradient_balance_min_multiplier", "gradient_balance_max_multiplier",
    "checkpoint_after_physics_active", "rate_data_weight", "pairwise_weight",
    "solution_hidden_dims", "rate_hidden_dims", "solution_activation",
    "rate_activation", "two_phase_refit",
)


@pytest.mark.parametrize("field", REQUIRED_TRAINER_FIELDS)
def test_trainer_config_declares_field(field):
    assert field in TrainerConfig.__dataclass_fields__


def test_runner_passes_every_required_field_to_trainer_config():
    """Every required field must be explicitly wired somewhere in the training pipeline.

    06_run_pinn_experiments.py delegates to build_trainer_config(), so the field
    assignment lives in config_builder.py.  Either location counts as explicitly
    wired — relying on TrainerConfig dataclass defaults is not acceptable.
    """
    runner_src = (CONFIG_PATH.parent / "scripts" / "06_run_pinn_experiments.py").read_text(
        encoding="utf-8"
    )
    builder_src = (CONFIG_PATH.parent / "src" / "pinn" / "config_builder.py").read_text(
        encoding="utf-8"
    )
    combined = runner_src + builder_src
    for field in REQUIRED_TRAINER_FIELDS:
        assert f"{field}=" in combined, (
            f"Neither 06_run_pinn_experiments.py nor config_builder.py sets {field}"
        )


def test_config_declares_the_new_loss_and_batching_keys():
    pinn = _pinn_config()
    assert "rate_data" in pinn["losses"]
    assert "pairwise" in pinn["losses"]
    assert pinn["training"]["batch_mode"] in {"cell_balanced", "sequential"}
    assert pinn["training"]["checkpoint_after_physics_active"] is True
    assert pinn["training"]["use_gradient_balance"] is True


def test_config_uses_only_the_unified_preprocessing():
    assert _pinn_config()["preprocessing"] == ["unified"]


def test_config_declares_seed_42_as_primary():
    pinn = _pinn_config()
    assert pinn["primary_seed"] == 42
    assert pinn["seeds"][0] == 42


def test_config_disables_rate_uses_u_hat():
    assert _pinn_config()["model"]["rate_uses_u_hat"] is False


def test_config_enables_residual_parameterisation():
    assert _pinn_config()["model"]["predict_delta_u"] is True


# ---------------------------------------------------------------------------
# Settings must reach the model
# ---------------------------------------------------------------------------
def test_dropout_reaches_the_model_modules():
    model = NaPINNQ(
        feature_dim=5, solution_dropout=0.15, rate_dropout=0.05,
        predict_delta_u=True, rate_uses_u_hat=False,
    )
    solution_dropouts = [
        m.p for m in model.solution.modules() if isinstance(m, torch.nn.Dropout)
    ]
    rate_dropouts = [
        m.p for m in model.rate.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert solution_dropouts and all(p == 0.15 for p in solution_dropouts)
    assert rate_dropouts and all(p == 0.05 for p in rate_dropouts)


def test_dropout_is_active_in_train_mode_and_off_in_eval():
    torch.manual_seed(0)
    model = NaPINNQ(
        feature_dim=5, solution_dropout=0.5, predict_delta_u=True,
        rate_uses_u_hat=False,
    )
    with torch.no_grad():
        for parameter in model.solution.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)
    features = torch.randn(64, 5)
    u_current = torch.ones(64)
    s0 = torch.zeros(64)
    s1 = torch.ones(64)

    model.train()
    a = model.solution(s1, features, u_current, s0)
    b = model.solution(s1, features, u_current, s0)
    assert not torch.allclose(a, b), "dropout had no effect in train mode"

    model.eval()
    c = model.solution(s1, features, u_current, s0)
    d = model.solution(s1, features, u_current, s0)
    assert torch.allclose(c, d), "dropout still active in eval mode"


def test_hidden_dims_reach_the_model():
    model = NaPINNQ(
        feature_dim=5, solution_hidden_dims=(7, 9), rate_hidden_dims=(3,),
        predict_delta_u=True, rate_uses_u_hat=False,
    )
    solution_widths = [
        m.out_features for m in model.solution.modules()
        if isinstance(m, torch.nn.Linear)
    ]
    rate_widths = [
        m.out_features for m in model.rate.modules()
        if isinstance(m, torch.nn.Linear)
    ]
    assert solution_widths == [7, 9, 1]
    assert rate_widths == [3, 1]


def test_phase1_and_refit_models_have_identical_architecture():
    """The refit must rebuild the same architecture from the same config."""
    kwargs = dict(
        feature_dim=11, solution_hidden_dims=(16, 16), rate_hidden_dims=(8, 8),
        solution_activation="tanh", rate_activation="tanh",
        solution_dropout=0.15, rate_dropout=0.05,
        rate_uses_u_hat=False, predict_delta_u=True,
    )
    a, b = NaPINNQ(**kwargs), NaPINNQ(**kwargs)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sa == sb
    assert a.predict_delta_u == b.predict_delta_u
    assert a.rate.uses_u_hat == b.rate.uses_u_hat


def test_every_solution_call_site_passes_the_anchor_stress():
    """No ``.solution(...)`` call in the trainer may omit ``stress_current``.

    Parsed with AST rather than line matching, because nearly every call in the
    trainer spans multiple lines and a textual check would silently skip them —
    which is exactly the class of omission this test exists to catch.
    """
    import ast

    trainer = CONFIG_PATH.parent / "src" / "pinn" / "trainer.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute) and function.attr == "solution"):
            continue
        n_arguments = len(node.args) + len(node.keywords)
        if n_arguments < 4:
            offenders.append(f"line {node.lineno}: {n_arguments} arguments")
    assert offenders == [], (
        f"solution() calls missing u_current/stress_current: {offenders}"
    )


def test_every_solution_call_site_was_actually_found():
    """Guard the guard: if the AST scan finds nothing, the test above is vacuous."""
    import ast

    trainer = CONFIG_PATH.parent / "src" / "pinn" / "trainer.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "solution"
    ]
    # Training next-state + anchor, collocation, validation, and the same four
    # again in the refit, plus the evaluation path.
    assert len(calls) >= 8


# ---------------------------------------------------------------------------
# Curriculum / loss-weight plumbing
# ---------------------------------------------------------------------------
def test_loss_weights_round_trip_the_new_components():
    weights = LossWeights.from_mapping({
        "data": 1.0, "pde": 0.01, "rate_data": 0.05, "pairwise": 0.02,
    })
    assert weights.rate_data == 0.05
    assert weights.pairwise == 0.02
    assert "rate_data" in weights.as_dict()


def test_loss_weights_pde_alias_maps_to_ode():
    """Legacy 'pde' key must be accepted and stored under the 'ode' field."""
    weights = LossWeights.from_mapping({"data": 1.0, "pde": 0.07})
    assert weights.ode == pytest.approx(0.07)
    assert weights.pde == pytest.approx(0.07)   # property alias
    assert "ode" in weights.as_dict()
    assert "pde" not in weights.as_dict()        # dict uses canonical name


def test_loss_weights_ode_key_accepted():
    """New 'ode' key must also be accepted (takes precedence over 'pde')."""
    weights = LossWeights.from_mapping({"data": 1.0, "ode": 0.03, "pde": 0.07})
    assert weights.ode == pytest.approx(0.03)    # ode wins


def test_loss_weights_residual_share_defaults_to_zero():
    weights = LossWeights.from_mapping({"data": 1.0, "ode": 0.01})
    assert weights.residual_share == pytest.approx(0.0)
    assert "residual_share" in weights.as_dict()


def test_residual_share_is_not_curriculum_ramped():
    """Like rate_data and pairwise, it is a data-like penalty."""
    weights = LossWeights.from_mapping({
        "data": 1.0, "ode": 0.5, "residual_share": 0.05,
    })
    effective = weights.effective(0.0)
    assert effective.residual_share == pytest.approx(0.05)


def test_rate_data_and_pairwise_are_not_curriculum_ramped():
    """They are data-like supervision, so they must be active from epoch 0."""
    weights = LossWeights.from_mapping({
        "data": 1.0, "pde": 0.5, "rate_data": 0.05, "pairwise": 0.02,
    })
    effective = weights.effective(0.0)
    assert effective.rate_data == 0.05
    assert effective.pairwise == 0.02
    assert effective.pde == 0.0


def test_curriculum_exposes_physics_full_epoch():
    schedule = CurriculumSchedule(
        max_epochs=500, warmup_fraction=0.05, ramp_end_fraction=0.20,
    )
    assert schedule.warmup_epochs == 25
    assert schedule.physics_full_epoch == 100
    assert schedule.factor(99) < 1.0
    assert schedule.factor(100) == 1.0


def test_configured_curriculum_activates_before_minimum_epochs():
    pinn = _pinn_config()
    schedule = CurriculumSchedule(
        max_epochs=int(pinn["training"]["maximum_epochs"]),
        warmup_fraction=float(pinn["curriculum"]["warmup_fraction"]),
        ramp_end_fraction=float(pinn["curriculum"]["ramp_end_fraction"]),
    )
    # physics_full_epoch=150 < minimum_epochs=350: physics is fully active well
    # before early stopping can fire, guaranteeing a PINN checkpoint.
    assert schedule.physics_full_epoch < int(pinn["training"]["minimum_epochs"])


def test_gradient_balancer_clips_and_survives_degenerate_gradients():
    balancer = GradientBalancer(ema=0.9, min_multiplier=0.1, max_multiplier=10.0)
    balancer.observe("data", 1.0)
    balancer.observe("pde", 1.0e-12)          # would imply a huge multiplier
    assert balancer.multiplier("pde") == 10.0
    balancer.observe("integral", 1.0e12)      # would imply a tiny multiplier
    assert balancer.multiplier("integral") == 0.1

    # Zero, negative and non-finite observations must be ignored, not stored.
    before = balancer.multiplier("pde")
    balancer.observe("pde", 0.0)
    balancer.observe("pde", float("nan"))
    balancer.observe("pde", float("inf"))
    assert balancer.multiplier("pde") == before


def test_gradient_balancer_multiplier_is_one_when_unobserved():
    balancer = GradientBalancer()
    assert balancer.multiplier("pde") == 1.0
    assert balancer.multiplier("data") == 1.0     # data is never rescaled


def test_gradient_balancer_state_is_finite_and_logged():
    balancer = GradientBalancer()
    balancer.observe("data", 2.0)
    balancer.observe("pde", 4.0)
    state = balancer.state()
    assert "adaptive_multiplier_pde" in state
    assert all(math.isfinite(v) for v in state.values())
    assert math.isclose(state["adaptive_multiplier_pde"], 0.5, rel_tol=1e-6)
