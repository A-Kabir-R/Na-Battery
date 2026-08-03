"""The hybrid rate must stay physical, decomposable and identifiable."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.degradation_laws import (  # noqa: E402
    R_GAS,
    HybridRateModel,
    ParametricStressRate,
    RateComponents,
    StressLawConfig,
    parameter_prior_penalty,
    residual_share_penalty,
)


def _law(**kwargs) -> ParametricStressRate:
    return ParametricStressRate(StressLawConfig(**kwargs))


def _inputs(n: int = 8, temperature: float = 25.0):
    return {
        "temperature_K": torch.full((n,), temperature + 273.15),
        "dod_fraction": torch.ones(n),
        "c_rate": torch.ones(n),
    }


def test_every_component_is_nonnegative():
    law = _law()
    cycling, cold, calendar = law(**_inputs())
    for component in (cycling, cold, calendar):
        assert (component >= 0).all()


def test_parameters_stay_inside_their_physical_bounds():
    law = _law()
    # Drive the unconstrained parameters to extremes; the squashing must hold.
    with torch.no_grad():
        for parameter in law.parameters():
            parameter.fill_(1e3)
    low, high = law.config.activation_energy_bounds_J_mol
    assert low <= float(law.activation_energy()) <= high
    with torch.no_grad():
        for parameter in law.parameters():
            parameter.fill_(-1e3)
    assert low <= float(law.activation_energy()) <= high
    beta_low, beta_high = law.config.dod_exponent_bounds
    assert beta_low <= float(law.dod_exponent()) <= beta_high


def test_c_rate_exponent_is_frozen_at_zero_by_default():
    """DOD and C-rate are confounded here, and the source paper reports no
    significant C-rate effect, so a free exponent would fit an artefact."""
    law = _law()
    assert float(law.c_rate_exponent()) == 0.0
    fast, _, _ = law(**{**_inputs(), "c_rate": torch.full((8,), 4.0)})
    slow, _, _ = law(**{**_inputs(), "c_rate": torch.full((8,), 0.25)})
    assert torch.allclose(fast, slow), "frozen exponent must make C-rate inert"


def test_c_rate_exponent_can_be_enabled_as_an_explicit_test():
    law = _law(fit_c_rate_exponent=True)
    with torch.no_grad():
        law.raw_c_exponent.fill_(2.0)
    assert float(law.c_rate_exponent()) > 0.0


def test_arrhenius_direction_is_correct():
    law = _law()
    with torch.no_grad():
        law.raw_activation.fill_(3.0)      # a clearly positive activation energy
    assert float(law.activation_energy()) > 0
    hot, _, _ = law(**_inputs(temperature=45.0))
    cold, _, _ = law(**_inputs(temperature=5.0))
    assert (hot > cold).all(), "positive activation energy must accelerate with heat"


def test_cold_regime_activates_only_below_the_transition():
    law = _law()
    with torch.no_grad():
        law.raw_cold_temperature.fill_(0.0)     # mid-range transition
        law.log_cold_scale.fill_(-2.0)
        law.log_cold_width.fill_(0.0)
    transition = float(law.cold_transition_temperature())
    _, cold_below, _ = law(**_inputs(temperature=transition - 273.15 - 30.0))
    _, cold_above, _ = law(**_inputs(temperature=transition - 273.15 + 30.0))
    assert (cold_below > cold_above).all()


def test_cold_regime_can_be_disabled():
    law = _law(enable_cold_regime=False)
    _, cold, _ = law(**_inputs(temperature=-40.0))
    assert torch.allclose(cold, torch.zeros_like(cold))


def test_calendar_term_is_inert_unless_enabled_and_supplied():
    law = _law(enable_calendar=False)
    _, _, calendar = law(**_inputs(), elapsed_time=torch.ones(8))
    assert torch.allclose(calendar, torch.zeros_like(calendar))
    enabled = _law(enable_calendar=True)
    _, _, active = enabled(**_inputs(), elapsed_time=torch.ones(8))
    assert (active > 0).all()


def test_components_sum_to_total():
    components = RateComponents(
        cycling=torch.tensor([1.0]), cold=torch.tensor([2.0]),
        calendar=torch.tensor([3.0]), diagnostic=torch.tensor([4.0]),
        residual=torch.tensor([5.0]),
    )
    assert float(components.total) == pytest.approx(15.0)
    assert float(components.residual_fraction()) == pytest.approx(5.0 / 15.0)


def _hybrid(**kwargs) -> HybridRateModel:
    return HybridRateModel(
        feature_dim=5, temperature_index=0, dod_index=1, c_rate_index=2,
        diagnostic_indices=(3, 4), **kwargs,
    )


def _features(n: int = 6) -> torch.Tensor:
    features = torch.zeros(n, 5)
    features[:, 0] = 25.0        # temperature, degC
    features[:, 1] = 100.0       # DOD, percent
    features[:, 2] = 1.0         # C-rate
    features[:, 3] = 0.05        # a diagnostic
    features[:, 4] = -0.01       # another
    return features


def test_hybrid_rate_is_nonnegative_and_finite():
    model = _hybrid()
    rate = model(torch.linspace(0.0, 1.0, 6), _features())
    assert (rate >= 0).all()
    assert torch.isfinite(rate).all()


def test_residual_starts_negligible_so_physics_must_be_argued_away():
    model = _hybrid()
    components = model.components(torch.linspace(0.0, 1.0, 6), _features())
    assert float(components.residual_fraction().mean()) < 0.25, (
        "a hybrid that starts as a black box never has to learn the physics"
    )


def test_residual_penalty_is_one_sided():
    small = RateComponents(*(torch.tensor([1.0]) for _ in range(4)),
                           residual=torch.tensor([0.01]))
    assert float(residual_share_penalty(small, limit=0.5)) == pytest.approx(0.0)
    large = RateComponents(
        cycling=torch.tensor([0.01]), cold=torch.zeros(1), calendar=torch.zeros(1),
        diagnostic=torch.zeros(1), residual=torch.tensor([10.0]),
    )
    assert float(residual_share_penalty(large, limit=0.5)) > 0.0


def test_gradients_reach_the_physical_parameters():
    model = _hybrid()
    rate = model(torch.linspace(0.0, 1.0, 6), _features())
    rate.sum().backward()
    assert model.parametric.log_k_ref.grad is not None
    assert torch.isfinite(model.parametric.log_k_ref.grad).all()
    assert model.parametric.raw_dod_exponent.grad is not None


def test_physical_parameters_are_reportable():
    model = _hybrid()
    parameters = model.physical_parameters()
    for key in ("k_ref_per_stress", "activation_energy_J_per_mol",
                "dod_exponent_beta", "c_rate_exponent_alpha",
                "cold_transition_temperature_K"):
        assert key in parameters
        assert parameters[key] == parameters[key]   # not NaN
    assert parameters["c_rate_exponent_fitted"] is False


def test_prior_penalty_is_finite_and_differentiable():
    model = _hybrid()
    penalty = parameter_prior_penalty(model)
    penalty.backward()
    assert torch.isfinite(penalty)


def test_activation_energy_units_round_trip():
    """A planted activation energy must reproduce the Arrhenius ratio."""
    law = _law()
    with torch.no_grad():
        law.raw_activation.fill_(0.0)
    activation = float(law.activation_energy())
    t_hot, t_cold = 318.15, 278.15
    hot, _, _ = law(temperature_K=torch.tensor([t_hot]),
                    dod_fraction=torch.ones(1), c_rate=torch.ones(1))
    cold, _, _ = law(temperature_K=torch.tensor([t_cold]),
                     dod_fraction=torch.ones(1), c_rate=torch.ones(1))
    expected = torch.exp(
        torch.tensor(-(activation / R_GAS) * (1 / t_hot - 1 / t_cold))
    )
    assert float(hot / cold) == pytest.approx(float(expected), rel=1e-5)
