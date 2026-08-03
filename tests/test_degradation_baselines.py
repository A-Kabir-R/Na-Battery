"""Physics baselines: recover known rates, and never look forward in time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.degradation_baselines import (
    ArrheniusDodRate,
    CellRandomInterceptRate,
    ConstantRate,
    PerConditionRate,
    R_GAS,
    ZeroChange,
    build_degradation_baselines,
    stress_design_diagnostics,
    stress_interval,
)


def _frame(rate_per_efc: float = 1e-4, n_cells: int = 4, n_anchors: int = 5,
           step: float = 40.0) -> tuple[pd.DataFrame, np.ndarray]:
    rows, deltas = [], []
    for c in range(n_cells):
        for a in range(1, n_anchors + 1):
            rows.append({
                "cell": f"C{c}", "condition": "cond", "EFC_cum": step * a,
                "T_degC": 25, "DOD_pct": 100, "C_ch": 1.0, "C_dis": 1.0,
            })
            deltas.append(-rate_per_efc * step)
    return pd.DataFrame(rows), np.asarray(deltas)


def test_stress_interval_is_backward_looking():
    frame, _ = _frame(step=40.0, n_anchors=3, n_cells=1)
    interval = stress_interval(frame)
    # First anchor spans birth (0) to its own block end; the rest are diffs.
    assert np.allclose(interval, [40.0, 40.0, 40.0])


def test_stress_interval_never_crosses_a_cell_boundary():
    frame, _ = _frame(n_cells=2, n_anchors=3, step=40.0)
    interval = stress_interval(frame)
    # If it differenced across cells the first anchor of cell 1 would be
    # 40 - 120 = -80 rather than its own cumulative 40.
    assert (interval > 0).all()
    assert np.allclose(interval, 40.0)


def test_constant_rate_recovers_a_known_rate():
    frame, y = _frame(rate_per_efc=2.5e-4)
    model = ConstantRate().fit(frame, y)
    assert np.isclose(model.fitted_parameters_["k_per_EFC"], 2.5e-4, rtol=1e-9)
    assert np.allclose(model.predict(frame), y)


def test_zero_change_predicts_no_change():
    frame, y = _frame()
    assert np.allclose(ZeroChange().fit(frame, y).predict(frame), 0.0)


def test_per_condition_rate_separates_conditions():
    fast, y_fast = _frame(rate_per_efc=4e-4, n_cells=2)
    slow, y_slow = _frame(rate_per_efc=1e-4, n_cells=2)
    slow["cell"] = slow["cell"] + "_slow"
    slow["condition"] = "slow"
    frame = pd.concat([fast, slow], ignore_index=True)
    y = np.concatenate([y_fast, y_slow])
    model = PerConditionRate().fit(frame, y)
    assert np.isclose(model.rates_["cond"], 4e-4, rtol=1e-6)
    assert np.isclose(model.rates_["slow"], 1e-4, rtol=1e-6)


def test_per_condition_rate_falls_back_for_an_unseen_condition():
    frame, y = _frame()
    model = PerConditionRate().fit(frame, y)
    unseen = frame.copy()
    unseen["condition"] = "never_seen"
    assert np.all(np.isfinite(model.predict(unseen)))
    assert np.allclose(model.predict(unseen), -model.fallback_ * stress_interval(unseen))


def test_arrhenius_recovers_a_planted_activation_energy():
    activation_J = 30_000.0
    k_ref, t_ref = 1e-4, 298.15
    rows, deltas = [], []
    for c, temperature in enumerate([0, 25, 45]):
        rate = k_ref * np.exp(-(activation_J / R_GAS)
                              * (1.0 / (temperature + 273.15) - 1.0 / t_ref))
        for a in range(1, 6):
            rows.append({
                "cell": f"C{c}", "condition": f"T{temperature}",
                "EFC_cum": 40.0 * a, "T_degC": temperature, "DOD_pct": 100,
                "C_ch": 1.0, "C_dis": 1.0,
            })
            deltas.append(-rate * 40.0)
    frame = pd.DataFrame(rows)
    model = ArrheniusDodRate().fit(frame, np.asarray(deltas))
    assert np.isclose(
        model.fitted_parameters_["activation_energy_J_per_mol"],
        activation_J, rtol=1e-4,
    )
    assert np.isclose(model.fitted_parameters_["k_ref_per_EFC"], k_ref, rtol=1e-4)


def test_arrhenius_keeps_capacity_gains_in_the_fit():
    frame, y = _frame(rate_per_efc=1e-4)
    contaminated = y.copy()
    contaminated[0] = +0.02          # a large measurement-noise capacity gain
    model = ArrheniusDodRate().fit(frame, contaminated)

    # Every usable row is in the fit; the gain is counted, not discarded.
    assert model.fitted_parameters_["n_rows_used"] == len(y)
    assert model.fitted_parameters_["n_rows_capacity_gain"] == 1

    # A log-space fit cannot take the logarithm of a negative rate, so it is
    # forced to drop this row and would report k_ref = 1e-4 unchanged. That is
    # selection on the sign of the target and it biases k_ref upward. Keeping
    # the row must pull the estimate down.
    assert model.fitted_parameters_["k_ref_per_EFC"] < 1e-4


def test_arrhenius_collapses_to_the_constant_rate_on_a_rank_one_design():
    # T, DOD and C-rate are all constant here, so every stress column is zero
    # and the law has nothing to explain beyond a single global rate.
    frame, y = _frame(rate_per_efc=1e-4)
    contaminated = y.copy()
    contaminated[0] = +0.02
    model = ArrheniusDodRate().fit(frame, contaminated)
    assert np.isclose(model.fitted_parameters_["k_ref_per_EFC"],
                      ConstantRate().fit(frame, contaminated).k_, rtol=1e-3)


def test_random_intercept_uses_pooled_rate_for_an_unseen_cell():
    frame, y = _frame()
    model = CellRandomInterceptRate().fit(frame, y)
    unseen = frame.copy()
    unseen["cell"] = "brand_new_cell"
    assert np.allclose(
        model.predict(unseen), -model.pooled_k_ * stress_interval(unseen),
    )


def test_shrinkage_is_evidence_dependent_not_a_fixed_half():
    # Cells must genuinely differ, otherwise every shrinkage target coincides
    # with the pooled slope and the knob is unobservable.
    fast, y_fast = _frame(rate_per_efc=4e-4, n_cells=2)
    slow, y_slow = _frame(rate_per_efc=1e-4, n_cells=2)
    slow["cell"] = slow["cell"] + "_slow"
    frame = pd.concat([fast, slow], ignore_index=True)
    y = np.concatenate([y_fast, y_slow])

    weak = CellRandomInterceptRate(shrinkage_strength=1e12).fit(frame, y)
    strong = CellRandomInterceptRate(shrinkage_strength=1e-12).fit(frame, y)

    # Huge prior -> everything collapses to the pooled slope.
    assert all(np.isclose(v, weak.pooled_k_) for v in weak.cell_k_.values())
    # Tiny prior -> each cell keeps its own rate.
    assert np.isclose(strong.cell_k_["C0"], 4e-4, rtol=1e-6)
    assert np.isclose(strong.cell_k_["C0_slow"], 1e-4, rtol=1e-6)
    # A prior proportional to the evidence would give a fixed 50% shrinkage and
    # make these two configurations identical. That was the bug this guards.
    assert not np.isclose(weak.cell_k_["C0"], strong.cell_k_["C0"])


def test_zero_length_block_contributes_no_degradation():
    frame, y = _frame(n_cells=1, n_anchors=3)
    frame.loc[1, "EFC_cum"] = frame.loc[0, "EFC_cum"]   # duplicate stress value
    model = ConstantRate().fit(frame, y)
    assert np.isfinite(model.predict(frame)).all()


def test_registry_exposes_the_constant_rate_baseline():
    registry = build_degradation_baselines()
    assert "ConstantRate" in registry, (
        "the degenerate case of the neural ODE is the one baseline the paper "
        "cannot omit"
    )
    for name, model in registry.items():
        frame, y = _frame()
        model.fit(frame, y)
        assert np.isfinite(model.predict(frame)).all(), name


def test_design_diagnostics_flag_unidentifiable_c_rate():
    # DOD varies but C-rate is constant within every DOD level: the exponents
    # cannot be separated.
    frame, _ = _frame(n_cells=2)
    frame.loc[: len(frame) // 2, "DOD_pct"] = 20
    diagnostics = stress_design_diagnostics(frame)
    assert diagnostics["c_rate_exponent_identifiable"] is False
    assert diagnostics["charge_equals_discharge_rate"] is True


def test_condition_number_is_scaled_before_it_is_reported():
    # A fully crossed, perfectly identifiable design. 1/T is O(3e-3) and the
    # intercept is O(1), so the unscaled condition number is inflated by the
    # ratio of the column norms and would read as catastrophic collinearity
    # against the ~30 threshold. Only the scaled index is interpretable.
    rows = []
    for temperature in (-10, 0, 25, 45):
        for dod in (20, 80, 100):
            for c_rate in (1.0, 2.0):
                rows.append({
                    "cell": f"C{temperature}_{dod}_{c_rate}", "condition": "x",
                    "EFC_cum": 40.0, "T_degC": temperature, "DOD_pct": dod,
                    "C_ch": c_rate, "C_dis": c_rate,
                })
    diagnostics = stress_design_diagnostics(pd.DataFrame(rows))
    assert diagnostics["c_rate_exponent_identifiable"] is True
    assert diagnostics["condition_number"] < 40.0
    assert (diagnostics["condition_number_unscaled"]
            > 10.0 * diagnostics["condition_number"])
