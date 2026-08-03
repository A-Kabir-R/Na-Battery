"""Input perturbations for robustness reporting.

A model that degrades gracefully is not the same as one that degrades
*honestly*. The requirement the paper should hold itself to is that prediction
intervals **widen** as inputs degrade -- a model whose accuracy falls while its
confidence does not is worse than useless in deployment, because nothing signals
that its output should be distrusted.

Perturbations act on the built feature table rather than on raw ``.ird`` signals.
That is a deliberate scope limit: a current-offset applied here propagates only
through the features that already encode current, not through re-integration of
the raw waveform. Sensor perturbations that require re-integration belong in the
canonical rebuild and are labelled ``requires_raw=True`` so the manuscript can
state which class of error was actually tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Perturbation:
    """One named input corruption."""

    name: str
    apply: Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]
    description: str
    requires_raw: bool = False

    def __call__(self, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        return self.apply(frame.copy(), rng)


def _scale_columns(frame: pd.DataFrame, columns: list[str], factor: float) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column] * factor
    return frame


def _offset_columns(frame: pd.DataFrame, columns: list[str], offset: float) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column] + offset
    return frame


def _noise_columns(frame: pd.DataFrame, columns: list[str], sigma: float,
                   rng: np.random.Generator) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=float)
            frame[column] = values + rng.normal(0.0, sigma, size=values.shape)
    return frame


#: Feature families, for the "missing diagnostics" perturbation.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "delta_q_v": ("dqv_logvar", "dqv_min", "dqv_l2", "dqv_skew",
                  "dqv_v_at_max_abs_dev", "dqv_overlap_span_V"),
    "incremental_capacity": ("icdv_dis_peak1_V_shift",
                             "icdv_dis_peak1_height_change"),
    "thermal": ("thermal_discharge_rise_degC", "thermal_max_dTdt_degC_s",
                "thermal_rise_per_Ah"),
    "resistance": ("dcir_prev_rpt_discharge_ohm",
                   "dcir_prev_rpt_delta_from_baseline_ohm"),
    "temporal": ("Q_discharge_Ah_r10_slope", "Q_discharge_Ah_r10_std",
                 "Q_discharge_Ah_dev_r10", "coulombic_efficiency_r10_mean",
                 "coulombic_efficiency_r10_slope"),
}

#: Availability flags paired with each family, so a masked family also reports
#: itself as unavailable rather than silently arriving as an imputed median.
FAMILY_FLAGS: dict[str, str] = {
    "delta_q_v": "dqv_available",
    "incremental_capacity": "icdv_available",
    "resistance": "dcir_available",
    "temporal": "temporal_window_available",
}


def drop_feature_family(family: str) -> Perturbation:
    """Blank a whole diagnostic family and clear its availability flag."""
    columns = list(FEATURE_FAMILIES[family])
    flag = FAMILY_FLAGS.get(family)

    def apply(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        for column in columns:
            if column in frame.columns:
                frame[column] = np.nan
        if flag and flag in frame.columns:
            frame[flag] = 0.0
        return frame

    return Perturbation(
        name=f"missing_{family}",
        apply=apply,
        description=f"{family} diagnostics unavailable (flag cleared)",
    )


def build_perturbations(*, temperature_offset_C: float = 2.0,
                        voltage_noise_V: float = 0.005,
                        current_gain: float = 1.02,
                        capacity_noise_Ah: float = 0.005,
                        resistance_gain: float = 1.10) -> list[Perturbation]:
    """Default perturbation battery.

    Magnitudes are instrument-plausible rather than adversarial: a 2 C sensor
    bias, a 5 mV voltage error, a 2% current gain error. Report the level with
    the result -- "robust to noise" means nothing without the amplitude.
    """
    return [
        Perturbation(
            "temperature_offset",
            lambda f, rng: _offset_columns(
                f, ["T_degC", "T_max"], temperature_offset_C),
            f"thermocouple bias of +{temperature_offset_C} C",
        ),
        Perturbation(
            "temperature_noise",
            lambda f, rng: _noise_columns(
                f, ["T_max", "temperature_span_C"], temperature_offset_C / 2, rng),
            "random thermal measurement noise",
        ),
        Perturbation(
            "voltage_noise",
            lambda f, rng: _noise_columns(
                f, ["voltage_hysteresis_V", "dqv_v_at_max_abs_dev"],
                voltage_noise_V, rng),
            f"{voltage_noise_V * 1000:.0f} mV voltage noise on curve features",
        ),
        Perturbation(
            "current_gain",
            lambda f, rng: _scale_columns(
                f, ["Q_discharge_Ah", "EFC_cum", "cumulative_Ah_throughput"],
                current_gain),
            f"{(current_gain - 1) * 100:.0f}% current gain error",
        ),
        Perturbation(
            "capacity_noise",
            lambda f, rng: _noise_columns(
                f, ["previous_rpt_Q_Ah"], capacity_noise_Ah, rng),
            f"{capacity_noise_Ah * 1000:.0f} mAh anchor-capacity noise",
        ),
        Perturbation(
            "resistance_gain",
            lambda f, rng: _scale_columns(
                f, ["dcir_prev_rpt_discharge_ohm"], resistance_gain),
            f"{(resistance_gain - 1) * 100:.0f}% DCIR scale error",
        ),
        Perturbation(
            "condition_metadata_masked",
            # Every nameplate condition field collapses to its median, i.e. the
            # deployment case where the operating condition is not recorded.
            # DOD must be included: it is the strongest condition signal in the
            # dataset, and an earlier version offset it by 0.0, which is a no-op.
            lambda f, rng: f.assign(**{
                column: f[column].median()
                for column in ("DOD_pct", "C_ch", "C_dis", "T_degC")
                if column in f.columns
            }),
            "nominal condition metadata (DOD, C-rate, T) replaced by its median",
        ),
        *[drop_feature_family(family) for family in FEATURE_FAMILIES],
    ]


def run_robustness(frame: pd.DataFrame, target: str,
                   predict: Callable[[pd.DataFrame], np.ndarray],
                   *, perturbations: list[Perturbation] | None = None,
                   interval_half_width: Callable[[pd.DataFrame], np.ndarray] | None = None,
                   seed: int = 0) -> pd.DataFrame:
    """Score a fitted model under each perturbation.

    ``predict`` must be a *fitted* model's prediction function: the point of the
    study is that the deployed model meets corrupted inputs, not that it is
    retrained on them.

    When ``interval_half_width`` is supplied the report includes whether the
    intervals widened, which is the property that actually matters.
    """
    perturbations = perturbations or build_perturbations()
    rng = np.random.default_rng(seed)
    y = frame[target].to_numpy(dtype=float)

    baseline_prediction = np.asarray(predict(frame), dtype=float)
    baseline_mae = float(np.nanmean(np.abs(y - baseline_prediction)))
    baseline_width = (
        float(np.nanmean(interval_half_width(frame)))
        if interval_half_width is not None else np.nan
    )

    rows = [{
        "perturbation": "none", "description": "unperturbed reference",
        "requires_raw": False, "MAE": baseline_mae, "MAE_ratio": 1.0,
        "mean_half_width": baseline_width, "width_ratio": 1.0,
        "widened": True, "status": "ok", "error": "",
    }]

    for perturbation in perturbations:
        try:
            perturbed = perturbation(frame, rng)
            prediction = np.asarray(predict(perturbed), dtype=float)
            mae = float(np.nanmean(np.abs(y - prediction)))
            width = (
                float(np.nanmean(interval_half_width(perturbed)))
                if interval_half_width is not None else np.nan
            )
            rows.append({
                "perturbation": perturbation.name,
                "description": perturbation.description,
                "requires_raw": perturbation.requires_raw,
                "MAE": mae,
                "MAE_ratio": mae / baseline_mae if baseline_mae > 0 else np.nan,
                "mean_half_width": width,
                "width_ratio": (
                    width / baseline_width
                    if interval_half_width is not None and baseline_width > 0
                    else np.nan
                ),
                # The property to report: did the model admit it was less sure?
                "widened": (
                    bool(width >= baseline_width)
                    if interval_half_width is not None else None
                ),
                "status": "ok",
                "error": "",
            })
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            rows.append({
                "perturbation": perturbation.name,
                "description": perturbation.description,
                "requires_raw": perturbation.requires_raw,
                "MAE": np.nan, "MAE_ratio": np.nan, "mean_half_width": np.nan,
                "width_ratio": np.nan, "widened": None,
                "status": "failed", "error": str(error),
            })
    return pd.DataFrame(rows)
