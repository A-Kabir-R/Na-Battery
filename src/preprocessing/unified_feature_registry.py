"""Fixed, version-controlled feature registry for the ``unified`` preprocessing.

Every retained model (Ridge, ExtraTrees, NaPINN-Q) consumes exactly the columns
declared here, in exactly this order. There is no per-model subset selection and
no target-driven selection anywhere in the pipeline: the list below was fixed by
physical relevance, availability and redundancy before any fold was fitted.

Dimensionality
--------------
The development set has 175 anchors / 26 cells, i.e. ~140 training rows per
outer fold. The registry is therefore deliberately compact (32 continuous
features + 4 availability flags = 36 columns). A 64-column matrix would put the
NaPINN-Q solution head at >1000 first-layer parameters on 140 samples, which
contradicts the compact-architecture requirement.

Roles
-----
Three continuous features carry a second, physics-specific role:

``EFC_cum``            -> ``stress_coordinate``  (NaPINN-Q differentiable coord)
``previous_rpt_Q_Ah``  -> ``initial_state``      (NaPINN-Q u_current)

For NaPINN-Q these must NOT be duplicated into the auxiliary feature tensor, so
:func:`model_feature_columns` accepts ``exclude_roles``. Classical models use the
full list, including the role-carrying columns, as ordinary features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

REGISTRY_VERSION = "unified-v1"

#: Hard ceiling from the revision plan. Exceeding it is a build-time failure.
MAX_MODEL_FEATURES = 64

FeatureKind = Literal["continuous", "flag"]
FeatureRole = Literal["feature", "stress_coordinate", "initial_state"]


@dataclass(frozen=True)
class FeatureSpec:
    """One registered model input column."""

    name: str
    group: str
    unit: str
    description: str
    kind: FeatureKind = "continuous"
    role: FeatureRole = "feature"
    #: Cycle-level columns this feature is derived from, for the temporal audit.
    sources: tuple[str, ...] = field(default_factory=tuple)
    #: Reference cycle window relative to the anchor. All values must be
    #: backward-looking; the audit rejects anything else.
    reference: str = "anchor_cycle"

    @property
    def is_flag(self) -> bool:
        return self.kind == "flag"


# ---------------------------------------------------------------------------
# Group A — battery state and stress conditions
# ---------------------------------------------------------------------------
_GROUP_A: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "previous_rpt_Q_Ah", "state", "Ah",
        "Capacity measured at the previous (already completed) RPT visit. "
        "NaPINN-Q initial state u_current.",
        role="initial_state", sources=("previous_rpt_Q_Ah",),
        reference="previous_rpt",
    ),
    FeatureSpec(
        "previous_rpt_SOH_pct", "state", "%",
        "Previous-RPT capacity relative to the cell's first valid RPT. "
        "Encodes initial_rpt_Q_Ah given previous_rpt_Q_Ah, so the baseline "
        "capacity is not registered separately.",
        sources=("previous_rpt_SOH_pct",), reference="previous_rpt",
    ),
    FeatureSpec(
        "EFC_cum", "state", "equivalent_full_cycles",
        "Cumulative equivalent full cycles at the anchor. Canonical stress "
        "coordinate; cumulative_Ah_throughput is a scalar multiple and is "
        "deliberately not registered.",
        role="stress_coordinate", sources=("Q_discharge_Ah",),
    ),
    FeatureSpec(
        "DOD_pct", "condition", "%", "Nominal depth of discharge of the block.",
        sources=("DOD_pct",), reference="protocol_constant",
    ),
    FeatureSpec(
        "C_ch", "condition", "1/h", "Nominal charge C-rate of the block.",
        sources=("C_ch",), reference="protocol_constant",
    ),
    FeatureSpec(
        "C_dis", "condition", "1/h", "Nominal discharge C-rate of the block.",
        sources=("C_dis",), reference="protocol_constant",
    ),
    FeatureSpec(
        "T_degC", "condition", "degC", "Nominal chamber temperature of the block.",
        sources=("T_degC",), reference="protocol_constant",
    ),
    FeatureSpec(
        "visit_index", "state", "count",
        "Zero-based index of the current cycling block for this cell. Copy of "
        "the ``visit`` identity column, registered under a distinct name so "
        "identity and feature namespaces never collide.",
        sources=("visit",), reference="protocol_constant",
    ),
)

# ---------------------------------------------------------------------------
# Group B — cycle aggregates at the anchor cycle and safe derived quantities
# ---------------------------------------------------------------------------
_GROUP_B: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "Q_discharge_Ah", "cycle_aggregate", "Ah",
        "Discharge capacity of the anchor cycle.",
        sources=("Q_discharge_Ah",),
    ),
    FeatureSpec(
        "coulombic_efficiency", "cycle_aggregate", "ratio",
        "Q_discharge / Q_charge of the anchor cycle. Makes Q_charge_Ah and "
        "capacity_throughput_ratio redundant.",
        sources=("Q_charge_Ah", "Q_discharge_Ah"),
    ),
    FeatureSpec(
        "energy_loss_Wh", "cycle_aggregate", "Wh",
        "E_charge_Wh - E_discharge_Wh at the anchor cycle.",
        sources=("E_charge_Wh", "E_discharge_Wh"),
    ),
    FeatureSpec(
        "voltage_hysteresis_V", "cycle_aggregate", "V",
        "mean_charge_voltage - mean_discharge_voltage, each computed as "
        "E/Q with safe division.",
        sources=("E_charge_Wh", "Q_charge_Ah", "E_discharge_Wh", "Q_discharge_Ah"),
    ),
    FeatureSpec(
        "T_max", "cycle_aggregate", "degC",
        "Maximum measured cell temperature during the anchor cycle.",
        sources=("T_max",),
    ),
    FeatureSpec(
        "temperature_span_C", "cycle_aggregate", "degC",
        "T_max - T_min at the anchor cycle.",
        sources=("T_max", "T_min"),
    ),
)

# ---------------------------------------------------------------------------
# Group C — compact backward-looking temporal features
#
# Only two degradation-sensitive variables are given temporal summaries, and
# only one window (10 cycles) survives redundancy pruning. Slopes use cycles at
# or before the anchor exclusively.
# ---------------------------------------------------------------------------
_GROUP_C: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "Q_discharge_Ah_r10_slope", "temporal", "Ah/cycle",
        "Robust backward slope of discharge capacity over the trailing 10 cycles.",
        sources=("Q_discharge_Ah",), reference="anchor_cycle_minus_9..anchor_cycle",
    ),
    FeatureSpec(
        "Q_discharge_Ah_r10_std", "temporal", "Ah",
        "Backward rolling standard deviation of discharge capacity, 10 cycles.",
        sources=("Q_discharge_Ah",), reference="anchor_cycle_minus_9..anchor_cycle",
    ),
    FeatureSpec(
        "Q_discharge_Ah_dev_r10", "temporal", "Ah",
        "Anchor-cycle discharge capacity minus its trailing 10-cycle mean.",
        sources=("Q_discharge_Ah",), reference="anchor_cycle_minus_9..anchor_cycle",
    ),
    FeatureSpec(
        "coulombic_efficiency_r10_mean", "temporal", "ratio",
        "Backward rolling mean coulombic efficiency, 10 cycles.",
        sources=("coulombic_efficiency",),
        reference="anchor_cycle_minus_9..anchor_cycle",
    ),
    FeatureSpec(
        "coulombic_efficiency_r10_slope", "temporal", "1/cycle",
        "Robust backward slope of coulombic efficiency over 10 cycles. "
        "Gassing-sensitive.",
        sources=("coulombic_efficiency",),
        reference="anchor_cycle_minus_9..anchor_cycle",
    ),
)

# ---------------------------------------------------------------------------
# Group D — true voltage-resolved Delta-Q(V)
#
# Reference definition (primary, plan option A): the last complete cycle of the
# current cycling block minus the first complete cycle of the same block. Both
# cycles precede the anchor, which is by construction the last complete cycle of
# the block, so the feature is causal without any cross-file state.
# ---------------------------------------------------------------------------
_GROUP_D: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "dqv_logvar", "dqv", "log(Ah^2)",
        "log variance of Delta-Q(V) over the shared voltage grid (Severson).",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
    FeatureSpec(
        "dqv_min", "dqv", "Ah",
        "Minimum of Delta-Q(V) over the shared voltage grid.",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
    FeatureSpec(
        "dqv_l2", "dqv", "Ah",
        "L2 norm of Delta-Q(V) over the shared voltage grid.",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
    FeatureSpec(
        "dqv_skew", "dqv", "dimensionless",
        "Skewness of the Delta-Q(V) distribution.",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
    FeatureSpec(
        "dqv_v_at_max_abs_dev", "dqv", "V",
        "Voltage at which |Delta-Q(V)| is largest.",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
    FeatureSpec(
        "dqv_overlap_span_V", "dqv", "V",
        "Width of the shared voltage interval used for Delta-Q(V).",
        sources=("Voltage",), reference="block_first..block_last",
    ),
)

# ---------------------------------------------------------------------------
# Group E — incremental capacity / differential voltage
#
# Only the first reproducible discharge dQ/dV peak is registered, referenced to
# the first complete cycle of the same block. Signed information is preserved.
# ---------------------------------------------------------------------------
_GROUP_E: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "icdv_dis_peak1_V_shift", "icdv", "V",
        "Voltage shift of the first discharge dQ/dV peak relative to the "
        "first complete cycle of the same block.",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
    FeatureSpec(
        "icdv_dis_peak1_height_change", "icdv", "Ah/V",
        "Signed change in the first discharge dQ/dV peak height relative to "
        "the first complete cycle of the same block.",
        sources=("Voltage", "Ah_counter"), reference="block_first..block_last",
    ),
)

# ---------------------------------------------------------------------------
# Group F — compact thermal descriptors
# ---------------------------------------------------------------------------
_GROUP_F: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "thermal_discharge_rise_degC", "thermal", "degC",
        "Temperature rise across the anchor cycle's discharge phase.",
        sources=("Temperature",),
    ),
    FeatureSpec(
        "thermal_max_dTdt_degC_s", "thermal", "degC/s",
        "Maximum |dT/dt| during the anchor cycle.",
        sources=("Temperature", "Time"),
    ),
    FeatureSpec(
        "thermal_rise_per_Ah", "thermal", "degC/Ah",
        "Discharge temperature rise divided by discharged capacity.",
        sources=("Temperature", "Ah_counter"),
    ),
)

# ---------------------------------------------------------------------------
# Group G — resistance from the PREVIOUS RPT only
#
# The standard-cycling protocol has a median in-block rest duration of 0 s, so
# DCIR and relaxation cannot be measured inside cycling blocks. Only the
# previous RPT/CU file carries usable current steps. Relaxation time constants
# are not registered: rest segments there are too short and too few to fit
# stably. See docs/UNIFIED_PREPROCESSING.md.
# ---------------------------------------------------------------------------
_GROUP_G: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "dcir_prev_rpt_discharge_ohm", "resistance", "ohm",
        "Discharge-side DC internal resistance dV/dI at the previous RPT.",
        sources=("Voltage", "Current"), reference="previous_rpt",
    ),
    FeatureSpec(
        "dcir_prev_rpt_delta_from_baseline_ohm", "resistance", "ohm",
        "Previous-RPT DCIR minus the cell's first valid RPT DCIR.",
        sources=("Voltage", "Current"), reference="initial_rpt..previous_rpt",
    ),
)

# ---------------------------------------------------------------------------
# Group H — interpretable condition x stress interactions
# ---------------------------------------------------------------------------
_GROUP_H: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "inverse_temperature_1_K", "interaction", "1/K",
        "1 / (T_degC + 273.15). No activation energy is assumed.",
        sources=("T_degC",), reference="protocol_constant",
    ),
    FeatureSpec(
        "stress_x_DOD_fraction", "interaction", "equivalent_full_cycles",
        "EFC_cum * (DOD_pct / 100).",
        sources=("Q_discharge_Ah", "DOD_pct"),
    ),
    FeatureSpec(
        "stress_x_inverse_temperature", "interaction", "EFC/K",
        "EFC_cum * inverse_temperature_1_K.",
        sources=("Q_discharge_Ah", "T_degC"),
    ),
    FeatureSpec(
        "stress_x_C_dis", "interaction", "EFC/h",
        "EFC_cum * C_dis.",
        sources=("Q_discharge_Ah", "C_dis"),
    ),
)

# ---------------------------------------------------------------------------
# Availability flags. Registered explicitly so that imputation of the
# corresponding continuous block is never silent.
# ---------------------------------------------------------------------------
_FLAGS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "dqv_available", "flag", "bool",
        "1 when Delta-Q(V) had a usable shared voltage interval.",
        kind="flag", sources=("Voltage", "Ah_counter"),
        reference="block_first..block_last",
    ),
    FeatureSpec(
        "icdv_available", "flag", "bool",
        "1 when a reproducible first discharge dQ/dV peak was found in both "
        "the anchor and the block-reference cycle.",
        kind="flag", sources=("Voltage", "Ah_counter"),
        reference="block_first..block_last",
    ),
    FeatureSpec(
        "dcir_available", "flag", "bool",
        "1 when the previous RPT contained a valid current step.",
        kind="flag", sources=("Voltage", "Current"), reference="previous_rpt",
    ),
    FeatureSpec(
        "temporal_window_available", "flag", "bool",
        "1 when at least MIN_TEMPORAL_OBSERVATIONS complete cycles precede "
        "the anchor within the block.",
        kind="flag", sources=("Q_discharge_Ah",),
        reference="anchor_cycle_minus_9..anchor_cycle",
    ),
)

#: Fixed order. Do not reorder — fold preprocessor state and feature hashes
#: depend on it.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    _GROUP_A + _GROUP_B + _GROUP_C + _GROUP_D + _GROUP_E
    + _GROUP_F + _GROUP_G + _GROUP_H + _FLAGS
)

#: Minimum number of trailing complete cycles required for Group C features.
MIN_TEMPORAL_OBSERVATIONS = 5

#: Identity, grouping and target-construction columns that must survive into
#: unified.parquet but must never be model inputs.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "anchor_id", "condition", "cell", "file_id", "path", "relative_path",
    "visit", "global_cycle", "cycle_in_file", "test_type", "file_start_time",
    "initial_rpt_Q_Ah", "previous_rpt_path", "previous_rpt_visit",
    "next_rpt_path", "next_rpt_visit", "next_rpt_match_method",
    "next_rpt_horizon_days", "is_prediction_anchor",
    "cumulative_Ah_throughput", "cycle_end_time_s",
)

#: Learned target plus the reporting views algebraically derived from it.
TARGET_COLUMNS: tuple[str, ...] = (
    "next_rpt_Q_Ah", "next_rpt_SOH_pct",
    "delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct",
)

PRIMARY_TARGET = "next_rpt_Q_Ah"


class RegistryError(RuntimeError):
    """Raised when the registry is internally inconsistent."""


def _validate() -> None:
    names = [spec.name for spec in FEATURE_SPECS]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RegistryError(f"duplicate registry entries: {duplicates}")
    if len(names) > MAX_MODEL_FEATURES:
        raise RegistryError(
            f"registry declares {len(names)} model features, exceeding the hard "
            f"maximum of {MAX_MODEL_FEATURES}"
        )
    overlap = sorted(set(names) & (set(TARGET_COLUMNS) | set(IDENTITY_COLUMNS)))
    if overlap:
        raise RegistryError(
            f"registry entries collide with identity/target columns: {overlap}"
        )
    stress = [spec.name for spec in FEATURE_SPECS if spec.role == "stress_coordinate"]
    if len(stress) != 1:
        raise RegistryError(
            f"exactly one stress_coordinate is required, found {stress}"
        )
    state = [spec.name for spec in FEATURE_SPECS if spec.role == "initial_state"]
    if len(state) != 1:
        raise RegistryError(
            f"exactly one initial_state feature is required, found {state}"
        )


_validate()


def model_feature_columns(*, exclude_roles: Iterable[FeatureRole] = ()) -> list[str]:
    """Registered model input columns in fixed order.

    Parameters
    ----------
    exclude_roles:
        Roles to drop. NaPINN-Q passes ``("stress_coordinate", "initial_state")``
        so that the differentiable stress coordinate and the initial state are
        not duplicated inside the auxiliary feature tensor. Classical models
        pass nothing and receive the full list.
    """
    excluded = set(exclude_roles)
    return [spec.name for spec in FEATURE_SPECS if spec.role not in excluded]


def continuous_feature_columns() -> list[str]:
    """Registered continuous columns — the ones RobustScaler is fitted on."""
    return [spec.name for spec in FEATURE_SPECS if not spec.is_flag]


def flag_feature_columns() -> list[str]:
    """Registered binary availability flags. These are never scaled."""
    return [spec.name for spec in FEATURE_SPECS if spec.is_flag]


def spec_for(name: str) -> FeatureSpec:
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"{name!r} is not a registered unified feature")


def stress_coordinate_column() -> str:
    return next(s.name for s in FEATURE_SPECS if s.role == "stress_coordinate")


def initial_state_column() -> str:
    return next(s.name for s in FEATURE_SPECS if s.role == "initial_state")


def group_of(name: str) -> str:
    return spec_for(name).group


def registry_manifest() -> list[dict[str, object]]:
    """Serializable description of the registry, written to the artifacts."""
    return [{
        "feature": spec.name,
        "group": spec.group,
        "kind": spec.kind,
        "role": spec.role,
        "unit": spec.unit,
        "reference": spec.reference,
        "source_columns": ";".join(spec.sources),
        "description": spec.description,
        "registry_version": REGISTRY_VERSION,
    } for spec in FEATURE_SPECS]


def registry_hash() -> str:
    """Stable hash of the registry contents, recorded alongside every fold."""
    import hashlib
    import json

    payload = json.dumps(registry_manifest(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


N_MODEL_FEATURES = len(FEATURE_SPECS)
N_CONTINUOUS_FEATURES = len(continuous_feature_columns())
N_FLAG_FEATURES = len(flag_feature_columns())
