# Unified preprocessing and NaPINN-Q revision

One preprocessing table, one feature registry, one fold-local transform, shared
by Previous-RPT, Ridge, ExtraTrees and NaPINN-Q.

## Why the dimensionality is 36 + 4, not 35–60

The prediction unit is the RPT-to-RPT anchor, and there are only:

| partition | anchors | cells |
|---|---|---|
| development | 175 | 26 |
| per outer fold (train / val) | 140 / 35 | 21 / 5 |
| locked holdout | 61 | 9 |

A 64-column matrix would put the NaPINN-Q solution head above 1000 first-layer
parameters on 140 training rows, contradicting the compact-architecture
requirement in the same plan. The registry therefore declares **36 continuous
features + 4 availability flags = 40 columns**, hard-capped at 64 by
`unified_feature_registry.MAX_MODEL_FEATURES`.

## What each model receives

| | features | notes |
|---|---|---|
| Previous-RPT | none | uses the same anchor rows, split manifest and targets |
| Ridge | all 40 | robust-scaled |
| ExtraTrees | all 40 | same matrix, for strict consistency |
| NaPINN-Q | 38 auxiliary | `EFC_cum` and `previous_rpt_Q_Ah` are excluded because they enter structurally as the stress coordinate and the initial state |

`UnifiedFoldPreprocessor` is the single transform: median imputation and robust
scaling fitted on training rows only, flags never scaled, fixed column order,
and no fold-local column dropping (so the matrix width is fold-invariant).

## Protocol findings that shaped the feature set

Inspected from `canonical/steps.parquet` (289 CU files, 381k steps):

* **Cycling blocks contain no usable current step.** Median in-block rest
  duration is **0 s**. DCIR and relaxation are not measurable there.
* **RPT/CU files carry a regular DCIR pulse train**: 3 SOC levels
  (pre-pulse OCV ≈ 3.60 / 2.95 / 2.20 V) × 3 C-rates (0.5C, 1C, 2C) × 2
  directions = **18 pulses of 12 s**, each after a ≥1800 s rest. Present in
  **285 / 289** files. Measured DCIR spans 0.047–0.115 Ω with drift up to
  +0.065 Ω — physically sensible for a 1.2 Ah cell and clearly informative.
* Pulses are selected by **physical identity** (phase + C-rate + pre-pulse OCV
  rank), never by ordinal: three files have a gap that shifts the train.
* Only the **previous** RPT is read. The next RPT defines the target.
* **Relaxation features are not implemented.** The 1800 s / 7200 s post-pulse
  rests could support an exponential fit; that was scoped out of this revision
  and remains open work. It is a real opportunity, not a protocol limitation.

## Delta-Q(V) reference definition

Primary definition (plan option A):

```
ΔQ(V) = Q_last_complete_cycle(V) − Q_first_complete_cycle(V)
```

both cycles drawn from the anchor's **own, already-finished** cycling block, on
a grid spanning only the shared voltage interval. This is causal without any
cross-file state because the anchor is by construction the block's last complete
cycle, and blocks contain 187–1000 cycles (median 200), so the two curves are
separated by substantial degradation.

The previous `Q_discharge_diff_logvar_r10` was a rolling variance of *scalar*
capacity differences — not a voltage-resolved quantity — and is not registered.

## IC/DV peak convention

Peak *location* is found on `|dQ/dV|`. This is required, not a shortcut: the
curve is parameterised as capacity-removed against ascending voltage, so for a
discharge phase `Q(V)` decreases and every incremental-capacity peak is a
*minimum* of the signed derivative. Two things differ from the legacy extractor:

* the reported height keeps the **sign** of `dQ/dV`;
* peaks are ordered by **voltage position**, not magnitude, so "peak 1" denotes
  the same physical transition in every fold. The legacy version reported only
  the single largest-magnitude peak, which can jump between transitions as a
  cell ages.

## NaPINN-Q corrections

### Hard initial condition

```
local_stress = s − s_current
u_hat(s)     = u_current + local_stress · f_θ(s, x)
u_hat(s_next) = u_current + Δs · f_θ(s_next, x)
du_hat/ds     = f_θ(s, x) + local_stress · ∂f_θ/∂s
```

`u_hat(s_current) = u_current` **exactly**, for any parameters. The final layer
of `f_θ` is zero-initialised, so an untrained model predicts persistence.

Previously `predict_delta_u` produced `u_current + f_θ(s, x)`, which satisfies
the initial condition only if `f_θ(s_current, x)` happens to vanish — it does
not. The IC was penalised, not enforced. With the correction the IC residual is
identically zero and its loss weight is inert by construction; it is still
computed and logged as a runtime assertion.

### Composite loss

```
L = λ_data·L_data + λ_rate_data·L_rate_data + λ_pde·m_pde·L_pde
  + λ_integral·m_integral·L_integral + λ_pair·m_pair·L_pair
  + λ_mono·m_mono·L_mono + λ_bounds·m_bounds·L_bounds
  + λ_rate_reg·L_rate_reg + λ_ic·L_ic
```

`m_*` are the clipped gradient-balance multipliers. `L_data` is the reference
component and is never rescaled. `rate_data` and `pairwise` are data-like and
are **not** curriculum-ramped.

Observed rate target:

```
observed_rate = max((u_current − u_true_next) / max(Δs, ε), 0)
```

clamped at zero so measurement-noise regeneration cannot create a negative
target for a Softplus head; transitions whose apparent gain exceeds the
fold-derived tolerance are masked out entirely.

### Curriculum and checkpoint eligibility

With `maximum_epochs=500`, `warmup_fraction=0.05`, `ramp_end_fraction=0.20`:

| | epoch |
|---|---|
| physics starts ramping | 25 |
| `physics_full_epoch` (factor = 1.0) | 100 |
| checkpoint eligibility begins | 100 |
| `minimum_epochs` | 180 |

Checkpoint selection and early-stopping patience both start at
`physics_full_epoch`. The previous gate used the *warm-up* epoch (16 with the old
settings) while full physics only arrived at epoch 80, so epochs 16–79 could win
the selection and the reported "full PINN" could be a partially-physics model.
A run now **fails** if the selected checkpoint has curriculum factor < 1.0.

### Two-phase refit

Phase 2 builds its curriculum from `refit_epochs = best_epoch + 1`, not from
`maximum_epochs`. Reusing the inner-training budget meant a refit shorter than
`ramp_end_fraction · maximum_epochs` never reached factor 1.0 — the model
producing the reported outer-validation predictions was trained under weaker
physics than the model whose epoch was selected. The refit now raises if it
never reaches full physics.

### Cell-balanced batching

`batch_mode: cell_balanced` had **no runtime effect** — it appeared nowhere in
the codebase, and sampling was a plain `randperm` over rows. With 21 training
cells contributing 2–14 anchors each that is a real imbalance. `batching.py`
now samples cells uniformly, then transitions within each cell, and the forward
pass is restricted to the sampled rows (previously the full frame was forwarded
on every batch and only the losses were indexed).

### Gradient balancing

Previously applied to the PDE term only, clipped to `[0.01, 100]`, and — the
reporting bug — the balance scale was **not logged**, so `effective_lambda_pde`
in the epoch log was the pre-balance value. Now every active component is
balanced against the data gradient, clipped to `[0.1, 10]`, and the epoch log
records nominal weights, curriculum factor, adaptive multipliers and component
gradient norms separately.

## Scientific-integrity notes

1. **The locked holdout is not pristine for the classical arm.** Holdout metrics
   for P1/P2/P3 × 5 models were computed during the earlier
   `extratrees_preprocessing_selection` run, and the preprocessing preference
   (p2/p3 > p1) was formed with those numbers available. The final report must
   state that the classical holdout is a **second look**, and a first look only
   for the unified / NaPINN-Q arm. `experiment.evaluate_holdout` ships as
   `false` so no further holdout access happens during development.

2. **Seed reporting.** Seed 42 is the pre-registered primary and the only seed
   used for any tuning decision. Seeds 43–46 are reported *only* as
   "spread across random initializations" — never seed-averaged as a headline,
   never as a selection criterion. On 140 training rows the initialization
   spread can be comparable to the Ridge-vs-PINN gap, so suppressing it would
   overstate the result.

3. **`L_rate_data` is not independent physics.** Under the hard-IC
   parameterisation `u_hat_next = u_current + Δs·f_θ`, the observed rate is an
   affine restatement of the data target, so this term mainly ties the two heads
   together and stabilises the rate head. It is worth having; it is not extra
   information.

4. **The pairwise loss is shipped disabled** (`pairwise: 0.0`) and reported as
   ablation A7. Its weight would have to be tuned on inner-validation folds of
   ~28 rows / 4–5 cells, which is too noisy to trust as a headline setting.

5. **Cross-cell metrics must be shown next to pooled ones.** Pooled R² on this
   dataset is dominated by between-cell capacity variance: Ridge p3 reached
   pooled R² 0.974 but cell-macro R² 0.495 and *negative* condition-macro R².
   Pooled numbers alone materially overstate generalization to an unseen cell.

6. **P1/P2/P3 are retained as a reported feature-family ablation.** They are not
   used for final training, but the comparison is already computed and is
   publishable evidence about which feature families carry the signal.

## Commands

```bash
# 1. Features (unified only; ~289 .ird files re-parsed for 2 cycles each)
python3 scripts/01_build_features.py --unified-only

# 2. Test suite
python3 -m pytest tests/ -q

# 3. Dry-run manifests
python3 scripts/02_run_all_experiments.py --dry-run
python3 scripts/06_run_pinn_experiments.py --dry-run

# 4. One-fold smoke tests
python3 scripts/06_run_pinn_experiments.py --smoke-test --fold 0 --seed 42

# 5. Five-fold development run, seed 42 primary
python3 scripts/02_run_all_experiments.py
python3 scripts/06_run_pinn_experiments.py --seed 42

# 6. Seed spread (robustness note only)
for s in 43 44 45 46; do
  python3 scripts/06_run_pinn_experiments.py --seed "$s"
done

# 7. Aggregation, plots and the identical-fold comparison
python3 scripts/03_aggregate_results.py
python3 scripts/07_aggregate_pinn_results.py
python3 scripts/09_combine_classical_pinn_results.py
python3 scripts/04_plot_results.py
python3 scripts/08_plot_pinn_results.py

# 8. Holdout — ONLY after every setting is frozen.
#    Set experiment.evaluate_holdout: true in config.yaml first.
python3 scripts/02_run_all_experiments.py
```
