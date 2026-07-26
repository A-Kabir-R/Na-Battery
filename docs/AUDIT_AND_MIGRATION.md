# Audit And Migration

## Blocking Defects Repaired

| Severity | Original area | Defect | Repair | Regression test |
|---|---|---|---|---|
| Critical | `src/preprocessing/p3_waveform.py` and external statistics script | Charge-edge boundaries counted file fragments and omitted final cycles | Shared StepID-aware discharge/recharge state machine retains incomplete boundaries | `tests/test_protocol_segmentation.py` |
| Critical | `src/targets/next_rpt.py` | Summed every discharge in a CU file | Selects a unique full-range 0.5C reference discharge | `tests/test_protocol_segmentation.py`, `tests/test_next_rpt.py` |
| High | `src/targets/next_rpt.py` | Duplicate visits resolved by row order | Duplicate validated `(condition, cell, visit)` keys raise | `tests/test_next_rpt.py` |
| High | `src/targets/failure_label.py` | Future CE/SOH/temperature proxies were broadcast as gassing outcomes | Proxy target removed; observed event table required | Target remains disabled without events |
| High | `src/cleaning/incomplete_cycles.py` use in P1 | Low capacity was treated as incomplete | Completeness is protocol-phase based; low-capacity complete cycles remain | Raw smoke-build QC |
| High | `src/pipeline/run_experiment.py` | Selection and imputation used all folds | `FoldPreprocessor` fits on training rows only | `tests/test_fold_preprocessing.py` |
| High | `src/evaluation/loss_curves.py` | Outer test fold was model `eval_set` | Curves use disposable inner grouped validation; final model sees outer training only | Integration smoke run |
| High | P3 exception handling | Parser/feature failures became empty successful output | Per-file coverage table and blocking minimum coverage | P3 raw-file smoke run |
| High | Metric error rows | Error strings entered numeric metric values | Numeric NaN plus separate status and error columns | Integration smoke run |

## Original File Disposition

| File or area | Disposition | Reason |
|---|---|---|
| `src/io/ird_parser.py` | Rewritten | Structured errors, metadata, mixed time parsing, timestamp QC |
| `src/io/loaders.py` | Updated | Canonical tables are preferred and required for cycle metrics |
| `src/preprocessing/p1_aggregate.py` | Rewritten | No capacity-based deletion, clipping, or global one-hot fitting |
| `src/preprocessing/p2_rolling_delta.py` | Updated | Correct EFC units, causal features, no proxy gassing markers |
| `src/preprocessing/p3_waveform.py` | Rewritten | Shared segmentation, physical V-Ah integration, explicit coverage |
| `src/targets/next_rpt.py` | Rewritten | Validated RPT extraction, chronology, one anchor |
| `src/targets/rul.py` | Retained as secondary | Sustained crossing and censor metadata; not a default benchmark |
| `src/targets/failure_label.py` | Rewritten and gated | Independent observed events are required |
| `src/pipeline/run_experiment.py` | Rewritten | Fold-local preprocessing and atomic fold persistence |
| `src/cleaning/incomplete_cycles.py` | Legacy only | Not used by the canonical model pipeline |
| `src/cleaning/outliers.py` | Legacy only | Raw current is not Hampel-filtered before segmentation |
| `data statics/scripts/analyze.py` | Superseded for ML | Its tables remain legacy comparison outputs, not canonical inputs |
| RunPod runtime parser rewrite | Deprecated | Runtime source rewriting must not be used for reproducible runs |

## Data Flow

```text
Standard cycling/*.ird
  -> exact-scope discovery
  -> file identity + checksum
  -> structured IRD parse + timestamp QC
  -> contiguous StepID summaries + inferred current sign
  -> CYC discharge/recharge state machine
  -> CU unique 0.5C reference discharge
  -> manifest / steps / cycles / RPT / metadata / QC
  -> complete-cycle P1 aggregates
  -> past-only P2 temporal features
  -> shared-segment P3 waveform features
  -> one anchor per cycling file with next chronological RPT
  -> persisted cell-grouped outer folds
  -> fold-fitted preprocessing and models
  -> atomic predictions, metrics, models, and status
```

## Migration

1. Preserve old CSVs and model outputs as legacy comparison artifacts.
2. Build canonical tables from raw Standard-cycling files.
3. Review parser failures, physical-bound errors, role mismatches, cycle coverage, and RPT status.
4. Confirm the inferred RPT selection rule with protocol documentation.
5. Rebuild P1, P2, P3, and prediction anchors from canonical tables.
6. Do not migrate old SOH, RUL, or failure labels.
7. Add authoritative events and censoring before enabling those targets.
8. Create and approve a persisted fold manifest before comparative model reporting.
9. Run baselines before adding advanced models.

## Remaining Work

- Persist one approved fold manifest and reuse it across all model families.
- Add survival and competing-risk tables after event definitions are approved.
- Add per-condition and cell-bootstrap confidence intervals.
- Add run IDs containing Git revision, data-manifest hash, and config hash.
- Add a SQLite experiment index and top-level run resume.
- Add segmentation/QC visualizations and the final linked HTML report.
- Remove RunPod runtime source rewriting and pin a lock file.
