# data statics — statistical analysis of the Standard-cycling sodium-ion dataset

Generated from the 548 `.ird` files (~13 GB) under
`Standard cycling/DOD*_*C*C_*degC/{CU_at_25degC,Cycling_Periods}/`.

Two parse errors (empty CU files misfiled in `Cycling_Periods`) were skipped —
see `logs/parse_errors.csv`.

## Layout

```
data statics/
├── scripts/
│   ├── analyze.py     # walk + parse .ird, extract per-file & per-cycle stats
│   └── plot.py        # render every plot from the tables
├── tables/
│   ├── inventory.csv         (548 rows)  one row per .ird file
│   ├── file_statistics.csv   (548 rows)  V/I/T/Ah stats, dt, rows, Q totals
│   ├── cycle_metrics.csv     (56 650 rows) per-cycle Q_ch, Q_dis, CE, EE, T_max, SOH
│   ├── degradation_rates.csv (46 rows)   per (condition, cell) fade slope + EOL
│   └── condition_summary.csv (16 rows)   condition-level mean metrics
├── plots/                    (94 PNGs)
│   ├── inventory/            file counts, cells per condition, CU/CYC split
│   ├── signals/              boxplots + histograms of V, I, T, Ah, dt
│   ├── sample_waveforms/     6-hour V/I/T traces and V-vs-Ah for one CYC file per condition
│   ├── cycles/               Q_dis, CE, EE, T_max vs cycle (per condition, cells overlaid)
│   ├── capacity_fade/        SOH panels, overlay, SOH vs throughput, EOL bar, slope bar
│   ├── heatmaps/             T × C-rate heat map of final SOH, fade slope, EOL, initial Q (one per DOD)
│   └── comparisons/          DOE bar charts (fade / EOL / SOH vs T, C-rate, DOD) + T-scatter
└── logs/
    ├── analyze.log           run log (~100 s on 6 workers)
    └── parse_errors.csv      files that failed to parse
```

## Reproduce

```
python scripts/analyze.py            # ~100 s with 6 workers
python scripts/plot.py               # ~30 s
```

`analyze.py --limit N` runs on only the first N files for a smoke test.

## What each table contains

| column                          | meaning                                              |
|---------------------------------|------------------------------------------------------|
| `condition`                     | e.g. `DOD100_1C1C_25degC`                            |
| `DOD_pct, C_ch, C_dis, T_degC`  | decoded factors                                      |
| `test_type`                     | `CU` (reference perf. test @ 25 °C) or `CYC` (aging) |
| `cell, visit, test_id`          | cell ID, visit index (V00, V01, …), test id          |
| `V_min/max/mean/std`            | per-file voltage statistics                          |
| `I_min/max/mean/std`            | per-file current statistics                          |
| `T_min/max/mean/std`            | per-file cell-temperature statistics                 |
| `Ah_min/max/span`               | Ah_counter range                                     |
| `dt_median_s`                   | median sample period                                 |
| `Q_charge_Ah_total`             | ∫I·dt for I>0 over the file                          |
| `Q_discharge_Ah_total`          | -∫I·dt for I<0                                       |

Per-cycle table adds `global_cycle`, `cumulative_Ah_throughput`, `Q_initial_Ah`,
`SOH_pct` (Q_dis divided by median of first 3 cycles).

Per-degradation table adds `fade_slope_pct_per_cycle`, `fade_intercept_pct`,
`eol80_cycle` (first cycle with SOH<80%), `CE_median`, `EE_median`.

## Key observations (from `condition_summary.csv`)

| condition                | cells | mean cycles | initial Q (Ah) | final SOH % | fade %/cycle |
|--------------------------|:----:|:-----------:|:--------------:|:-----------:|:------------:|
| DOD20_2C2C_25degC        | 3    | 3400        | 0.24           | 99.99       | -0.000       |
| DOD20_2C2C_40degC        | 3    | 3400        | 0.24           | 99.99       | -0.000       |
| DOD80_2C2C_25degC        | 3    | 2600        | 0.94           | 97.96       | -0.0004      |
| DOD80_2C2C_40degC        | 3    | 2200        | 0.96           | 97.00       | -0.0007      |
| DOD100_1C1C_25degC       | 3    | 1604        | 1.17           | 91.53       | -0.005       |
| DOD100_2C2C_25degC       | 3    | 1912        | 1.16           | 85.98       | -0.007       |
| DOD100_1C1C_40degC       | 3    | 857         | 1.22           | 91.49       | -0.010       |
| DOD100_0.5C0.5C_-10degC  | 3    | 202         | 1.07           | **48.91**   | **-0.425**   |
| DOD100_1C1C_-10degC      | 3    | 164         | 1.05           | 72.24       | -0.223       |
| DOD80_1C1C_-10degC       | 3    | 304         | 0.95           | 58.58       | -0.354       |
| DOD100_2C2C_-10degC      | 2    | 304         | 1.04           | 69.66       | -0.329       |
| 50 °C conditions         | 2–3  | 190–405     | 1.07–1.24      | 100–118*    | ~-0.002      |

\* SOH >100 % at 50 °C reflects capacity activation early in cycling (initial Q
computed from the first 3 cycles is lower than mid-life Q); interpret those
rows as "no measurable fade" rather than actual gain.

## Signal-level notes

- Median sample period is ~1 s (see `plots/signals/hist_dt_median.png`), with
  a mean of ~1.7 s (rest steps and CV holds inflate the mean).
- Voltage envelope for the cell class is ~1.5 V (cutoff) to ~3.8 V (upper cutoff).
- CU files at any aging condition are all run at 25 °C — visible in
  `T_mean_by_condition_CYC.png` (the CYC-only version), which correctly shows
  temperature setpoints of -10, 25, 40, 50 °C.
- The largest peak currents (~5 A) appear in 2C2C conditions with DOD100.

## Failure-mode signal

Cold-temperature aging (-10 °C) dominates as the failure driver:

- DOD100_0.5C0.5C_-10degC loses **~50 % SOH in only ~200 cycles**, a fade rate
  ~100× higher than the 25 °C 1C1C reference.
- CE at -10 °C drops well below 1.0 (0.77–0.94) versus ~1.00 at 25 °C+, which
  is consistent with reversible-capacity loss from plating / SEI growth in the
  cold.
- Energy efficiency in cold conditions collapses to 0.60–0.72 versus 0.93 at
  25 °C.

See `plots/comparisons/scatter_fade_vs_T.png` for the summary view.
