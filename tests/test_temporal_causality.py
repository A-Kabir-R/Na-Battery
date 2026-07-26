import pandas as pd
from pandas.testing import assert_frame_equal

from src.preprocessing.p2_rolling_delta import build_p2


def _cycles():
    rows = []
    for cycle in range(1, 16):
        rows.append({
            "condition": "DOD100_1C1C_25degC",
            "cell": "CELL1",
            "visit": 1,
            "test_type": "CYC",
            "path": "file.ird",
            "file_id": "file-1",
            "cycle_in_file": cycle,
            "global_cycle": cycle,
            "cycle_complete": True,
            "cycle_qc_flags": "",
            "Q_charge_Ah": 1.2 - cycle * 0.001,
            "Q_discharge_Ah": 1.19 - cycle * 0.001,
            "E_charge_Wh": 4.0,
            "E_discharge_Wh": 3.5,
            "coulombic_efficiency": 0.99,
            "energy_efficiency": 0.875,
            "V_max": 3.8,
            "V_min": 1.5,
            "T_max": 26.0,
            "T_min": 24.0,
            "T_mean": 25.0,
            "Ah_span": 1.2,
            "duration_s": 7200.0,
            "cumulative_Ah_throughput": cycle * 1.19,
            "DOD_pct": 100,
            "C_ch": 1.0,
            "C_dis": 1.0,
            "T_degC": 25,
        })
    return pd.DataFrame(rows)


def test_future_cycles_do_not_change_earlier_temporal_features():
    original = _cycles()
    changed = original.copy()
    changed.loc[changed["global_cycle"] > 10, "Q_discharge_Ah"] = 1000.0
    before = build_p2(original)
    after = build_p2(changed)
    feature_columns = [column for column in before.columns if column not in {"Q_discharge_Ah"}]
    assert_frame_equal(
        before.loc[before["global_cycle"] <= 10, feature_columns].reset_index(drop=True),
        after.loc[after["global_cycle"] <= 10, feature_columns].reset_index(drop=True),
    )
