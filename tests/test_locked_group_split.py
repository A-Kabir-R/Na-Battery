import pandas as pd
import pytest

from src.splits.group_kfold import (
    build_locked_split_manifest,
    validate_locked_split_manifest,
)


def _data():
    rows = []
    for index in range(40):
        for anchor in range(1 + index % 3):
            rows.append({
                "cell": f"cell-{index:02d}",
                "condition": f"condition-{index % 8}",
                "anchor": anchor,
            })
    return pd.DataFrame(rows)


def test_locked_split_is_grouped_deterministic_and_75_25():
    data = _data()
    first = build_locked_split_manifest(data, n_splits=5, random_state=42)
    second = build_locked_split_manifest(
        data.sample(frac=1, random_state=9), n_splits=5, random_state=42
    )
    validate_locked_split_manifest(data, first, n_splits=5)
    assert first.sort_values("cell").reset_index(drop=True).equals(
        second.sort_values("cell").reset_index(drop=True)
    )
    assert set(first["outer_role"]) == {"development", "holdout"}
    assert first[first["outer_role"] == "holdout"]["cv_fold"].isna().all()
    assert set(first.loc[first["outer_role"] == "development", "cv_fold"].astype(int)) == set(range(5))
    assert abs((first["outer_role"] == "holdout").mean() - 0.25) <= 0.03


def test_locked_split_rejects_condition_or_checksum_changes():
    data = _data()
    manifest = build_locked_split_manifest(data, n_splits=5, random_state=42)
    changed = data.copy()
    changed.loc[changed["cell"] == "cell-00", "condition"] = "corrected-condition"
    with pytest.raises(ValueError, match="condition assignments changed"):
        validate_locked_split_manifest(changed, manifest, n_splits=5)
    tampered = manifest.copy()
    index = tampered[tampered["outer_role"] == "development"].index[0]
    tampered.loc[index, "cv_fold"] = (int(tampered.loc[index, "cv_fold"]) + 1) % 5
    with pytest.raises(ValueError):
        validate_locked_split_manifest(data, tampered, n_splits=5)
