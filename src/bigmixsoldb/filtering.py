from __future__ import annotations

from typing import Any

import pandas as pd

from bigmixsoldb.constants import FRACTION_UNITS, OUTPUT_COLUMNS
from bigmixsoldb.postprocess import is_missing


def has_value(value: Any) -> bool:
    return not is_missing(value)


def classify_mixture_type(row: pd.Series) -> str:
    has_solvent_1 = has_value(row.get("Solvent 1"))
    has_solvent_2 = has_value(row.get("Solvent 2"))
    has_solvent_3 = has_value(row.get("Solvent 3"))
    has_extra = has_value(row.get("Extra Solvents"))

    if has_extra:
        return "extra"
    if has_solvent_1 and has_solvent_2 and has_solvent_3:
        return "ternary"
    if has_solvent_1 and has_solvent_2:
        return "binary"
    return "single"


def is_valid_species(value: Any) -> bool:
    return has_value(value)


def filter_entries_like_reference(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if dataframe.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {}

    df = dataframe.copy()
    df["_mixture_type"] = df.apply(classify_mixture_type, axis=1)
    stats: dict[str, dict[str, int]] = {}
    filtered_parts: list[pd.DataFrame] = []

    smiles_by_type = {
        "single": ["SMILES_Compound", "SMILES_Solvent_1"],
        "binary": ["SMILES_Compound", "SMILES_Solvent_1", "SMILES_Solvent_2"],
        "ternary": ["SMILES_Compound", "SMILES_Solvent_1", "SMILES_Solvent_2", "SMILES_Solvent_3"],
        "extra": ["SMILES_Compound", "SMILES_Solvent_1", "SMILES_Solvent_2", "SMILES_Solvent_3"],
    }

    for mixture_type, group in df.groupby("_mixture_type", sort=True):
        smiles_columns = [column for column in smiles_by_type[mixture_type] if column in group.columns]
        smiles_ok = group[smiles_columns].apply(lambda column: column.map(is_valid_species)).all(axis=1)

        if mixture_type == "single":
            unit_ok = group["Concentration Unit"].isin(FRACTION_UNITS) | group["Concentration Unit"].map(is_missing)
        else:
            unit_ok = group["Concentration Unit"].isin(FRACTION_UNITS)
            if mixture_type == "binary":
                values_ok = group["Concentration Solvent 1"].map(has_value) & group["Concentration Solvent 2"].map(has_value)
            elif mixture_type == "ternary":
                values_ok = (
                    group["Concentration Solvent 1"].map(has_value)
                    & group["Concentration Solvent 2"].map(has_value)
                    & group["Concentration Solvent 3"].map(has_value)
                )
            else:
                values_ok = pd.Series(True, index=group.index)
            unit_ok &= values_ok

        filtered = group[smiles_ok & unit_ok].copy()
        stats[mixture_type] = {
            "before": int(len(group)),
            "after": int(len(filtered)),
            "smiles_removed": int((~smiles_ok).sum()),
            "unit_removed": int((~unit_ok).sum()),
            "both_removed": int((~smiles_ok & ~unit_ok).sum()),
        }
        if mixture_type != "extra":
            filtered_parts.append(filtered)

    if not filtered_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), stats

    combined = pd.concat(filtered_parts, ignore_index=True)
    combined = combined.drop(columns=["_mixture_type"], errors="ignore")
    combined = combined[[column for column in OUTPUT_COLUMNS if column in combined.columns]]
    return combined, stats