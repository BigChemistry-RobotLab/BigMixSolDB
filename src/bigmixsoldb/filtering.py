from __future__ import annotations

import math
from typing import Any

import pandas as pd

from bigmixsoldb.constants import FRACTION_UNITS, OUTPUT_COLUMNS
from bigmixsoldb.postprocess import is_missing, parse_numeric_value


CONCENTRATION_SUM_TOLERANCE = 1e-6


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


def single_concentration_value_ok(value: Any) -> bool:
    if is_missing(value):
        return True
    numeric_value = parse_numeric_value(value)
    return (
        numeric_value is not None
        and math.isfinite(numeric_value)
        and abs(numeric_value - 1.0) <= CONCENTRATION_SUM_TOLERANCE
    )


def concentration_sum_ok(group: pd.DataFrame, mixture_type: str) -> pd.Series:
    concentration_columns_by_type = {
        "binary": ["Concentration Solvent 1", "Concentration Solvent 2"],
        "ternary": [
            "Concentration Solvent 1",
            "Concentration Solvent 2",
            "Concentration Solvent 3",
        ],
    }
    concentration_columns = concentration_columns_by_type.get(mixture_type)
    if concentration_columns is None:
        return pd.Series(True, index=group.index)

    numeric_concentrations = pd.DataFrame(
        {
            column: group[column].map(parse_numeric_value)
            for column in concentration_columns
            if column in group.columns
        },
        index=group.index,
    )
    if len(numeric_concentrations.columns) != len(concentration_columns):
        return pd.Series(False, index=group.index)

    numeric_ok = numeric_concentrations.notna().all(axis=1)
    concentration_sum = numeric_concentrations.sum(axis=1)
    sum_ok = (concentration_sum - 1.0).abs() <= CONCENTRATION_SUM_TOLERANCE
    return numeric_ok & sum_ok


def filter_valid_solubility_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    numeric = dataframe["Solubility"].map(parse_numeric_value)
    finite = numeric.map(lambda value: value is not None and math.isfinite(value))
    fraction = dataframe["Solubility Unit"].isin(FRACTION_UNITS)
    in_range = numeric.map(
        lambda value: value is not None and 0.0 <= value <= 1.0
    )
    return dataframe[finite & (~fraction | in_range)].copy().reset_index(drop=True)


def filter_supported_solubility_units(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    return dataframe[dataframe["Solubility Unit"].isin(FRACTION_UNITS)].copy().reset_index(
        drop=True
    )


def filter_entries_like_reference(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
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
        smiles_columns = [
            column
            for column in smiles_by_type[mixture_type]
            if column in group.columns
        ]
        smiles_ok = (
            group[smiles_columns]
            .apply(lambda column: column.map(is_valid_species))
            .all(axis=1)
        )
        if "Temperature" in group.columns:
            temperature_ok = group["Temperature"].map(has_value)
        else:
            temperature_ok = pd.Series(False, index=group.index)

        if mixture_type == "single":
            unit_ok = group["Concentration Unit"].isin(FRACTION_UNITS) | group[
                "Concentration Unit"
            ].map(is_missing)
            unit_ok &= group["Concentration Solvent 1"].map(
                single_concentration_value_ok
            )
            concentration_sum_ok_mask = pd.Series(True, index=group.index)
        else:
            unit_ok = group["Concentration Unit"].isin(FRACTION_UNITS)
            if mixture_type == "binary":
                values_ok = group["Concentration Solvent 1"].map(
                    has_value
                ) & group["Concentration Solvent 2"].map(has_value)
            elif mixture_type == "ternary":
                values_ok = (
                    group["Concentration Solvent 1"].map(has_value)
                    & group["Concentration Solvent 2"].map(has_value)
                    & group["Concentration Solvent 3"].map(has_value)
                )
            else:
                values_ok = pd.Series(True, index=group.index)
            unit_ok &= values_ok
            concentration_sum_ok_mask = concentration_sum_ok(group, mixture_type)

        remaining = pd.Series(True, index=group.index)
        smiles_removed = remaining & ~smiles_ok
        remaining &= smiles_ok
        unit_removed = remaining & ~unit_ok
        remaining &= unit_ok
        concentration_sum_removed = remaining & ~concentration_sum_ok_mask
        remaining &= concentration_sum_ok_mask
        temperature_removed = remaining & ~temperature_ok
        remaining &= temperature_ok
        extra_scope_removed = (
            remaining.copy()
            if mixture_type == "extra"
            else pd.Series(False, index=group.index)
        )
        remaining &= ~extra_scope_removed

        filtered = group[remaining].copy()
        stats[mixture_type] = {
            "before": int(len(group)),
            "after": int(len(filtered)),
            "smiles_removed": int(smiles_removed.sum()),
            "unit_removed": int(unit_removed.sum()),
            "solubility_unit_removed": 0,
            "concentration_sum_removed": int(concentration_sum_removed.sum()),
            "temperature_removed": int(temperature_removed.sum()),
            "extra_scope_removed": int(extra_scope_removed.sum()),
            "both_removed": 0,
        }
        if not filtered.empty:
            filtered_parts.append(filtered)

    if not filtered_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), stats

    combined = pd.concat(filtered_parts, ignore_index=True)
    combined = combined.drop(columns=["_mixture_type"], errors="ignore")
    combined = combined[[column for column in OUTPUT_COLUMNS if column in combined.columns]]
    return combined, stats
