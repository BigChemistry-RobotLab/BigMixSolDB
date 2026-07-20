from __future__ import annotations

import argparse
import os
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

from bigmixsoldb.doi import normalize_doi

SMILES_COLUMNS = [
    "SMILES_Compound",
    "SMILES_Solvent_1",
    "SMILES_Solvent_2",
    "SMILES_Solvent_3",
]

SOLVENT_COLUMNS = [
    ("SMILES_Solvent_1", "Concentration Solvent 1"),
    ("SMILES_Solvent_2", "Concentration Solvent 2"),
    ("SMILES_Solvent_3", "Concentration Solvent 3"),
]

REQUIRED_COLUMNS = [
    *SMILES_COLUMNS,
    "Concentration Solvent 1",
    "Concentration Solvent 2",
    "Concentration Solvent 3",
    "Concentration Unit",
    "Solubility",
    "Solubility Unit",
    "Temperature",
    "Temperature Unit",
    "Pressure",
    "Pressure Unit",
    "doi",
]

CONDITION_COLUMNS = [
    "Concentration Unit",
    "Solubility",
    "Solubility Unit",
    "Temperature",
    "Temperature Unit",
]

PRECISION_COLUMNS = [
    "Solvent 1 Concentration",
    "Solvent 2 Concentration",
    "Solvent 3 Concentration",
    "Solubility",
    "Temperature",
]

ATMOSPHERIC_PRESSURE_PA = 101_325.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse cross-DOI duplicate solubility measurements using normalized "
            "keys and conservative numeric tolerances. "
            "Mixture solvent components are matched as unordered "
            "(solvent InChIKey, concentration) pairs by default."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="CSV to clean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. If omitted, the input CSV is overwritten.",
    )
    parser.add_argument(
        "--doi-source",
        type=Path,
        help=(
            "Optional CSV used only to collect DOI values for each measurement key. "
            "Useful when duplicate rows were already removed from the input."
        ),
    )
    parser.add_argument(
        "--duplicate-report",
        type=Path,
        help="Optional CSV report with one row per cross-DOI duplicate measurement group.",
    )
    parser.add_argument(
        "--doi-duplicate-report",
        type=Path,
        help=(
            "Optional CSV report with one row per DOI participating in any "
            "cross-DOI duplicate group."
        ),
    )
    parser.add_argument(
        "--ordered-solvents",
        action="store_true",
        help=(
            "Match solvent columns positionally, reproducing the older behavior. "
            "By default, solvent component-concentration pairs are matched "
            "order-insensitively."
        ),
    )
    parser.add_argument(
        "--keep-stereochemistry",
        action="store_true",
        help=(
            "Keep stereochemistry when deriving InChIKeys. By default, "
            "stereochemistry is removed before matching."
        ),
    )
    parser.add_argument(
        "--pressure-mode",
        choices=["relative-tolerance", "strict", "ignore"],
        default="relative-tolerance",
        help=(
            "How pressure participates in duplicate matching. "
            "'relative-tolerance' treats missing pressure as unspecified and "
            "recognized pressure values within the configured relative tolerance "
            "as equivalent; 'strict' requires exact pressure/unit equality; "
            "'ignore' removes pressure from the duplicate key."
        ),
    )
    parser.add_argument(
        "--pressure-relative-tolerance",
        type=float,
        default=0.02,
        help=(
            "Symmetric relative tolerance used between recognized pressure values "
            "after conversion to Pa. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--precision-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Match values that differ only by consistently reported precision. Enabled by default."
        ),
    )
    parser.add_argument(
        "--same-doi-duplicates",
        action="store_true",
        help=(
            "Also collapse duplicate rows when all duplicates have the same DOI. "
            "By default, only duplicate groups spanning multiple DOI are collapsed."
        ),
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, input_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"{input_path} is missing required column(s): {joined}")


def split_dois(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [doi.strip() for doi in str(value).split(";") if doi.strip()]


def normalized_split_dois(value: object) -> list[str]:
    return [normalize_doi(doi) for doi in split_dois(value)]


def join_dois(values: Iterable[object]) -> str | float:
    dois: list[str] = []
    seen: set[str] = set()
    for value in values:
        for doi in split_dois(value):
            normalized = normalize_doi(doi)
            if normalized not in seen:
                seen.add(normalized)
                dois.append(doi)
    if not dois:
        return float("nan")
    return "; ".join(dois)


def is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def first_non_missing(values: pd.Series) -> object:
    for value in values:
        if not is_missing(value):
            return value
    return values.iloc[0] if len(values) else float("nan")


class StructureKeyer:
    def __init__(self, *, keep_stereochemistry: bool) -> None:
        self.keep_stereochemistry = keep_stereochemistry
        self._cache: dict[str, object] = {}

    def inchikey(self, value: object) -> object:
        if pd.isna(value):
            return value

        smiles = str(value).strip()
        if not smiles:
            return value
        if smiles in self._cache:
            return self._cache[smiles]

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            result: object = smiles
        else:
            if not self.keep_stereochemistry:
                Chem.RemoveStereochemistry(molecule)
            result = inchi.MolToInchiKey(molecule)
        self._cache[smiles] = result
        return result


def pressure_to_pa(pressure: object, unit: object) -> float | None:
    if is_missing(pressure):
        return None
    try:
        value = float(pressure)
    except (TypeError, ValueError):
        return None

    if is_missing(unit):
        return value

    normalized_unit = str(unit).strip().lower()
    multipliers = {
        "pa": 1.0,
        "kpa": 1_000.0,
        "mpa": 1_000_000.0,
        "bar": 100_000.0,
        "atm": ATMOSPHERIC_PRESSURE_PA,
        "torr": ATMOSPHERIC_PRESSURE_PA / 760.0,
        "mmhg": ATMOSPHERIC_PRESSURE_PA / 760.0,
        "psi": 6_894.757293168,
    }
    multiplier = multipliers.get(normalized_unit)
    if multiplier is None:
        return None
    return value * multiplier


def pressure_key(
    pressure: object,
    unit: object,
    *,
    pressure_mode: str,
    pressure_relative_tolerance: float = 0.02,
) -> tuple[object, object]:
    if pressure_mode == "ignore":
        return ("__ignored__", "__ignored__")
    if pressure_mode == "strict":
        return (pressure, unit)
    if is_missing(pressure):
        return ("__missing__", "__missing__")

    pressure_pa = pressure_to_pa(pressure, unit)
    if pressure_pa is not None:
        return (round(pressure_pa, 6), "Pa")
    return (pressure, unit)


def pressures_equivalent(
    left_pressure: object,
    left_unit: object,
    right_pressure: object,
    right_unit: object,
    *,
    pressure_mode: str,
    relative_tolerance: float,
) -> bool:
    if pressure_mode == "ignore":
        return True
    if pressure_mode == "strict":
        pressure_equal = (
            is_missing(left_pressure) and is_missing(right_pressure)
        ) or left_pressure == right_pressure
        unit_equal = (is_missing(left_unit) and is_missing(right_unit)) or left_unit == right_unit
        return bool(pressure_equal and unit_equal)
    if is_missing(left_pressure) or is_missing(right_pressure):
        return True

    left_pa = pressure_to_pa(left_pressure, left_unit)
    right_pa = pressure_to_pa(right_pressure, right_unit)
    if left_pa is None or right_pa is None:
        return left_pressure == right_pressure and left_unit == right_unit
    scale = max(abs(left_pa), abs(right_pa))
    if scale == 0:
        return left_pa == right_pa
    return abs(left_pa - right_pa) / scale <= relative_tolerance


def partition_by_pressure(
    indices: Iterable[object],
    df: pd.DataFrame,
    *,
    pressure_mode: str,
    relative_tolerance: float,
) -> list[list[object]]:
    """Build deterministic complete-link groups under pairwise pressure equivalence."""
    groups: list[list[object]] = []
    for index in indices:
        for group in groups:
            if all(
                pressures_equivalent(
                    df.at[index, "Pressure"],
                    df.at[index, "Pressure Unit"],
                    df.at[member, "Pressure"],
                    df.at[member, "Pressure Unit"],
                    pressure_mode=pressure_mode,
                    relative_tolerance=relative_tolerance,
                )
                for member in group
            ):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def sort_value(value: object) -> str:
    if is_missing(value):
        return ""
    return str(value)


def canonical_solvent_pairs(
    row: pd.Series,
    *,
    structure_keyer: StructureKeyer,
    ordered_solvents: bool,
) -> list[tuple[object, object]]:
    pairs: list[tuple[object, object]] = []
    for solvent_column, concentration_column in SOLVENT_COLUMNS:
        solvent = structure_keyer.inchikey(row[solvent_column])
        concentration = row[concentration_column]
        if is_missing(solvent) and is_missing(concentration):
            continue
        pairs.append((solvent, concentration))

    if not ordered_solvents:
        pairs.sort(key=lambda pair: (sort_value(pair[0]), sort_value(pair[1])))

    while len(pairs) < len(SOLVENT_COLUMNS):
        pairs.append((float("nan"), float("nan")))
    return pairs[: len(SOLVENT_COLUMNS)]


def key_dataframe(
    df: pd.DataFrame,
    *,
    keep_stereochemistry: bool,
    ordered_solvents: bool,
    pressure_mode: str,
    pressure_relative_tolerance: float = 0.02,
) -> pd.DataFrame:
    structure_keyer = StructureKeyer(keep_stereochemistry=keep_stereochemistry)
    key = pd.DataFrame(index=df.index)
    key["InChIKey_Compound"] = df["SMILES_Compound"].map(structure_keyer.inchikey)

    solvent_keys = [
        df[solvent_column].map(structure_keyer.inchikey) for solvent_column, _ in SOLVENT_COLUMNS
    ]
    concentrations = [df[column] for _, column in SOLVENT_COLUMNS]
    solvent_pairs: list[list[tuple[object, object]]] = []
    for values in zip(*solvent_keys, *concentrations):
        pairs = [
            (values[position], values[position + len(SOLVENT_COLUMNS)])
            for position in range(len(SOLVENT_COLUMNS))
            if not (
                is_missing(values[position]) and is_missing(values[position + len(SOLVENT_COLUMNS)])
            )
        ]
        if not ordered_solvents:
            pairs.sort(key=lambda pair: (sort_value(pair[0]), sort_value(pair[1])))
        pairs.extend([(float("nan"), float("nan"))] * (len(SOLVENT_COLUMNS) - len(pairs)))
        solvent_pairs.append(pairs[: len(SOLVENT_COLUMNS)])
    for index in range(len(SOLVENT_COLUMNS)):
        key[f"Solvent {index + 1} InChIKey"] = [pairs[index][0] for pairs in solvent_pairs]
        key[f"Solvent {index + 1} Concentration"] = [pairs[index][1] for pairs in solvent_pairs]

    for column in CONDITION_COLUMNS:
        key[column] = df[column]

    pressure_keys = [
        pressure_key(
            pressure,
            unit,
            pressure_mode=pressure_mode,
            pressure_relative_tolerance=pressure_relative_tolerance,
        )
        for pressure, unit in zip(df["Pressure"], df["Pressure Unit"])
    ]
    key["Pressure"] = [value for value, _ in pressure_keys]
    key["Pressure Unit"] = [unit for _, unit in pressure_keys]
    return key


def raw_token_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return string tokens when a caller did not read a separate raw CSV."""
    raw = pd.DataFrame(index=df.index)
    for column in df.columns:
        raw[column] = df[column].map(lambda value: "" if is_missing(value) else str(value))
    return raw


def read_csv_with_tokens(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed = pd.read_csv(path, low_memory=False)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    raw.index = parsed.index
    return parsed, raw


def precision_token_dataframe(
    df: pd.DataFrame,
    raw_tokens: pd.DataFrame,
    *,
    keep_stereochemistry: bool,
    ordered_solvents: bool,
) -> pd.DataFrame:
    structure_keyer = StructureKeyer(keep_stereochemistry=keep_stereochemistry)
    tokens = pd.DataFrame(index=df.index)
    solvent_keys = [
        df[solvent_column].map(structure_keyer.inchikey) for solvent_column, _ in SOLVENT_COLUMNS
    ]
    concentrations = [df[column] for _, column in SOLVENT_COLUMNS]
    raw_concentrations = [raw_tokens[column] for _, column in SOLVENT_COLUMNS]
    solvent_tokens: list[list[str]] = []
    for values in zip(*solvent_keys, *concentrations, *raw_concentrations):
        pairs: list[tuple[object, object, str]] = []
        width = len(SOLVENT_COLUMNS)
        for position in range(width):
            solvent = values[position]
            concentration = values[position + width]
            token = values[position + 2 * width]
            if is_missing(solvent) and is_missing(concentration):
                continue
            pairs.append((solvent, concentration, token))
        if not ordered_solvents:
            pairs.sort(key=lambda pair: (sort_value(pair[0]), sort_value(pair[1])))
        values = [pair[2] for pair in pairs]
        values.extend([""] * (len(SOLVENT_COLUMNS) - len(values)))
        solvent_tokens.append(values[: len(SOLVENT_COLUMNS)])

    for position in range(len(SOLVENT_COLUMNS)):
        tokens[f"Solvent {position + 1} Concentration"] = [
            values[position] for values in solvent_tokens
        ]
    tokens["Solubility"] = raw_tokens["Solubility"]
    tokens["Temperature"] = raw_tokens["Temperature"]
    return tokens


def decimal_from_token(token: object) -> Decimal | None:
    if is_missing(token):
        return None
    try:
        value = Decimal(str(token).strip())
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def significant_digits(value: Decimal) -> int:
    if value.is_zero():
        return 1
    return len(value.as_tuple().digits)


def precision_relation(
    left: pd.Series,
    right: pd.Series,
) -> tuple[str, list[str]] | None:
    """Return the consistently coarser side, or ``conflict`` for mixed direction."""
    directions: set[str] = set()
    differing: list[str] = []
    for column in PRECISION_COLUMNS:
        left_value = decimal_from_token(left[column])
        right_value = decimal_from_token(right[column])
        if (left_value is None) != (right_value is None):
            return None
        if left_value is None or left_value == right_value:
            continue
        differing.append(column)

        if left_value.as_tuple().exponent > right_value.as_tuple().exponent:
            coarse_name, coarse, fine = "left", left_value, right_value
        elif right_value.as_tuple().exponent > left_value.as_tuple().exponent:
            coarse_name, coarse, fine = "right", right_value, left_value
        else:
            return None

        if significant_digits(coarse) < 3:
            return None
        quantum = Decimal(1).scaleb(coarse.as_tuple().exponent)
        if fine.quantize(quantum, rounding=ROUND_HALF_UP) != coarse:
            return None
        directions.add(coarse_name)

    if not differing:
        return None
    if len(directions) > 1:
        return "conflict", differing
    return next(iter(directions)), differing


def plausible_candidate_pairs(
    indices: list[object],
    numeric_tokens: pd.DataFrame,
) -> Iterable[tuple[object, object]]:
    """Yield a conservative numeric window containing every valid precision pair."""
    indices = [
        index
        for index in indices
        if all(
            is_missing(numeric_tokens.at[index, column])
            or decimal_from_token(numeric_tokens.at[index, column]) is not None
            for column in PRECISION_COLUMNS
        )
    ]
    if len(indices) < 2:
        return
    decimal_columns = {
        column: [decimal_from_token(numeric_tokens.at[index, column]) for index in indices]
        for column in PRECISION_COLUMNS
    }
    chosen = max(
        PRECISION_COLUMNS,
        key=lambda column: len({value for value in decimal_columns[column] if value is not None}),
    )
    if len({value for value in decimal_columns[chosen] if value is not None}) < 2:
        return

    ordered = sorted(zip(decimal_columns[chosen], indices), key=lambda item: item[0])
    relative_window = Decimal("0.0051")
    for left_position, (left_value, left_index) in enumerate(ordered):
        if left_value is None:
            continue
        for right_value, right_index in ordered[left_position + 1 :]:
            if right_value is None:
                continue
            scale = max(abs(left_value), abs(right_value))
            if scale.is_zero() or abs(right_value - left_value) <= scale * relative_window:
                yield left_index, right_index
                continue
            break


def prefixed_key_dataframe(key: pd.DataFrame) -> pd.DataFrame:
    return key.rename(columns={column: f"__key_{column}" for column in key.columns})


def map_group_values(key: pd.DataFrame, values_by_group: pd.Series) -> pd.Series:
    key_index = pd.MultiIndex.from_frame(prefixed_key_dataframe(key))
    return pd.Series(key_index.map(values_by_group), index=key.index)


def doi_lookup(
    source: pd.DataFrame,
    *,
    keep_stereochemistry: bool,
    ordered_solvents: bool,
    pressure_mode: str,
    pressure_relative_tolerance: float = 0.02,
) -> pd.Series:
    source_key = prefixed_key_dataframe(
        key_dataframe(
            source,
            keep_stereochemistry=keep_stereochemistry,
            ordered_solvents=ordered_solvents,
            pressure_mode=pressure_mode,
            pressure_relative_tolerance=pressure_relative_tolerance,
        )
    )
    source_work = pd.concat([source_key, source["doi"].rename("__doi")], axis=1)
    return source_work.groupby(list(source_key.columns), dropna=False, sort=False)["__doi"].agg(
        join_dois
    )


def merge_dois_from_source(
    df: pd.DataFrame,
    source: pd.DataFrame,
    *,
    keep_stereochemistry: bool,
    ordered_solvents: bool,
    pressure_mode: str,
    pressure_relative_tolerance: float = 0.02,
) -> pd.DataFrame:
    merged = df.copy()
    source_lookup = doi_lookup(
        source,
        keep_stereochemistry=keep_stereochemistry,
        ordered_solvents=ordered_solvents,
        pressure_mode=pressure_mode,
        pressure_relative_tolerance=pressure_relative_tolerance,
    )
    source_key = key_dataframe(
        merged,
        keep_stereochemistry=keep_stereochemistry,
        ordered_solvents=ordered_solvents,
        pressure_mode=pressure_mode,
        pressure_relative_tolerance=pressure_relative_tolerance,
    )
    source_values = map_group_values(source_key, source_lookup)
    merged["doi"] = source_values.where(source_values.notna(), merged["doi"])
    return merged


def doi_row_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in df["doi"]:
        for doi in normalized_split_dois(value):
            counts[doi] += 1
    return dict(counts)


def update_doi_stats(
    stats: dict[str, dict[str, int]],
    group: pd.DataFrame,
    *,
    representative_index: int,
) -> None:
    group_dois = sorted({doi for value in group["doi"] for doi in normalized_split_dois(value)})
    removed_group = group.drop(index=representative_index)
    for doi in group_dois:
        stats[doi]["duplicate_groups"] += 1
        stats[doi]["rows_in_duplicate_groups"] += int(
            group["doi"].map(lambda value: doi in normalized_split_dois(value)).sum()
        )
        stats[doi]["rows_removed_as_duplicates"] += int(
            removed_group["doi"].map(lambda value: doi in normalized_split_dois(value)).sum()
        )
        if doi in normalized_split_dois(group.at[representative_index, "doi"]):
            stats[doi]["retained_as_original_representative"] += 1


def merge_group_into_representative(
    merged: pd.DataFrame,
    group: pd.DataFrame,
    *,
    representative_index: int,
) -> None:
    for column in merged.columns:
        if column == "doi":
            merged.at[representative_index, column] = join_dois(group[column])
        elif is_missing(merged.at[representative_index, column]):
            merged.at[representative_index, column] = first_non_missing(group[column])


def duplicate_report_row(
    key_group: pd.DataFrame,
    data_group: pd.DataFrame,
    *,
    representative_index: int,
    collapsible_rows: int,
    match_type: str = "exact",
    match_status: str = "collapsed",
    differing_columns: Iterable[str] = (),
    candidate_original_indices: Iterable[object] = (),
) -> dict[str, object]:
    merged_dois = join_dois(data_group["doi"])
    doi_values = split_dois(merged_dois)
    representative = key_group.loc[representative_index].to_dict()
    representative.update(
        {
            "rows": len(data_group),
            "collapsible_rows": collapsible_rows,
            "doi_count": len(doi_values),
            "doi": "; ".join(doi_values),
            "representative_original_index": representative_index,
            "match_type": match_type,
            "match_status": match_status,
            "differing_columns": "; ".join(differing_columns),
            "candidate_original_indices": "; ".join(
                str(index) for index in candidate_original_indices
            ),
        }
    )
    return representative


def deduplicate(
    df: pd.DataFrame,
    key: pd.DataFrame,
    *,
    same_doi_duplicates: bool,
    raw_tokens: pd.DataFrame | None = None,
    precision_aware: bool = True,
    keep_stereochemistry: bool = False,
    ordered_solvents: bool = False,
    pressure_mode: str = "relative-tolerance",
    pressure_relative_tolerance: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]], int]:
    merged = df.copy()
    keep_mask = pd.Series(True, index=df.index)
    prefixed_key = prefixed_key_dataframe(key)
    group_columns = list(prefixed_key.columns)
    if pressure_mode == "relative-tolerance":
        group_columns = [
            column
            for column in group_columns
            if column not in {"__key_Pressure", "__key_Pressure Unit"}
        ]
    duplicate_key_mask = prefixed_key.duplicated(subset=group_columns, keep=False)
    work = pd.concat([prefixed_key.loc[duplicate_key_mask], df.loc[duplicate_key_mask]], axis=1)
    report_rows: list[dict[str, object]] = []
    doi_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    collapsed_rows = 0

    for _, base_group in work.groupby(group_columns, dropna=False, sort=False):
        pressure_groups = (
            partition_by_pressure(
                base_group.index,
                df,
                pressure_mode=pressure_mode,
                relative_tolerance=pressure_relative_tolerance,
            )
            if pressure_mode == "relative-tolerance"
            else [list(base_group.index)]
        )
        for indices in pressure_groups:
            if len(indices) < 2:
                continue
            data_group = df.loc[indices]
            doi_values = split_dois(join_dois(data_group["doi"]))
            is_cross_doi = len(doi_values) > 1
            should_collapse = same_doi_duplicates or is_cross_doi
            representative_index = int(data_group.index[0])

            if should_collapse:
                merge_group_into_representative(
                    merged,
                    data_group,
                    representative_index=representative_index,
                )
                duplicate_indices = data_group.index.drop(representative_index)
                keep_mask.loc[duplicate_indices] = False
                collapsed_rows += len(duplicate_indices)

            if is_cross_doi:
                update_doi_stats(
                    doi_stats,
                    data_group,
                    representative_index=representative_index,
                )
                pressure_representations = {
                    (sort_value(row["Pressure"]), sort_value(row["Pressure Unit"]))
                    for _, row in data_group.iterrows()
                }
                pressure_match = (
                    pressure_mode == "relative-tolerance" and len(pressure_representations) > 1
                )
                report_rows.append(
                    duplicate_report_row(
                        key.loc[indices],
                        data_group,
                        representative_index=representative_index,
                        collapsible_rows=(len(data_group) - 1 if should_collapse else 0),
                        match_type="pressure_tolerance" if pressure_match else "exact",
                        match_status="collapsed" if should_collapse else "skipped_ambiguous",
                        differing_columns=(("Pressure", "Pressure Unit") if pressure_match else ()),
                        candidate_original_indices=data_group.index,
                    )
                )

    if precision_aware:
        if raw_tokens is None:
            raw_tokens = raw_token_dataframe(df)
        numeric_tokens = precision_token_dataframe(
            df,
            raw_tokens,
            keep_stereochemistry=keep_stereochemistry,
            ordered_solvents=ordered_solvents,
        )
        remaining_indices = list(df.index[keep_mask])
        base_columns = [column for column in key.columns if column not in PRECISION_COLUMNS]
        if pressure_mode == "relative-tolerance":
            base_columns = [
                column for column in base_columns if column not in {"Pressure", "Pressure Unit"}
            ]
        base = prefixed_key_dataframe(key.loc[remaining_indices, base_columns])
        for column in PRECISION_COLUMNS:
            base[f"__missing_{column}"] = key.loc[remaining_indices, column].map(is_missing)
        base_work = pd.concat([base, merged.loc[remaining_indices, ["doi"]]], axis=1)

        for _, base_group in base_work.groupby(list(base.columns), dropna=False, sort=False):
            indices = list(base_group.index)
            if len(indices) < 2:
                continue

            adjacency: dict[object, set[object]] = defaultdict(set)
            edge_info: dict[tuple[object, object], tuple[str, list[str]]] = {}
            for left_index, right_index in plausible_candidate_pairs(indices, numeric_tokens):
                if not pressures_equivalent(
                    merged.at[left_index, "Pressure"],
                    merged.at[left_index, "Pressure Unit"],
                    merged.at[right_index, "Pressure"],
                    merged.at[right_index, "Pressure Unit"],
                    pressure_mode=pressure_mode,
                    relative_tolerance=pressure_relative_tolerance,
                ):
                    continue
                pair = merged.loc[[left_index, right_index]]
                pair_dois = {doi for value in pair["doi"] for doi in normalized_split_dois(value)}
                if not same_doi_duplicates and len(pair_dois) < 2:
                    continue
                relation = precision_relation(
                    numeric_tokens.loc[left_index], numeric_tokens.loc[right_index]
                )
                if relation is None:
                    continue
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                edge_info[(left_index, right_index)] = relation

            unseen = set(adjacency)
            while unseen:
                start = next(iter(unseen))
                component: set[object] = set()
                stack = [start]
                while stack:
                    current = stack.pop()
                    if current in component:
                        continue
                    component.add(current)
                    stack.extend(adjacency[current] - component)
                unseen -= component

                component_edges = {
                    pair: info
                    for pair, info in edge_info.items()
                    if pair[0] in component and pair[1] in component
                }
                differing = sorted(
                    {column for _, columns in component_edges.values() for column in columns}
                )
                ordered_component = [index for index in indices if index in component]
                isolated_pair = len(component) == 2 and len(component_edges) == 1
                relation = next(iter(component_edges.values()))[0]
                can_collapse = isolated_pair and relation != "conflict"

                if can_collapse:
                    left_index, right_index = next(iter(component_edges))
                    representative_index = right_index if relation == "left" else left_index
                    data_group = merged.loc[ordered_component]
                    merge_group_into_representative(
                        merged,
                        data_group,
                        representative_index=representative_index,
                    )
                    duplicate_index = (
                        left_index if representative_index == right_index else right_index
                    )
                    keep_mask.at[duplicate_index] = False
                    collapsed_rows += 1
                    component_dois = {
                        doi for value in data_group["doi"] for doi in normalized_split_dois(value)
                    }
                    if len(component_dois) > 1:
                        update_doi_stats(
                            doi_stats,
                            data_group,
                            representative_index=representative_index,
                        )
                        report_rows.append(
                            duplicate_report_row(
                                key.loc[ordered_component],
                                data_group,
                                representative_index=int(representative_index),
                                collapsible_rows=1,
                                match_type="reported_precision",
                                match_status="collapsed",
                                differing_columns=differing,
                                candidate_original_indices=ordered_component,
                            )
                        )
                else:
                    data_group = merged.loc[ordered_component]
                    representative_index = ordered_component[0]
                    component_dois = {
                        doi for value in data_group["doi"] for doi in normalized_split_dois(value)
                    }
                    if len(component_dois) > 1:
                        report_rows.append(
                            duplicate_report_row(
                                key.loc[ordered_component],
                                data_group,
                                representative_index=int(representative_index),
                                collapsible_rows=0,
                                match_type="ambiguous_reported_precision",
                                match_status="skipped_ambiguous",
                                differing_columns=differing,
                                candidate_original_indices=ordered_component,
                            )
                        )

    cleaned = merged.loc[keep_mask].copy()
    return cleaned, pd.DataFrame(report_rows), doi_stats, collapsed_rows


def doi_duplicate_report(
    df: pd.DataFrame,
    *,
    doi_stats: dict[str, dict[str, int]],
) -> pd.DataFrame:
    total_counts = doi_row_counts(df)
    columns = [
        "doi",
        "total_rows_in_input",
        "duplicate_groups",
        "rows_in_duplicate_groups",
        "rows_removed_as_duplicates",
        "retained_as_original_representative",
    ]
    rows: list[dict[str, object]] = []
    for doi in sorted(doi_stats):
        stats = doi_stats[doi]
        rows.append(
            {
                "doi": doi,
                "total_rows_in_input": total_counts.get(doi, 0),
                "duplicate_groups": stats.get("duplicate_groups", 0),
                "rows_in_duplicate_groups": stats.get("rows_in_duplicate_groups", 0),
                "rows_removed_as_duplicates": stats.get("rows_removed_as_duplicates", 0),
                "retained_as_original_representative": stats.get(
                    "retained_as_original_representative", 0
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    report = pd.DataFrame(rows, columns=columns)
    return report.sort_values(
        ["duplicate_groups", "rows_in_duplicate_groups", "doi"],
        ascending=[False, False, True],
    )


def write_csv(df: pd.DataFrame, output_path: Path, *, overwrite_input: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_input:
        df.to_csv(output_path, index=False)
        return

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    df.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def print_summary(
    *,
    input_path: Path,
    output_path: Path,
    rows_before: int,
    rows_after: int,
    collapsed_rows: int,
    report: pd.DataFrame,
    doi_report: pd.DataFrame,
    ordered_solvents: bool,
    pressure_mode: str,
    keep_stereochemistry: bool,
    precision_aware: bool,
) -> None:
    solvent_mode = (
        "ordered solvent columns" if ordered_solvents else "unordered solvent-concentration pairs"
    )
    stereo_mode = "stereochemistry kept" if keep_stereochemistry else "stereochemistry ignored"
    precision_mode = "precision-aware" if precision_aware else "exact numeric matching"
    print(
        f"Cleaned {input_path} "
        f"({solvent_mode}; {stereo_mode}; pressure={pressure_mode}; "
        f"{precision_mode}; file-order retention)."
    )
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after : {rows_after:,}")
    print(f"Collapsed  : {collapsed_rows:,}")
    if {"match_status", "match_type"}.issubset(report.columns):
        collapsed_report = report[report["match_status"] == "collapsed"]
        pressure_count = int((collapsed_report["match_type"] == "pressure_tolerance").sum())
        precision_count = int((collapsed_report["match_type"] == "reported_precision").sum())
        ambiguous_count = int((report["match_status"] == "skipped_ambiguous").sum())
    else:
        pressure_count = precision_count = ambiguous_count = 0
    print(f"Pressure-tolerance groups collapsed  : {pressure_count:,}")
    print(f"Reported-precision groups collapsed  : {precision_count:,}")
    print(f"Ambiguous precision groups skipped   : {ambiguous_count:,}")
    print(f"Output     : {output_path}")

    if report.empty:
        print("Cross-DOI duplicate groups: 0")
    else:
        print(f"Cross-DOI duplicate groups: {len(report):,}")
        print(f"Rows in cross-DOI duplicate groups: {int(report['rows'].sum()):,}")
        print(f"Collapsible cross-DOI rows: {int(report['collapsible_rows'].sum()):,}")
    print(f"DOIs participating in cross-DOI duplicates: {len(doi_report):,}")


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path

    if input_path is None:
        raise SystemExit("An input CSV path is required.")
    if args.pressure_relative_tolerance < 0:
        raise SystemExit("--pressure-relative-tolerance must be non-negative.")

    df, raw_tokens = read_csv_with_tokens(input_path)
    validate_columns(df, input_path)

    if args.doi_source:
        source = pd.read_csv(args.doi_source, low_memory=False)
        validate_columns(source, args.doi_source)
        df = merge_dois_from_source(
            df,
            source,
            keep_stereochemistry=args.keep_stereochemistry,
            ordered_solvents=args.ordered_solvents,
            pressure_mode=args.pressure_mode,
            pressure_relative_tolerance=args.pressure_relative_tolerance,
        )

    key = key_dataframe(
        df,
        keep_stereochemistry=args.keep_stereochemistry,
        ordered_solvents=args.ordered_solvents,
        pressure_mode=args.pressure_mode,
        pressure_relative_tolerance=args.pressure_relative_tolerance,
    )
    cleaned, report, doi_stats, collapsed_rows = deduplicate(
        df,
        key,
        same_doi_duplicates=args.same_doi_duplicates,
        raw_tokens=raw_tokens,
        precision_aware=args.precision_aware,
        keep_stereochemistry=args.keep_stereochemistry,
        ordered_solvents=args.ordered_solvents,
        pressure_mode=args.pressure_mode,
        pressure_relative_tolerance=args.pressure_relative_tolerance,
    )
    doi_report = doi_duplicate_report(df, doi_stats=doi_stats)

    write_csv(cleaned, output_path, overwrite_input=output_path == input_path)
    if args.duplicate_report:
        args.duplicate_report.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.duplicate_report, index=False)
    if args.doi_duplicate_report:
        args.doi_duplicate_report.parent.mkdir(parents=True, exist_ok=True)
        doi_report.to_csv(args.doi_duplicate_report, index=False)
    print_summary(
        input_path=input_path,
        output_path=output_path,
        rows_before=len(df),
        rows_after=len(cleaned),
        collapsed_rows=collapsed_rows,
        report=report,
        doi_report=doi_report,
        ordered_solvents=args.ordered_solvents,
        pressure_mode=args.pressure_mode,
        keep_stereochemistry=args.keep_stereochemistry,
        precision_aware=args.precision_aware,
    )
    if args.duplicate_report:
        print(f"Duplicate report   : {args.duplicate_report}")
    if args.doi_duplicate_report:
        print(f"DOI report         : {args.doi_duplicate_report}")


if __name__ == "__main__":
    main()
