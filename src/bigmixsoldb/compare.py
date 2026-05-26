from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

from bigmixsoldb.postprocess import is_missing, normalize_unit_text

RDLogger.DisableLog("rdApp.*")

ROUND_SOLUBILITY = 10
ROUND_TEMPERATURE = 2
ROUND_FRACTION = 4
EPS = 1e-9


def clean_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def normalize_doi(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return ""
    text = text.replace("_", "/").replace("", ":")
    text = text.lower()
    text = re.sub(r"\s*\(.*?\)\s*$", "", text)
    return text.strip()


@lru_cache(maxsize=50_000)
def _canonicalize_smiles_text(text: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_smiles(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return _canonicalize_smiles_text(text)


def _required_reference_smiles(
    row: pd.Series,
    fields: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, Any] | None]:
    smiles_by_field: dict[str, str] = {}
    missing_fields: list[str] = []
    raw_values: dict[str, str | None] = {}

    for field in fields:
        raw_values[field] = clean_text(row.get(field))
        smiles = canonicalize_smiles(row.get(field))
        if smiles is None:
            missing_fields.append(field)
        else:
            smiles_by_field[field] = smiles

    if missing_fields:
        return smiles_by_field, {
            "reason": "missing_reference_smiles",
            "missing_smiles_fields": missing_fields,
            "raw_smiles_values": {field: raw_values[field] for field in missing_fields},
        }
    return smiles_by_field, None


@lru_cache(maxsize=50_000)
def _smiles_to_inchi(text: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        return inchi if inchi else None
    except Exception:
        return None


def smiles_to_inchi(value: Any) -> str | None:
    text = canonicalize_smiles(value)
    if text is None:
        return None
    return _smiles_to_inchi(text)


def smiles_to_inchikey(value: Any) -> str | None:
    inchi = smiles_to_inchi(value)
    if inchi is None:
        return None
    try:
        return InchiToInchiKey(inchi)
    except Exception:
        return None


def round_or_none(value: Any, digits: int) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return round(float(numeric), digits)


def classify_solubility(value: Any, unit: Any) -> tuple[str | None, float | None]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None, None
    numeric = float(numeric)
    unit_text = normalize_unit_text(unit)
    if unit_text is None:
        return None, None
    compact = unit_text.replace(" ", "")

    if compact in {"molefraction", "mol/mol", "molmol^-1", "molmol-1"}:
        return "mole_fraction", round(numeric, ROUND_SOLUBILITY)
    if compact.startswith("g/100g"):
        return "g_per_100g", round(numeric, ROUND_SOLUBILITY)
    if compact == "g/g":
        return "g_per_100g", round(numeric * 100.0, ROUND_SOLUBILITY)
    return None, None


def classify_fraction(value: Any, unit: Any) -> tuple[str | None, float | None]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None, None
    numeric = float(numeric)
    unit_text = normalize_unit_text(unit)
    if unit_text is None:
        return None, None
    compact = unit_text.replace(" ", "")

    direct = {
        "molefraction": "mole",
        "massfraction": "mass",
        "weightfraction": "mass",
        "volumefraction": "volume",
        "x": "mole",
        "w": "mass",
        "phi": "volume",
        "ϕ": "volume",
    }
    percent = {
        "%": "unknown_percent",
        "mol%": "mole",
        "mole%": "mole",
        "wt%": "mass",
        "weight%": "mass",
        "mass%": "mass",
        "vol%": "volume",
        "volume%": "volume",
    }

    if compact in direct:
        fraction_type = direct[compact]
        fraction_value = numeric
    elif compact in percent:
        fraction_type = percent[compact]
        fraction_value = numeric / 100.0
    else:
        return None, None

    if fraction_type == "unknown_percent":
        return None, None
    if not (0.0 - EPS <= fraction_value <= 1.0 + EPS):
        return None, None
    clipped = min(max(fraction_value, 0.0), 1.0)
    return fraction_type, round(clipped, ROUND_FRACTION)


def normalize_name_key(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text.lower()).strip()


def entity_identity(smiles: Any, name: Any) -> str | None:
    inchi = smiles_to_inchi(smiles)
    if inchi:
        return f"inchi:{inchi}"
    normalized_name = normalize_name_key(name)
    if normalized_name:
        return f"name:{normalized_name}"
    return None


@dataclass
class Record:
    dataset: str
    row_index: int
    doi: str
    mode: str
    metric: str
    temperature: float
    solubility: float
    fraction_type: str | None
    fraction_solvent1: float | None
    compound_name: str | None
    solute_smiles: str | None
    solvent1_name: str | None
    solvent2_name: str | None
    solvent1_smiles: str | None
    solvent2_smiles: str | None
    fraction_solvent1_raw: float | None = None


def record_composition(record: Record) -> tuple[float, float] | None:
    if record.mode != "binary" or record.fraction_solvent1 is None:
        return None
    f1 = round(record.fraction_solvent1, ROUND_FRACTION)
    f2 = round(1.0 - record.fraction_solvent1, ROUND_FRACTION)
    return tuple(sorted((f1, f2)))


def serialize_record(record: Record) -> dict[str, Any]:
    return {
        "dataset": record.dataset,
        "row_index": record.row_index,
        "doi": record.doi,
        "mode": record.mode,
        "metric": record.metric,
        "temperature": record.temperature,
        "solubility": record.solubility,
        "fraction_type": record.fraction_type,
        "fraction_solvent1": record.fraction_solvent1,
        "composition": record_composition(record),
        "compound_name": record.compound_name,
        "solute_smiles": record.solute_smiles,
        "solvent1_name": record.solvent1_name,
        "solvent2_name": record.solvent2_name,
        "solvent_smiles": [record.solvent1_smiles, record.solvent2_smiles],
    }


def standardized_row_to_record(row_index: int, row: pd.Series, dataset: str) -> tuple[Record | None, dict[str, Any] | None]:
    doi = normalize_doi(row.get("doi") or row.get("Source"))
    if not doi:
        return None, {"row_index": row_index, "reason": "missing_doi"}

    temperature = round_or_none(row.get("Temperature"), ROUND_TEMPERATURE)
    if temperature is None:
        return None, {"row_index": row_index, "reason": "missing_temperature", "doi": doi}

    metric, solubility = classify_solubility(row.get("Solubility"), row.get("Solubility Unit"))
    if metric is None or solubility is None:
        return None, {"row_index": row_index, "reason": "unsupported_solubility_unit", "doi": doi}

    solvents = []
    for idx in (1, 2, 3):
        name = clean_text(row.get(f"Solvent {idx}"))
        smiles = canonicalize_smiles(row.get(f"SMILES_Solvent_{idx}"))
        if name is None and smiles is None:
            continue
        solvents.append((name, smiles))

    if not is_missing(row.get("Extra Solvents")) or len(solvents) > 2:
        return None, {"row_index": row_index, "reason": "unsupported_mixture_order", "doi": doi}
    if not solvents:
        return None, {"row_index": row_index, "reason": "missing_solvent", "doi": doi}

    if len(solvents) == 1:
        return Record(
            dataset=dataset,
            row_index=row_index,
            doi=doi,
            mode="single",
            metric=metric,
            temperature=temperature,
            solubility=solubility,
            fraction_type=None,
            fraction_solvent1=None,
            compound_name=clean_text(row.get("Compound Name")),
            solute_smiles=canonicalize_smiles(row.get("SMILES_Compound")),
            solvent1_name=solvents[0][0],
            solvent2_name=None,
            solvent1_smiles=solvents[0][1],
            solvent2_smiles=None,
        ), None

    fraction_type, fraction_solvent1 = classify_fraction(
        row.get("Concentration Solvent 1"),
        row.get("Concentration Unit"),
    )
    if fraction_type is None or fraction_solvent1 is None:
        return None, {"row_index": row_index, "reason": "unsupported_fraction_unit", "doi": doi}

    return Record(
        dataset=dataset,
        row_index=row_index,
        doi=doi,
        mode="binary",
        metric=metric,
        temperature=temperature,
        solubility=solubility,
        fraction_type=fraction_type,
        fraction_solvent1=fraction_solvent1,
        compound_name=clean_text(row.get("Compound Name")),
        solute_smiles=canonicalize_smiles(row.get("SMILES_Compound")),
        solvent1_name=solvents[0][0],
        solvent2_name=solvents[1][0],
        solvent1_smiles=solvents[0][1],
        solvent2_smiles=solvents[1][1],
    ), None


def load_standardized_records(path: str | Path, dataset: str) -> tuple[list[Record], list[dict[str, Any]]]:
    dataframe = pd.read_csv(path, low_memory=False)
    records: list[Record] = []
    excluded: list[dict[str, Any]] = []
    for row_index, row in dataframe.iterrows():
        record, exclusion = standardized_row_to_record(row_index, row, dataset)
        if record is not None:
            records.append(record)
        elif exclusion is not None:
            excluded.append(exclusion)
    return records, excluded


def load_mixturesoldb_records(path: str | Path) -> tuple[list[Record], list[dict[str, Any]]]:
    dataframe = pd.read_csv(path, low_memory=False)
    records: list[Record] = []
    excluded: list[dict[str, Any]] = []
    for row_index, row in dataframe.iterrows():
        doi = normalize_doi(row.get("Source"))
        temperature = round_or_none(row.get("Temperature_K"), ROUND_TEMPERATURE)
        fraction_solvent1 = pd.to_numeric(row.get("Fraction_Solvent1"), errors="coerce")
        fraction_type = normalize_unit_text(row.get("Fraction_Type"))
        if not doi or temperature is None or pd.isna(fraction_solvent1):
            excluded.append({"row_index": row_index, "reason": "missing_required_fields", "doi": doi})
            continue

        measurements: list[tuple[str, float]] = []
        sol_mole = round_or_none(row.get("Solubility(mole_fraction)"), ROUND_SOLUBILITY)
        if sol_mole is not None:
            measurements.append(("mole_fraction", sol_mole))
        sol_g100 = round_or_none(row.get("Solubility(g/g100)"), ROUND_SOLUBILITY)
        if sol_g100 is not None:
            measurements.append(("g_per_100g", sol_g100))
        if not measurements:
            excluded.append({"row_index": row_index, "reason": "missing_solubility", "doi": doi})
            continue

        fraction_value = round(float(fraction_solvent1), ROUND_FRACTION)
        if abs(fraction_value - 1.0) <= EPS:
            required_smiles = ("SMILES_Solute", "SMILES_Solvent1")
            smiles_by_field, smiles_exclusion = _required_reference_smiles(row, required_smiles)
            if smiles_exclusion is not None:
                excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
                continue
            mode = "single"
            solvent_name = clean_text(row.get("Solvent1"))
            solvent_smiles = smiles_by_field["SMILES_Solvent1"]
            active_fraction_type = None
            active_fraction_solvent1 = None
        elif abs(fraction_value) <= EPS:
            required_smiles = ("SMILES_Solute", "SMILES_Solvent2")
            smiles_by_field, smiles_exclusion = _required_reference_smiles(row, required_smiles)
            if smiles_exclusion is not None:
                excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
                continue
            mode = "single"
            solvent_name = clean_text(row.get("Solvent2"))
            solvent_smiles = smiles_by_field["SMILES_Solvent2"]
            active_fraction_type = None
            active_fraction_solvent1 = None
        else:
            required_smiles = ("SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2")
            smiles_by_field, smiles_exclusion = _required_reference_smiles(row, required_smiles)
            if smiles_exclusion is not None:
                excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
                continue
            mode = "binary"
            solvent_name = None
            solvent_smiles = None
            active_fraction_type = fraction_type if fraction_type in {"mole", "mass", "volume"} else None
            active_fraction_solvent1 = fraction_value
            if active_fraction_type is None:
                excluded.append({"row_index": row_index, "reason": "unsupported_fraction_type", "doi": doi})
                continue

        for metric, solubility in measurements:
            if mode == "single":
                records.append(
                    Record(
                        dataset="mixturesoldb",
                        row_index=row_index,
                        doi=doi,
                        mode=mode,
                        metric=metric,
                        temperature=temperature,
                        solubility=solubility,
                        fraction_type=None,
                        fraction_solvent1=None,
                        compound_name=clean_text(row.get("Compound_Name")),
                        solute_smiles=smiles_by_field["SMILES_Solute"],
                        solvent1_name=solvent_name,
                        solvent2_name=None,
                        solvent1_smiles=solvent_smiles,
                        solvent2_smiles=None,
                    )
                )
            else:
                records.append(
                    Record(
                        dataset="mixturesoldb",
                        row_index=row_index,
                        doi=doi,
                        mode=mode,
                        metric=metric,
                        temperature=temperature,
                        solubility=solubility,
                        fraction_type=active_fraction_type,
                        fraction_solvent1=active_fraction_solvent1,
                        compound_name=clean_text(row.get("Compound_Name")),
                        solute_smiles=smiles_by_field["SMILES_Solute"],
                        solvent1_name=clean_text(row.get("Solvent1")),
                        solvent2_name=clean_text(row.get("Solvent2")),
                        solvent1_smiles=smiles_by_field["SMILES_Solvent1"],
                        solvent2_smiles=smiles_by_field["SMILES_Solvent2"],
                    )
                )
    return records, excluded


def load_bigsoldb_records(path: str | Path) -> tuple[list[Record], list[dict[str, Any]]]:
    dataframe = pd.read_csv(path, low_memory=False)
    records: list[Record] = []
    excluded: list[dict[str, Any]] = []
    for row_index, row in dataframe.iterrows():
        doi = normalize_doi(row.get("Source"))
        temperature = round_or_none(row.get("Temperature_K"), ROUND_TEMPERATURE)
        solubility = round_or_none(row.get("Solubility(mole_fraction)"), ROUND_SOLUBILITY)
        if not doi or temperature is None or solubility is None:
            excluded.append({"row_index": row_index, "reason": "missing_required_fields", "doi": doi})
            continue
        smiles_by_field, smiles_exclusion = _required_reference_smiles(row, ("SMILES_Solute", "SMILES_Solvent"))
        if smiles_exclusion is not None:
            excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
            continue
        records.append(
            Record(
                dataset="bigsoldb",
                row_index=row_index,
                doi=doi,
                mode="single",
                metric="mole_fraction",
                temperature=temperature,
                solubility=solubility,
                fraction_type=None,
                fraction_solvent1=None,
                compound_name=clean_text(row.get("Compound_Name")),
                solute_smiles=smiles_by_field["SMILES_Solute"],
                solvent1_name=clean_text(row.get("Solvent")),
                solvent2_name=None,
                solvent1_smiles=smiles_by_field["SMILES_Solvent"],
                solvent2_smiles=None,
            )
        )
    return records, excluded


def load_records(path: str | Path, fmt: str, dataset_name: str) -> tuple[list[Record], list[dict[str, Any]]]:
    normalized_format = fmt.lower()
    if normalized_format == "standardized":
        return load_standardized_records(path, dataset_name)
    if normalized_format == "mixturesoldb":
        return load_mixturesoldb_records(path)
    if normalized_format == "bigsoldb":
        return load_bigsoldb_records(path)
    raise ValueError(f"Unsupported comparison format: {fmt}")


def record_system_signature(record: Record) -> tuple[Any, ...] | None:
    solute_identity = entity_identity(record.solute_smiles, record.compound_name)
    if solute_identity is None:
        return None
    solvent1_identity = entity_identity(record.solvent1_smiles, record.solvent1_name)
    if solvent1_identity is None:
        return None
    if record.mode == "single":
        return (record.doi, "single", solute_identity, solvent1_identity)

    solvent2_identity = entity_identity(record.solvent2_smiles, record.solvent2_name)
    if solvent2_identity is None:
        return None
    first_two = tuple(sorted((solvent1_identity, solvent2_identity)))
    return (record.doi, "binary", solute_identity, first_two[0], first_two[1])


def record_metric_signature(record: Record) -> tuple[Any, ...] | None:
    signature = record_system_signature(record)
    if signature is None:
        return None
    return signature + (record.metric,)


def exact_key(record: Record) -> tuple[Any, ...] | None:
    signature = record_metric_signature(record)
    if signature is None:
        return None
    return signature + (
        record.temperature,
        record.solubility,
        record.fraction_type,
        tuple(record_composition(record) or []),
    )


def pair_by_key(
    predicted: list[Record],
    reference: list[Record],
    key_builder: Callable[[Record], tuple[Any, ...] | None],
) -> tuple[list[tuple[Record, Record]], list[Record], list[Record]]:
    left_grouped: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    right_grouped: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    left_unpaired: list[Record] = []
    right_unpaired: list[Record] = []

    for record in predicted:
        key = key_builder(record)
        if key is None:
            left_unpaired.append(record)
        else:
            left_grouped[key].append(record)

    for record in reference:
        key = key_builder(record)
        if key is None:
            right_unpaired.append(record)
        else:
            right_grouped[key].append(record)

    paired: list[tuple[Record, Record]] = []
    for key in sorted(set(left_grouped) | set(right_grouped), key=str):
        left_rows = sorted(left_grouped.get(key, []), key=lambda record: record.row_index)
        right_rows = sorted(right_grouped.get(key, []), key=lambda record: record.row_index)
        pair_count = min(len(left_rows), len(right_rows))
        paired.extend(zip(left_rows[:pair_count], right_rows[:pair_count]))
        left_unpaired.extend(left_rows[pair_count:])
        right_unpaired.extend(right_rows[pair_count:])

    return paired, left_unpaired, right_unpaired


def partial_match_score(predicted: Record, reference: Record) -> tuple[Any, ...]:
    temperature_delta = abs(predicted.temperature - reference.temperature)
    solubility_delta = abs(predicted.solubility - reference.solubility)
    composition_pred = predicted.fraction_solvent1 if predicted.fraction_solvent1 is not None else -1.0
    composition_ref = reference.fraction_solvent1 if reference.fraction_solvent1 is not None else -1.0
    composition_delta = abs(composition_pred - composition_ref)
    return (
        int(predicted.fraction_type == reference.fraction_type),
        -composition_delta,
        -temperature_delta,
        -solubility_delta,
    )


def pair_by_metric_signature(predicted: list[Record], reference: list[Record]) -> tuple[list[tuple[Record, Record]], list[Record], list[Record]]:
    left_grouped: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    right_grouped: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    left_unpaired: list[Record] = []
    right_unpaired: list[Record] = []

    for record in predicted:
        key = record_metric_signature(record)
        if key is None:
            left_unpaired.append(record)
        else:
            left_grouped[key].append(record)

    for record in reference:
        key = record_metric_signature(record)
        if key is None:
            right_unpaired.append(record)
        else:
            right_grouped[key].append(record)

    paired: list[tuple[Record, Record]] = []
    for key in sorted(set(left_grouped) | set(right_grouped), key=str):
        left_rows = left_grouped.get(key, [])
        right_rows = right_grouped.get(key, [])
        if not left_rows:
            right_unpaired.extend(right_rows)
            continue
        if not right_rows:
            left_unpaired.extend(left_rows)
            continue

        scored_pairs: list[tuple[tuple[Any, ...], int, int]] = []
        for left_index, left_record in enumerate(left_rows):
            for right_index, right_record in enumerate(right_rows):
                scored_pairs.append((partial_match_score(left_record, right_record), left_index, right_index))
        scored_pairs.sort(reverse=True)

        used_left: set[int] = set()
        used_right: set[int] = set()
        for _, left_index, right_index in scored_pairs:
            if left_index in used_left or right_index in used_right:
                continue
            used_left.add(left_index)
            used_right.add(right_index)
            paired.append((left_rows[left_index], right_rows[right_index]))

        left_unpaired.extend(record for index, record in enumerate(left_rows) if index not in used_left)
        right_unpaired.extend(record for index, record in enumerate(right_rows) if index not in used_right)

    return paired, left_unpaired, right_unpaired


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _value_difference(left: Any, reference: Any, left_label: str) -> dict[str, Any]:
    difference = {
        left_label: _json_value(left),
        "reference": _json_value(reference),
    }
    if isinstance(left, (int, float)) and isinstance(reference, (int, float)):
        difference["delta"] = round(float(left) - float(reference), ROUND_SOLUBILITY)
    return difference


def _measurement_difference_fields(
    predicted: Record,
    reference: Record,
    *,
    left_label: str,
) -> dict[str, dict[str, Any]]:
    comparable = {
        "mode": (predicted.mode, reference.mode),
        "metric": (predicted.metric, reference.metric),
        "temperature": (predicted.temperature, reference.temperature),
        "solubility": (predicted.solubility, reference.solubility),
        "fraction_type": (predicted.fraction_type, reference.fraction_type),
        "composition": (record_composition(predicted), record_composition(reference)),
    }
    differences: dict[str, dict[str, Any]] = {}
    for field, (predicted_value, reference_value) in comparable.items():
        if predicted_value != reference_value:
            differences[field] = _value_difference(predicted_value, reference_value, left_label)
    return differences


def build_difference_fields(predicted: Record, reference: Record) -> dict[str, dict[str, Any]]:
    return _measurement_difference_fields(predicted, reference, left_label="predicted")


def _entity_summary(name: str | None, smiles: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "smiles": smiles,
        "inchikey": smiles_to_inchikey(smiles) if smiles else None,
    }


def _entity_comparison_key(name: str | None, smiles: str | None) -> str | None:
    inchikey = smiles_to_inchikey(smiles) if smiles else None
    if inchikey:
        return f"inchikey:{inchikey}"
    normalized_name = normalize_name_key(name)
    if normalized_name:
        return f"name:{normalized_name}"
    return None


def _record_solvent_entries(record: Record) -> list[dict[str, Any]]:
    if record.mode == "single":
        entries = [
            {
                "slot": 1,
                "fraction": 1.0,
                **_entity_summary(record.solvent1_name, record.solvent1_smiles),
            }
        ]
    elif record.mode == "binary":
        fraction_value = (
            record.fraction_solvent1_raw
            if record.fraction_solvent1_raw is not None
            else record.fraction_solvent1
        )
        if fraction_value is None:
            fraction1 = None
            fraction2 = None
        else:
            fraction1 = round(float(fraction_value), ROUND_FRACTION)
            fraction2 = round(1.0 - float(fraction_value), ROUND_FRACTION)
        entries = [
            {
                "slot": 1,
                "fraction": fraction1,
                **_entity_summary(record.solvent1_name, record.solvent1_smiles),
            },
            {
                "slot": 2,
                "fraction": fraction2,
                **_entity_summary(record.solvent2_name, record.solvent2_smiles),
            },
        ]
    else:
        entries = []

    return sorted(
        entries,
        key=lambda entry: (
            entry.get("inchikey") or normalize_name_key(entry.get("name")) or "",
            entry.get("fraction") if entry.get("fraction") is not None else -1.0,
            entry.get("slot") or 0,
        ),
    )


def _record_solvent_comparison_key(record: Record) -> tuple[Any, ...]:
    entries = _record_solvent_entries(record)
    return tuple(
        (
            entry.get("inchikey") or normalize_name_key(entry.get("name")),
            entry.get("fraction"),
        )
        for entry in entries
    )


def _identity_difference_fields(
    predicted: Record,
    reference: Record,
    *,
    left_label: str,
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    predicted_solute_key = _entity_comparison_key(predicted.compound_name, predicted.solute_smiles)
    reference_solute_key = _entity_comparison_key(reference.compound_name, reference.solute_smiles)
    if predicted_solute_key != reference_solute_key:
        differences["solute"] = {
            left_label: _entity_summary(predicted.compound_name, predicted.solute_smiles),
            "reference": _entity_summary(reference.compound_name, reference.solute_smiles),
        }

    if _record_solvent_comparison_key(predicted) != _record_solvent_comparison_key(reference):
        differences["solvents"] = {
            left_label: _record_solvent_entries(predicted),
            "reference": _record_solvent_entries(reference),
        }
    return differences


def build_record_difference_fields(
    predicted: Record,
    reference: Record,
    *,
    left_label: str = "input",
    include_identity: bool = True,
) -> dict[str, dict[str, Any]]:
    differences = _measurement_difference_fields(predicted, reference, left_label=left_label)
    if include_identity:
        differences.update(_identity_difference_fields(predicted, reference, left_label=left_label))
    return differences


def _record_brief(record: Record) -> dict[str, Any]:
    return {
        "row_index": record.row_index,
        "dataset": record.dataset,
        "mode": record.mode,
        "metric": record.metric,
        "temperature": record.temperature,
        "solubility": record.solubility,
        "fraction_type": record.fraction_type,
        "composition": record_composition(record),
        "solute": _entity_summary(record.compound_name, record.solute_smiles),
        "solvents": _record_solvent_entries(record),
    }


def _paired_difference_report(
    predicted: Record,
    reference: Record,
    *,
    match_basis: str,
    include_identity: bool = True,
) -> dict[str, Any] | None:
    differences = build_record_difference_fields(
        predicted,
        reference,
        left_label="input",
        include_identity=include_identity,
    )
    if not differences:
        return None
    return {
        "input_row_index": predicted.row_index,
        "reference_row_index": reference.row_index,
        "match_basis": match_basis,
        "differences": differences,
    }


def _record_occurrence_key(record: Record) -> tuple[Any, ...]:
    return (record.dataset, record.row_index, _record_identity(record))


def _pair_occurrence_key(predicted: Record, reference: Record) -> tuple[Any, ...]:
    return (_record_occurrence_key(predicted), _record_occurrence_key(reference))


def _records_by_doi(records: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.doi:
            grouped[record.doi].append(record)
    return grouped


def _exclusions_by_doi(excluded: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in excluded or []:
        doi = str(item.get("doi") or "")
        if doi:
            grouped[doi].append(item)
    return grouped


def _records_with_dois(records: list[Record], dois: set[str]) -> list[Record]:
    return [record for record in records if record.doi in dois]


def _exclusions_with_dois(excluded: list[dict[str, Any]], dois: set[str]) -> list[dict[str, Any]]:
    return [item for item in excluded if str(item.get("doi") or "") in dois]


def _build_doi_match_pass_reports(
    predicted_records: list[Record],
    reference_records: list[Record],
    pairs: list[tuple[Record, Record]],
    predicted_unmatched: list[Record],
    reference_unmatched: list[Record],
) -> dict[str, dict[str, Any]]:
    predicted_by_doi = _records_by_doi(predicted_records)
    reference_by_doi = _records_by_doi(reference_records)
    pair_counts = Counter(predicted_record.doi for predicted_record, _ in pairs if predicted_record.doi)
    predicted_unmatched_by_doi = _records_by_doi(predicted_unmatched)
    reference_unmatched_by_doi = _records_by_doi(reference_unmatched)
    dois = sorted(set(predicted_by_doi) | set(reference_by_doi))

    reports: dict[str, dict[str, Any]] = {}
    for doi in dois:
        input_only = sorted(predicted_unmatched_by_doi.get(doi, []), key=lambda record: record.row_index)
        reference_only = sorted(
            reference_unmatched_by_doi.get(doi, []),
            key=lambda record: record.row_index,
        )
        reports[doi] = {
            "matched_count": pair_counts.get(doi, 0),
            "input_only_count": len(input_only),
            "reference_only_count": len(reference_only),
            "input_only": [_record_brief(record) for record in input_only],
            "reference_only": [_record_brief(record) for record in reference_only],
        }
    return reports


def _build_doi_reports(
    predicted_records: list[Record],
    reference_records: list[Record],
    exact_pairs: list[tuple[Record, Record]],
    partial_pairs: list[tuple[Record, Record]],
    predicted_unmatched: list[Record],
    reference_unmatched: list[Record],
    *,
    predicted_excluded: list[dict[str, Any]] | None = None,
    reference_excluded: list[dict[str, Any]] | None = None,
    additional_row_differences: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    predicted_by_doi = _records_by_doi(predicted_records)
    reference_by_doi = _records_by_doi(reference_records)
    predicted_unmatched_by_doi = _records_by_doi(predicted_unmatched)
    reference_unmatched_by_doi = _records_by_doi(reference_unmatched)
    predicted_excluded_by_doi = _exclusions_by_doi(predicted_excluded)
    reference_excluded_by_doi = _exclusions_by_doi(reference_excluded)

    exact_counts = Counter(predicted_record.doi for predicted_record, _ in exact_pairs if predicted_record.doi)
    partial_counts = Counter(
        predicted_record.doi for predicted_record, _ in partial_pairs if predicted_record.doi
    )
    row_differences_by_doi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for predicted_record, reference_record in partial_pairs:
        report = _paired_difference_report(
            predicted_record,
            reference_record,
            match_basis="same_doi_system_and_metric",
            include_identity=False,
        )
        if report is not None and predicted_record.doi:
            row_differences_by_doi[predicted_record.doi].append(report)

    for report in additional_row_differences or []:
        doi = str(report.get("doi") or "")
        if not doi:
            continue
        report_without_doi = {key: value for key, value in report.items() if key != "doi"}
        row_differences_by_doi[doi].append(report_without_doi)

    dois = sorted(
        set(predicted_by_doi)
        | set(reference_by_doi)
        | set(predicted_excluded_by_doi)
        | set(reference_excluded_by_doi)
    )
    reports: dict[str, dict[str, Any]] = {}
    for doi in dois:
        input_only = sorted(predicted_unmatched_by_doi.get(doi, []), key=lambda record: record.row_index)
        reference_only = sorted(
            reference_unmatched_by_doi.get(doi, []),
            key=lambda record: record.row_index,
        )
        row_differences = sorted(
            row_differences_by_doi.get(doi, []),
            key=lambda item: (
                item.get("input_row_index", -1),
                item.get("reference_row_index", -1),
                item.get("match_basis", ""),
            ),
        )
        predicted_excluded_for_doi = predicted_excluded_by_doi.get(doi, [])
        reference_excluded_for_doi = reference_excluded_by_doi.get(doi, [])
        reports[doi] = {
            "input_count": len(predicted_by_doi.get(doi, [])),
            "reference_count": len(reference_by_doi.get(doi, [])),
            "exact_matches": exact_counts.get(doi, 0),
            "partial_matches": partial_counts.get(doi, 0),
            "row_wise_differences": row_differences,
            "input_only_count": len(input_only),
            "reference_only_count": len(reference_only),
            "input_only": [_record_brief(record) for record in input_only],
            "reference_only": [_record_brief(record) for record in reference_only],
            "excluded_input_count": len(predicted_excluded_for_doi),
            "excluded_reference_count": len(reference_excluded_for_doi),
            "excluded_input_reasons": _count_exclusion_reasons(predicted_excluded_for_doi),
            "excluded_reference_reasons": _count_exclusion_reasons(reference_excluded_for_doi),
        }
    return reports


def compare_record_sets(
    predicted: list[Record],
    reference: list[Record],
    *,
    predicted_excluded: list[dict[str, Any]] | None = None,
    reference_excluded: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    exact_pairs, predicted_remaining, reference_remaining = pair_by_key(predicted, reference, exact_key)
    partial_pairs, predicted_unmatched, reference_unmatched = pair_by_metric_signature(
        predicted_remaining,
        reference_remaining,
    )

    partial_mismatches = []
    for predicted_record, reference_record in partial_pairs:
        differences = build_difference_fields(predicted_record, reference_record)
        if differences:
            partial_mismatches.append(
                {
                    "predicted": serialize_record(predicted_record),
                    "reference": serialize_record(reference_record),
                    "differences": differences,
                }
            )

    return {
        "summary": {
            "predicted_records": len(predicted),
            "reference_records": len(reference),
            "exact_matches": len(exact_pairs),
            "partial_matches": len(partial_pairs),
            "missing_in_prediction": len(reference_unmatched),
            "extra_in_prediction": len(predicted_unmatched),
        },
        "partial_mismatches": partial_mismatches,
        "missing_in_prediction": [serialize_record(record) for record in reference_unmatched],
        "extra_in_prediction": [serialize_record(record) for record in predicted_unmatched],
        "doi_reports": _build_doi_reports(
            predicted,
            reference,
            exact_pairs,
            partial_pairs,
            predicted_unmatched,
            reference_unmatched,
            predicted_excluded=predicted_excluded,
            reference_excluded=reference_excluded,
        ),
    }


def _count_exclusion_reasons(excluded: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("reason", "unknown")) for item in excluded)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


def _normalize_unit_text(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = text.lower()
    text = text.replace("\\", "")
    text = text.replace("$", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _classify_solubility(value: Any, unit: Any) -> tuple[str | None, float | None]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None, None
    numeric = float(numeric)
    unit_text = _normalize_unit_text(unit)
    if unit_text is None:
        return None, None

    if unit_text in {
        "mole fraction",
        "mol/mol",
        "mol mol^-1",
        "mol mol-1",
        "mol mol^(-1)",
    }:
        return "mole_fraction", round(numeric, ROUND_SOLUBILITY)

    if unit_text.startswith("g/100 g") or unit_text.startswith("g/100g"):
        return "g_per_100g", round(numeric, ROUND_SOLUBILITY)

    if unit_text == "g/g":
        return "g_per_100g", round(numeric * 100.0, ROUND_SOLUBILITY)

    return None, None


def _classify_fraction(value: Any, unit: Any) -> tuple[str | None, float | None]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None, None
    numeric = float(numeric)
    unit_text = _normalize_unit_text(unit)
    if unit_text is None:
        return None, None

    direct_fraction_units = {
        "mole fraction": "mole",
        "mass fraction": "mass",
        "weight fraction": "mass",
        "volume fraction": "volume",
        "x": "mole",
        "w": "mass",
        "phi": "volume",
        "ϕ": "volume",
    }
    percent_fraction_units = {
        "%": "unknown_percent",
        "mol%": "mole",
        "mole %": "mole",
        "mole%": "mole",
        "wt%": "mass",
        "weight %": "mass",
        "weight%": "mass",
        "mass %": "mass",
        "mass%": "mass",
        "vol%": "volume",
        "volume %": "volume",
        "volume%": "volume",
    }

    if unit_text in direct_fraction_units:
        fraction_type = direct_fraction_units[unit_text]
        fraction_value = numeric
    elif unit_text in percent_fraction_units:
        fraction_type = percent_fraction_units[unit_text]
        fraction_value = numeric / 100.0
    else:
        return None, None

    if fraction_type == "unknown_percent":
        return None, None
    if not (0.0 - EPS <= fraction_value <= 1.0 + EPS):
        return None, None
    clipped = min(max(fraction_value, 0.0), 1.0)
    return fraction_type, round(clipped, ROUND_FRACTION)


def _normalize_fraction_type(value: Any) -> str | None:
    text = _normalize_unit_text(value)
    if text in {"mole", "mass", "volume"}:
        return text
    return None


def _is_effectively_zero(value: float) -> bool:
    return abs(value) <= EPS


def _is_effectively_one(value: float) -> bool:
    return abs(value - 1.0) <= EPS


def _standardized_solvent_slots(row: pd.Series, max_slots: int = 4) -> list[tuple[str | None, str | None]]:
    solvents: list[tuple[str | None, str | None]] = []
    for idx in range(1, max_slots + 1):
        name = clean_text(row.get(f"Solvent {idx}"))
        smiles = canonicalize_smiles(row.get(f"SMILES_Solvent_{idx}"))
        if name is None and smiles is None:
            continue
        solvents.append((name, smiles))
    return solvents


def _record_identity(record: Record) -> tuple[Any, ...]:
    return (
        record.mode,
        record.doi,
        record.temperature,
        record.metric,
        record.solubility,
        record.fraction_type,
        record.fraction_solvent1,
        record.compound_name,
        record.solute_smiles,
        record.solvent1_name,
        record.solvent2_name,
        (record.solvent1_smiles, record.solvent2_smiles),
    )


def _dedupe_records(records: list[Record]) -> list[Record]:
    deduped: list[Record] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        identity = _record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(record)
    return deduped


def _key_without_smiles(record: Record) -> tuple[Any, ...] | None:
    if record.mode == "single":
        return ("single", record.metric, record.solubility, record.temperature, record.doi)

    fraction_value = record.fraction_solvent1_raw if record.fraction_solvent1_raw is not None else record.fraction_solvent1
    if record.mode == "binary" and record.fraction_type is not None and fraction_value is not None:
        f1 = round(float(fraction_value), ROUND_FRACTION)
        f2 = round(1.0 - float(fraction_value), ROUND_FRACTION)
        composition = tuple(sorted((f1, f2)))
        return (
            "binary",
            record.fraction_type,
            composition[0],
            composition[1],
            record.metric,
            record.solubility,
            record.temperature,
            record.doi,
        )
    return None


def _key_with_smiles(record: Record) -> tuple[Any, ...] | None:
    if record.solute_smiles is None:
        return None
    solute_inchikey = smiles_to_inchikey(record.solute_smiles)

    if record.mode == "single":
        if record.solvent1_smiles is None:
            return None
        solvent_inchikey = smiles_to_inchikey(record.solvent1_smiles)
        return (
            "single",
            solute_inchikey,
            solvent_inchikey,
            record.metric,
            record.solubility,
            record.temperature,
            record.doi,
        )

    if record.mode == "binary" and record.fraction_solvent1 is not None and record.fraction_type is not None:
        if record.solvent1_smiles is None or record.solvent2_smiles is None:
            return None
        solvent1_inchikey = smiles_to_inchikey(record.solvent1_smiles)
        solvent2_inchikey = smiles_to_inchikey(record.solvent2_smiles)
        fraction_value = record.fraction_solvent1_raw if record.fraction_solvent1_raw is not None else record.fraction_solvent1
        f1 = round(float(fraction_value), ROUND_FRACTION)
        f2 = round(1.0 - float(fraction_value), ROUND_FRACTION)
        pairs = tuple(
            sorted(
                ((solvent1_inchikey, f1), (solvent2_inchikey, f2)),
                key=lambda item: (item[0] or "", item[1]),
            )
        )
        return (
            "binary",
            solute_inchikey,
            pairs[0][0],
            pairs[0][1],
            pairs[1][0],
            pairs[1][1],
            record.fraction_type,
            record.metric,
            record.solubility,
            record.temperature,
            record.doi,
        )
    return None


def _metric_selection_signature(record: Record) -> tuple[Any, ...] | None:
    solute_key = _entity_comparison_key(record.compound_name, record.solute_smiles)
    if solute_key is None:
        return None
    solvent_key = _record_solvent_comparison_key(record)
    if not solvent_key or any(item[0] is None for item in solvent_key):
        return None
    return (
        record.doi,
        record.mode,
        record.temperature,
        record.fraction_type,
        solvent_key,
        solute_key,
    )


def _select_reference_records_for_predicted_metrics(
    reference_records: list[Record],
    predicted_records: list[Record],
) -> tuple[list[Record], dict[str, int]]:
    predicted_keys_with_smiles = {
        key for key in (_key_with_smiles(record) for record in predicted_records) if key is not None
    }
    predicted_keys_without_smiles = {
        key for key in (_key_without_smiles(record) for record in predicted_records) if key is not None
    }
    predicted_metrics_by_signature: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for record in predicted_records:
        signature = _metric_selection_signature(record)
        if signature is not None:
            predicted_metrics_by_signature[signature].add(record.metric)

    records_by_source_row: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for record in reference_records:
        records_by_source_row[(record.dataset, record.row_index)].append(record)

    selected: list[Record] = []
    stats: Counter[str] = Counter()
    metric_priority = {"mole_fraction": 0, "g_per_100g": 1}

    for source_records in records_by_source_row.values():
        unique_metrics = {record.metric for record in source_records}
        if len(unique_metrics) <= 1:
            selected.extend(source_records)
            continue

        stats["multi_metric_source_rows"] += 1
        stats["extra_metric_records_removed"] += len(source_records) - 1

        def selection_score(record: Record) -> tuple[int, int, int]:
            signature = _metric_selection_signature(record)
            predicted_metrics = (
                predicted_metrics_by_signature.get(signature, set())
                if signature is not None
                else set()
            )
            return (
                int(_key_with_smiles(record) in predicted_keys_with_smiles),
                int(_key_without_smiles(record) in predicted_keys_without_smiles),
                int(record.metric in predicted_metrics),
            )

        chosen = max(
            source_records,
            key=lambda record: (
                selection_score(record),
                -metric_priority.get(record.metric, 99),
                -record.row_index,
            ),
        )
        selected.append(chosen)
        stats[f"selected_{chosen.metric}"] += 1

    return selected, dict(stats)


def _load_standardized_records_for_bigsoldb(path: str | Path) -> tuple[list[Record], list[dict[str, Any]], int]:
    dataframe = pd.read_csv(path, low_memory=False)
    records: list[Record] = []
    excluded: list[dict[str, Any]] = []

    for row_index, row in dataframe.iterrows():
        doi = normalize_doi(row.get("doi"))
        temperature = round_or_none(row.get("Temperature"), ROUND_TEMPERATURE)
        metric, solubility = _classify_solubility(row.get("Solubility"), row.get("Solubility Unit"))
        solvents = _standardized_solvent_slots(row)
        solvent_count = len(solvents)

        if solvent_count == 0:
            excluded.append({"row_index": row_index, "reason": "no_solvent", "doi": doi})
            continue
        if solvent_count >= 2:
            excluded.append({"row_index": row_index, "reason": "not_single_solvent", "doi": doi})
            continue
        if temperature is None:
            excluded.append({"row_index": row_index, "reason": "missing_temperature", "doi": doi})
            continue
        if metric != "mole_fraction" or solubility is None:
            excluded.append({"row_index": row_index, "reason": "unsupported_solubility_unit", "doi": doi})
            continue

        records.append(
            Record(
                dataset="predicted",
                row_index=row_index,
                doi=doi,
                mode="single",
                metric=metric,
                temperature=temperature,
                solubility=solubility,
                fraction_type=None,
                fraction_solvent1=None,
                compound_name=clean_text(row.get("Compound Name")),
                solute_smiles=canonicalize_smiles(row.get("SMILES_Compound")),
                solvent1_name=solvents[0][0],
                solvent2_name=None,
                solvent1_smiles=solvents[0][1],
                solvent2_smiles=None,
            )
        )

    return records, excluded, len(dataframe)


def _load_standardized_records_for_mixturesoldb(path: str | Path) -> tuple[list[Record], list[dict[str, Any]], int]:
    dataframe = pd.read_csv(path, low_memory=False)
    records: list[Record] = []
    excluded: list[dict[str, Any]] = []

    for row_index, row in dataframe.iterrows():
        doi = normalize_doi(row.get("doi"))
        temperature = round_or_none(row.get("Temperature"), ROUND_TEMPERATURE)
        metric, solubility = _classify_solubility(row.get("Solubility"), row.get("Solubility Unit"))
        solvents = _standardized_solvent_slots(row)
        solvent_count = len(solvents)

        if solvent_count == 0:
            excluded.append({"row_index": row_index, "reason": "no_solvent", "doi": doi})
            continue
        if solvent_count >= 3:
            excluded.append({"row_index": row_index, "reason": "more_than_two_solvents", "doi": doi})
            continue
        if temperature is None:
            excluded.append({"row_index": row_index, "reason": "missing_temperature", "doi": doi})
            continue
        if metric is None or solubility is None:
            excluded.append({"row_index": row_index, "reason": "unsupported_solubility_unit", "doi": doi})
            continue

        if solvent_count == 1:
            records.append(
                Record(
                    dataset="predicted",
                    row_index=row_index,
                    doi=doi,
                    mode="single",
                    metric=metric,
                    temperature=temperature,
                    solubility=solubility,
                    fraction_type=None,
                    fraction_solvent1=None,
                    compound_name=clean_text(row.get("Compound Name")),
                    solute_smiles=canonicalize_smiles(row.get("SMILES_Compound")),
                    solvent1_name=solvents[0][0],
                    solvent2_name=None,
                    solvent1_smiles=solvents[0][1],
                    solvent2_smiles=None,
                )
            )
            continue

        fraction_type, fraction_solvent1 = _classify_fraction(
            row.get("Concentration Solvent 1"),
            row.get("Concentration Unit"),
        )
        if fraction_type is None or fraction_solvent1 is None:
            excluded.append({"row_index": row_index, "reason": "unsupported_fraction_unit", "doi": doi})
            continue

        records.append(
            Record(
                dataset="predicted",
                row_index=row_index,
                doi=doi,
                mode="binary",
                metric=metric,
                temperature=temperature,
                solubility=solubility,
                fraction_type=fraction_type,
                fraction_solvent1=fraction_solvent1,
                compound_name=clean_text(row.get("Compound Name")),
                solute_smiles=canonicalize_smiles(row.get("SMILES_Compound")),
                solvent1_name=solvents[0][0],
                solvent2_name=solvents[1][0],
                solvent1_smiles=solvents[0][1],
                solvent2_smiles=solvents[1][1],
                fraction_solvent1_raw=fraction_solvent1,
            )
        )

    return records, excluded, len(dataframe)


def _build_report(
    *,
    comparison_mode: str,
    predicted_records: list[Record],
    predicted_excluded: list[dict[str, Any]],
    predicted_total_rows: int,
    reference_records: list[Record],
    reference_excluded: list[dict[str, Any]],
    reference_total_rows: int,
    reference_label: str,
) -> dict[str, Any]:
    deduped_reference_records = _dedupe_records(reference_records)
    selected_reference_records, reference_metric_selection = _select_reference_records_for_predicted_metrics(
        deduped_reference_records,
        predicted_records,
    )
    predicted_dois = {record.doi for record in predicted_records if record.doi}
    reference_dois = {record.doi for record in selected_reference_records if record.doi}
    shared_dois = predicted_dois & reference_dois

    comparable_records_all_dois = predicted_records
    reference_records_all_dois = selected_reference_records
    comparable_records = _records_with_dois(comparable_records_all_dois, shared_dois)
    reference_records_with_shared_doi = _records_with_dois(reference_records_all_dois, shared_dois)
    predicted_excluded_with_shared_doi = _exclusions_with_dois(predicted_excluded, shared_dois)
    reference_excluded_with_shared_doi = _exclusions_with_dois(reference_excluded, shared_dois)
    excluded_missing_doi = len(comparable_records_all_dois) - len(comparable_records)
    reference_records_without_shared_doi = len(reference_records_all_dois) - len(reference_records_with_shared_doi)
    reference_source_rows_all_dois = {record.row_index for record in reference_records_all_dois}
    reference_source_rows_with_shared_doi = {
        record.row_index for record in reference_records_with_shared_doi
    }

    without_smiles_pairs, predicted_non_matches_without_smiles, reference_non_matches_without_smiles = pair_by_key(
        comparable_records,
        reference_records_with_shared_doi,
        _key_without_smiles,
    )
    predicted_matches_without_smiles = [predicted_record for predicted_record, _ in without_smiles_pairs]
    matching_without_smiles = {
        _key_without_smiles(predicted_record)
        for predicted_record, _ in without_smiles_pairs
        if _key_without_smiles(predicted_record) is not None
    }
    reference_matched_row_indices_without_smiles = {reference_record.row_index for _, reference_record in without_smiles_pairs}

    with_smiles_pairs, predicted_non_matches_with_smiles, reference_non_matches_with_smiles = pair_by_key(
        comparable_records,
        reference_records_with_shared_doi,
        _key_with_smiles,
    )
    predicted_matches_with_smiles = [predicted_record for predicted_record, _ in with_smiles_pairs]
    matching_with_smiles = {
        _key_with_smiles(predicted_record)
        for predicted_record, _ in with_smiles_pairs
        if _key_with_smiles(predicted_record) is not None
    }
    reference_matched_row_indices_with_smiles = {reference_record.row_index for _, reference_record in with_smiles_pairs}

    with_smiles_predicted_keys = {
        _record_occurrence_key(predicted_record) for predicted_record, _ in with_smiles_pairs
    }
    with_smiles_pair_keys = {
        _pair_occurrence_key(predicted_record, reference_record)
        for predicted_record, reference_record in with_smiles_pairs
    }
    fails_only_with_smiles = [
        record
        for record in predicted_matches_without_smiles
        if _record_occurrence_key(record) not in with_smiles_predicted_keys
    ]
    smiles_mismatch_differences: list[dict[str, Any]] = []
    for predicted_record, reference_record in without_smiles_pairs:
        if _pair_occurrence_key(predicted_record, reference_record) in with_smiles_pair_keys:
            continue
        report = _paired_difference_report(
            predicted_record,
            reference_record,
            match_basis="matched_without_smiles_but_not_with_smiles",
            include_identity=True,
        )
        if report is not None:
            smiles_mismatch_differences.append({"doi": predicted_record.doi, **report})

    row_exact_pairs, predicted_row_remaining, reference_row_remaining = pair_by_key(
        comparable_records,
        reference_records_with_shared_doi,
        exact_key,
    )
    row_partial_pairs, predicted_row_unmatched, reference_row_unmatched = pair_by_metric_signature(
        predicted_row_remaining,
        reference_row_remaining,
    )
    doi_reports = _build_doi_reports(
        comparable_records,
        reference_records_with_shared_doi,
        row_exact_pairs,
        row_partial_pairs,
        predicted_row_unmatched,
        reference_row_unmatched,
        predicted_excluded=predicted_excluded_with_shared_doi,
        reference_excluded=reference_excluded_with_shared_doi,
        additional_row_differences=smiles_mismatch_differences,
    )
    without_smiles_reports = _build_doi_match_pass_reports(
        comparable_records,
        reference_records_with_shared_doi,
        without_smiles_pairs,
        predicted_non_matches_without_smiles,
        reference_non_matches_without_smiles,
    )
    with_smiles_reports = _build_doi_match_pass_reports(
        comparable_records,
        reference_records_with_shared_doi,
        with_smiles_pairs,
        predicted_non_matches_with_smiles,
        reference_non_matches_with_smiles,
    )
    empty_pass_report = {
        "matched_count": 0,
        "input_only_count": 0,
        "reference_only_count": 0,
        "input_only": [],
        "reference_only": [],
    }
    for doi, doi_report in doi_reports.items():
        doi_report["without_smiles"] = without_smiles_reports.get(doi, empty_pass_report)
        doi_report["with_smiles"] = with_smiles_reports.get(doi, empty_pass_report)

    unmatched_dois = sorted({record.doi for record in predicted_non_matches_with_smiles if record.doi})
    reference_shared_rows = len(reference_records_with_shared_doi)

    summary = {
        "comparison_scope": "shared_doi",
        "predicted_total_rows": predicted_total_rows,
        "reference_total_rows": reference_total_rows,
        "predicted_comparable_rows_all_dois": len(comparable_records_all_dois),
        "reference_comparable_rows_all_dois": len(reference_records_all_dois),
        "reference_comparable_source_rows_all_dois": len(reference_source_rows_all_dois),
        "predicted_comparable_rows": len(comparable_records),
        "reference_comparable_rows": reference_shared_rows,
        "predicted_comparable_rows_with_shared_doi": len(comparable_records),
        "reference_comparable_rows_with_shared_doi": reference_shared_rows,
        "reference_comparable_source_rows_with_shared_doi": len(reference_source_rows_with_shared_doi),
        "predicted_comparable_rows_without_shared_doi": excluded_missing_doi,
        "reference_comparable_rows_without_shared_doi": reference_records_without_shared_doi,
        "excluded_missing_doi": excluded_missing_doi,
        "reference_missing_smiles_skipped": _count_exclusion_reasons(
            reference_excluded_with_shared_doi
        ).get("missing_reference_smiles", 0),
        "reference_missing_smiles_skipped_all_dois": _count_exclusion_reasons(reference_excluded).get(
            "missing_reference_smiles",
            0,
        ),
        "reference_keys_without_smiles": len(
            {
                key
                for key in (_key_without_smiles(record) for record in reference_records_with_shared_doi)
                if key is not None
            }
        ),
        "reference_keys_with_smiles": len(
            {
                key
                for key in (_key_with_smiles(record) for record in reference_records_with_shared_doi)
                if key is not None
            }
        ),
        "predicted_unique_dois": len(predicted_dois),
        "reference_unique_dois": len(reference_dois),
        "shared_doi_count": len(shared_dois),
        "predicted_exclusion_reasons": _count_exclusion_reasons(predicted_excluded_with_shared_doi),
        "reference_exclusion_reasons": _count_exclusion_reasons(reference_excluded_with_shared_doi),
        "predicted_exclusion_reasons_all_dois": _count_exclusion_reasons(predicted_excluded),
        "reference_exclusion_reasons_all_dois": _count_exclusion_reasons(reference_excluded),
        "reference_metric_selection": reference_metric_selection,
        "without_smiles": {
            "matching_unique_keys": len(matching_without_smiles),
            "matched_predicted_rows": len(predicted_matches_without_smiles),
            "coverage_predicted_shared_doi_percent": round(
                _percentage(len(predicted_matches_without_smiles), len(comparable_records)), 2
            ),
            "non_matching_predicted_rows": len(predicted_non_matches_without_smiles),
            "reference_records_covered": len(without_smiles_pairs),
            "reference_rows_covered": len(reference_matched_row_indices_without_smiles),
            "reference_source_rows_covered": len(reference_matched_row_indices_without_smiles),
            "reference_coverage_percent": round(
                _percentage(len(without_smiles_pairs), reference_shared_rows), 2
            ),
            "reference_source_row_coverage_percent": round(
                _percentage(
                    len(reference_matched_row_indices_without_smiles),
                    len(reference_source_rows_with_shared_doi),
                ),
                2,
            ),
        },
        "with_smiles": {
            "matching_unique_keys": len(matching_with_smiles),
            "matched_predicted_rows": len(predicted_matches_with_smiles),
            "coverage_predicted_shared_doi_percent": round(
                _percentage(len(predicted_matches_with_smiles), len(comparable_records)), 2
            ),
            "non_matching_predicted_rows": len(predicted_non_matches_with_smiles),
            "reference_records_covered": len(with_smiles_pairs),
            "reference_rows_covered": len(reference_matched_row_indices_with_smiles),
            "reference_source_rows_covered": len(reference_matched_row_indices_with_smiles),
            "reference_coverage_percent": round(
                _percentage(len(with_smiles_pairs), reference_shared_rows), 2
            ),
            "reference_source_row_coverage_percent": round(
                _percentage(
                    len(reference_matched_row_indices_with_smiles),
                    len(reference_source_rows_with_shared_doi),
                ),
                2,
            ),
        },
        "smiles_mismatch_analysis": {
            "rows_matching_without_smiles": len(predicted_matches_without_smiles),
            "rows_fail_once_smiles_required": len(fails_only_with_smiles),
            "unique_dois_with_non_matching_rows": len(unmatched_dois),
        },
    }

    return {
        "comparison_mode": comparison_mode,
        "reference_label": reference_label,
        "summary": summary,
        "excluded_predicted": predicted_excluded,
        "excluded_reference": reference_excluded,
        "shared_doi_predicted_records": len(comparable_records),
        "shared_doi_reference_records": len(reference_records_with_shared_doi),
        "non_matching_predicted_without_smiles": [
            serialize_record(record) for record in predicted_non_matches_without_smiles
        ],
        "non_matching_reference_without_smiles": [
            serialize_record(record) for record in reference_non_matches_without_smiles
        ],
        "non_matching_predicted_with_smiles": [
            serialize_record(record) for record in predicted_non_matches_with_smiles
        ],
        "non_matching_reference_with_smiles": [
            serialize_record(record) for record in reference_non_matches_with_smiles
        ],
        "unmatched_dois_with_smiles": unmatched_dois,
        "doi_reports": doi_reports,
    }


def _compare_csv_files_bigsoldb(predicted_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    predicted_records, predicted_excluded, predicted_total_rows = _load_standardized_records_for_bigsoldb(predicted_path)
    reference_records, reference_excluded = load_bigsoldb_records(reference_path)
    reference_total_rows = len(pd.read_csv(reference_path, low_memory=False))
    return _build_report(
        comparison_mode="bigsoldb",
        predicted_records=predicted_records,
        predicted_excluded=predicted_excluded,
        predicted_total_rows=predicted_total_rows,
        reference_records=reference_records,
        reference_excluded=reference_excluded,
        reference_total_rows=reference_total_rows,
        reference_label="BigSolDB",
    )


def _load_mixturesoldb_records(path: str | Path) -> tuple[list[Record], list[dict[str, Any]]]:
    dataframe = pd.read_csv(path, low_memory=False)
    records: list[Record] = []
    excluded: list[dict[str, Any]] = []

    for row_index, row in dataframe.iterrows():
        doi = normalize_doi(row.get("Source"))
        temperature = round_or_none(row.get("Temperature_K"), ROUND_TEMPERATURE)
        fraction_solvent1 = pd.to_numeric(row.get("Fraction_Solvent1"), errors="coerce")
        fraction_type = _normalize_fraction_type(row.get("Fraction_Type"))
        if temperature is None or pd.isna(fraction_solvent1):
            excluded.append({"row_index": row_index, "reason": "missing_required_fields", "doi": doi})
            continue
        fraction_solvent1 = float(fraction_solvent1)

        measurements: list[tuple[str, float]] = []
        sol_mole = round_or_none(row.get("Solubility(mole_fraction)"), ROUND_SOLUBILITY)
        if sol_mole is not None:
            measurements.append(("mole_fraction", sol_mole))
        sol_g100 = round_or_none(row.get("Solubility(g/g100)"), ROUND_SOLUBILITY)
        if sol_g100 is not None:
            measurements.append(("g_per_100g", sol_g100))
        if not measurements:
            excluded.append({"row_index": row_index, "reason": "missing_solubility", "doi": doi})
            continue

        if _is_effectively_one(fraction_solvent1):
            required_smiles = ("SMILES_Solute", "SMILES_Solvent1")
            smiles_by_field, smiles_exclusion = _required_reference_smiles(row, required_smiles)
            if smiles_exclusion is not None:
                excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
                continue
            mode = "single"
            active_solvent_smiles = smiles_by_field["SMILES_Solvent1"]
            active_solvent_name = clean_text(row.get("Solvent1"))
        elif _is_effectively_zero(fraction_solvent1):
            required_smiles = ("SMILES_Solute", "SMILES_Solvent2")
            smiles_by_field, smiles_exclusion = _required_reference_smiles(row, required_smiles)
            if smiles_exclusion is not None:
                excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
                continue
            mode = "single"
            active_solvent_smiles = smiles_by_field["SMILES_Solvent2"]
            active_solvent_name = clean_text(row.get("Solvent2"))
        else:
            required_smiles = ("SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2")
            smiles_by_field, smiles_exclusion = _required_reference_smiles(row, required_smiles)
            if smiles_exclusion is not None:
                excluded.append({"row_index": row_index, "doi": doi, **smiles_exclusion})
                continue
            mode = "binary"
            active_solvent_smiles = None
            active_solvent_name = None
            if fraction_type is None:
                excluded.append({"row_index": row_index, "reason": "unsupported_fraction_type", "doi": doi})
                continue

        for metric, solubility in measurements:
            if mode == "single":
                records.append(
                    Record(
                        dataset="mixturesoldb",
                        row_index=row_index,
                        doi=doi,
                        mode=mode,
                        metric=metric,
                        temperature=temperature,
                        solubility=solubility,
                        fraction_type=None,
                        fraction_solvent1=None,
                        compound_name=clean_text(row.get("Compound_Name")),
                        solute_smiles=smiles_by_field["SMILES_Solute"],
                        solvent1_name=active_solvent_name,
                        solvent2_name=None,
                        solvent1_smiles=active_solvent_smiles,
                        solvent2_smiles=None,
                    )
                )
            else:
                records.append(
                    Record(
                        dataset="mixturesoldb",
                        row_index=row_index,
                        doi=doi,
                        mode=mode,
                        metric=metric,
                        temperature=temperature,
                        solubility=solubility,
                        fraction_type=fraction_type,
                        fraction_solvent1=round(fraction_solvent1, ROUND_FRACTION),
                        compound_name=clean_text(row.get("Compound_Name")),
                        solute_smiles=smiles_by_field["SMILES_Solute"],
                        solvent1_name=clean_text(row.get("Solvent1")),
                        solvent2_name=clean_text(row.get("Solvent2")),
                        solvent1_smiles=smiles_by_field["SMILES_Solvent1"],
                        solvent2_smiles=smiles_by_field["SMILES_Solvent2"],
                        fraction_solvent1_raw=fraction_solvent1,
                    )
                )

    return _dedupe_records(records), excluded


def _compare_csv_files_mixturesoldb(predicted_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    predicted_records, predicted_excluded, predicted_total_rows = _load_standardized_records_for_mixturesoldb(predicted_path)
    reference_records, reference_excluded = _load_mixturesoldb_records(reference_path)
    reference_total_rows = len(pd.read_csv(reference_path, low_memory=False))
    return _build_report(
        comparison_mode="mixturesoldb",
        predicted_records=predicted_records,
        predicted_excluded=predicted_excluded,
        predicted_total_rows=predicted_total_rows,
        reference_records=reference_records,
        reference_excluded=reference_excluded,
        reference_total_rows=reference_total_rows,
        reference_label="MixtureSolDB",
    )


def _augment_summary(
    report: dict[str, Any],
    predicted_records: list[Record],
    reference_records: list[Record],
    predicted_excluded: list[dict[str, Any]],
    reference_excluded: list[dict[str, Any]],
) -> None:
    summary = report["summary"]

    predicted_dois = {record.doi for record in predicted_records if record.doi}
    reference_dois = {record.doi for record in reference_records if record.doi}
    shared_dois = predicted_dois & reference_dois

    predicted_shared_rows = sum(1 for record in predicted_records if record.doi in shared_dois)
    reference_shared_rows = sum(1 for record in reference_records if record.doi in shared_dois)
    matched_rows = int(summary["exact_matches"]) + int(summary["partial_matches"])

    summary["comparison_scope"] = "shared_doi"
    summary["predicted_total_rows"] = len(predicted_records) + len(predicted_excluded)
    summary["reference_total_rows"] = len(reference_records) + len(reference_excluded)
    summary["predicted_comparable_rows_all_dois"] = len(predicted_records)
    summary["reference_comparable_rows_all_dois"] = len(reference_records)
    summary["predicted_unique_dois"] = len(predicted_dois)
    summary["reference_unique_dois"] = len(reference_dois)
    summary["shared_doi_count"] = len(shared_dois)
    summary["predicted_records_with_shared_doi"] = predicted_shared_rows
    summary["reference_records_with_shared_doi"] = reference_shared_rows
    summary["predicted_records_without_shared_doi"] = len(predicted_records) - predicted_shared_rows
    summary["reference_records_without_shared_doi"] = len(reference_records) - reference_shared_rows
    summary["matched_records"] = matched_rows
    summary["exact_match_rate_predicted_shared_doi_percent"] = round(
        _percentage(int(summary["exact_matches"]), predicted_shared_rows), 2
    )
    summary["exact_match_rate_reference_shared_doi_percent"] = round(
        _percentage(int(summary["exact_matches"]), reference_shared_rows), 2
    )
    summary["paired_match_rate_predicted_shared_doi_percent"] = round(
        _percentage(matched_rows, predicted_shared_rows), 2
    )
    summary["paired_match_rate_reference_shared_doi_percent"] = round(
        _percentage(matched_rows, reference_shared_rows), 2
    )
    summary["predicted_exclusion_reasons"] = _count_exclusion_reasons(predicted_excluded)
    summary["reference_exclusion_reasons"] = _count_exclusion_reasons(reference_excluded)


def compare_csv_files(
    predicted_path: str | Path,
    reference_path: str | Path,
    *,
    predicted_format: str = "standardized",
    reference_format: str = "standardized",
) -> dict[str, Any]:
    if predicted_format == "standardized" and reference_format == "bigsoldb":
        return _compare_csv_files_bigsoldb(predicted_path, reference_path)
    if predicted_format == "standardized" and reference_format == "mixturesoldb":
        return _compare_csv_files_mixturesoldb(predicted_path, reference_path)

    predicted_records, predicted_excluded = load_records(predicted_path, predicted_format, "predicted")
    reference_records, reference_excluded = load_records(reference_path, reference_format, "reference")
    predicted_dois = {record.doi for record in predicted_records if record.doi}
    reference_dois = {record.doi for record in reference_records if record.doi}
    shared_dois = predicted_dois & reference_dois
    predicted_records_with_shared_doi = _records_with_dois(predicted_records, shared_dois)
    reference_records_with_shared_doi = _records_with_dois(reference_records, shared_dois)
    predicted_excluded_with_shared_doi = _exclusions_with_dois(predicted_excluded, shared_dois)
    reference_excluded_with_shared_doi = _exclusions_with_dois(reference_excluded, shared_dois)
    report = compare_record_sets(
        predicted_records_with_shared_doi,
        reference_records_with_shared_doi,
        predicted_excluded=predicted_excluded_with_shared_doi,
        reference_excluded=reference_excluded_with_shared_doi,
    )
    report["comparison_mode"] = "generic"
    report["excluded_predicted"] = predicted_excluded
    report["excluded_reference"] = reference_excluded
    report["summary"]["excluded_predicted"] = len(predicted_excluded_with_shared_doi)
    report["summary"]["excluded_reference"] = len(reference_excluded_with_shared_doi)
    report["summary"]["excluded_predicted_all_dois"] = len(predicted_excluded)
    report["summary"]["excluded_reference_all_dois"] = len(reference_excluded)
    _augment_summary(report, predicted_records, reference_records, predicted_excluded, reference_excluded)
    return report


def print_comparison_summary(
    report: dict[str, Any],
    *,
    predicted_label: str = "Predicted CSV",
    reference_label: str = "Reference CSV",
) -> None:
    comparison_mode = report.get("comparison_mode", "generic")
    if comparison_mode in {"bigsoldb", "mixturesoldb"}:
        summary = report["summary"]
        reference_label = report.get("reference_label", reference_label)
        reference_source_rows_all = summary.get("reference_comparable_source_rows_all_dois")
        reference_source_rows_shared = summary.get("reference_comparable_source_rows_with_shared_doi")
        reference_smiles_skips_all = summary.get("reference_missing_smiles_skipped_all_dois")
        predicted_exclusions_all = summary.get("predicted_exclusion_reasons_all_dois", {})
        predicted_exclusions_shared = summary.get("predicted_exclusion_reasons", {})
        reference_metric_selection = summary.get("reference_metric_selection", {})

        print("\n" + "=" * 80)
        print("INPUTS AND SCOPE")
        print("=" * 80)
        print(f"{predicted_label} raw CSV rows: {summary['predicted_total_rows']}")
        print(f"{reference_label} raw CSV rows: {summary['reference_total_rows']}")
        print("Comparison scope: shared DOI records only")
        print(f"Unique comparable {predicted_label} DOIs: {summary['predicted_unique_dois']}")
        print(f"Unique comparable {reference_label} DOIs: {summary['reference_unique_dois']}")
        print(f"Shared DOIs: {summary['shared_doi_count']}")

        print("\n" + "=" * 80)
        print("NORMALIZED RECORD COUNTS")
        print("=" * 80)
        print(
            f"{predicted_label} comparable measurement records, all DOIs: "
            f"{summary['predicted_comparable_rows_all_dois']} / {summary['predicted_total_rows']} raw rows"
        )
        if reference_source_rows_all is None:
            print(
                f"{reference_label} comparable measurement records, all DOIs: "
                f"{summary['reference_comparable_rows_all_dois']}"
            )
        else:
            print(
                f"{reference_label} comparable measurement records, all DOIs: "
                f"{summary['reference_comparable_rows_all_dois']} from "
                f"{reference_source_rows_all} source rows"
            )
        if comparison_mode == "mixturesoldb":
            print(
                "Note: one MixtureSolDB source row can produce two records when both "
                "mole-fraction and g/100g solubilities are present."
            )
            if reference_metric_selection:
                print(
                    "Collapsed MixtureSolDB source rows with both solubility columns: "
                    f"{reference_metric_selection.get('multi_metric_source_rows', 0)} "
                    "source rows; removed "
                    f"{reference_metric_selection.get('extra_metric_records_removed', 0)} "
                    "extra metric records."
                )
        reference_source_note = (
            f" from {reference_source_rows_shared} source rows"
            if reference_source_rows_shared is not None
            else ""
        )
        print(
            f"Shared-DOI {reference_label} comparable measurement records: "
            f"{summary['reference_comparable_rows']}{reference_source_note}"
        )
        if reference_smiles_skips_all is None:
            print(
                f"Shared-DOI {reference_label} source rows skipped for missing reference SMILES: "
                f"{summary['reference_missing_smiles_skipped']}"
            )
        else:
            print(
                f"Shared-DOI {reference_label} source rows skipped for missing reference SMILES: "
                f"{summary['reference_missing_smiles_skipped']} "
                f"({reference_smiles_skips_all} across all DOIs)"
            )
        print(
            f"Shared-DOI unique {reference_label} comparison keys, ignoring SMILES/InChI: "
            f"{summary['reference_keys_without_smiles']}"
        )
        print(
            f"Shared-DOI unique {reference_label} comparison keys, including InChI: "
            f"{summary['reference_keys_with_smiles']}"
        )

        print("\n" + "=" * 80)
        print("COMPARABILITY SUMMARY")
        print("=" * 80)
        print(
            f"Shared-DOI comparable {predicted_label} measurement records: "
            f"{summary['predicted_comparable_rows']} / {summary['predicted_comparable_rows_all_dois']}"
        )
        print(
            f"{predicted_label} comparable records excluded because DOI is absent in {reference_label}: "
            f"{summary['excluded_missing_doi']}"
        )
        print(
            f"{reference_label} comparable records excluded because DOI is absent in {predicted_label}: "
            f"{summary['reference_comparable_rows_without_shared_doi']}"
        )
        if predicted_exclusions_all:
            print(
                f"{predicted_label} raw rows excluded before DOI filtering: "
                f"{sum(predicted_exclusions_all.values())}"
            )
        if predicted_exclusions_shared:
            print(f"{predicted_label} raw rows excluded within shared DOI scope:")
            for reason, count in predicted_exclusions_shared.items():
                print(f"  {reason}: {count}")

        without_smiles = summary["without_smiles"]
        without_smiles_reference_records = without_smiles.get(
            "reference_records_covered",
            without_smiles.get("reference_rows_covered", 0),
        )
        print("\n" + "=" * 80)
        print("COMPARISON WITHOUT SMILES")
        print("=" * 80)
        print(f"Matching unique keys: {without_smiles['matching_unique_keys']}")
        print(
            f"Matched {predicted_label} records: {without_smiles['matched_predicted_rows']} / "
            f"{summary['predicted_comparable_rows']} "
            f"({without_smiles['coverage_predicted_shared_doi_percent']:.2f}%)"
        )
        print(f"Unmatched {predicted_label} records: {without_smiles['non_matching_predicted_rows']}")
        print(
            f"Covered {reference_label} measurement records: "
            f"{without_smiles_reference_records} / {summary['reference_comparable_rows']} "
            f"({without_smiles['reference_coverage_percent']:.2f}%)"
        )
        if reference_source_rows_shared is not None:
            without_smiles_source_rows = without_smiles.get(
                "reference_source_rows_covered",
                without_smiles_reference_records,
            )
            print(
                f"Covered {reference_label} source rows: "
                f"{without_smiles_source_rows} / {reference_source_rows_shared} "
                f"({without_smiles.get('reference_source_row_coverage_percent', 0.0):.2f}%)"
            )

        with_smiles = summary["with_smiles"]
        with_smiles_reference_records = with_smiles.get(
            "reference_records_covered",
            with_smiles.get("reference_rows_covered", 0),
        )
        print("\n" + "=" * 80)
        print("COMPARISON INCLUDING SMILES")
        print("=" * 80)
        print(f"Matching unique keys including InChI: {with_smiles['matching_unique_keys']}")
        print(
            f"Matched {predicted_label} records: {with_smiles['matched_predicted_rows']} / "
            f"{summary['predicted_comparable_rows']} "
            f"({with_smiles['coverage_predicted_shared_doi_percent']:.2f}%)"
        )
        print(f"Unmatched {predicted_label} records: {with_smiles['non_matching_predicted_rows']}")
        print(
            f"Covered {reference_label} measurement records: "
            f"{with_smiles_reference_records} / {summary['reference_comparable_rows']} "
            f"({with_smiles['reference_coverage_percent']:.2f}%)"
        )
        if reference_source_rows_shared is not None:
            with_smiles_source_rows = with_smiles.get(
                "reference_source_rows_covered",
                with_smiles_reference_records,
            )
            print(
                f"Covered {reference_label} source rows: "
                f"{with_smiles_source_rows} / {reference_source_rows_shared} "
                f"({with_smiles.get('reference_source_row_coverage_percent', 0.0):.2f}%)"
            )

        mismatch_analysis = summary["smiles_mismatch_analysis"]
        print("\n" + "=" * 80)
        print("SMILES/INCHI IMPACT")
        print("=" * 80)
        print(
            f"{predicted_label} records matched when identity is ignored: "
            f"{mismatch_analysis['rows_matching_without_smiles']}"
        )
        print(
            f"Matched records lost when InChI is required: "
            f"{mismatch_analysis['rows_fail_once_smiles_required']}"
        )
        print(
            f"Unique DOIs with non-matching InChI-aware rows: "
            f"{mismatch_analysis['unique_dois_with_non_matching_rows']}"
        )
        return

    summary = report["summary"]

    print("\n" + "=" * 80)
    print("COMPARABILITY SUMMARY")
    print("=" * 80)
    print(
        f"Shared-DOI {predicted_label} comparable rows: "
        f"{summary['predicted_records']} / {summary['predicted_comparable_rows_all_dois']}"
    )
    print(
        f"Shared-DOI {reference_label} comparable rows: "
        f"{summary['reference_records']} / {summary['reference_comparable_rows_all_dois']}"
    )
    print(f"{predicted_label} unique DOIs: {summary['predicted_unique_dois']}")
    print(f"{reference_label} unique DOIs: {summary['reference_unique_dois']}")
    print(f"Shared DOIs: {summary['shared_doi_count']}")
    print(
        f"{predicted_label} comparable rows with DOI present in {reference_label}: "
        f"{summary['predicted_records_with_shared_doi']}"
    )
    print(
        f"{reference_label} comparable rows with DOI present in {predicted_label}: "
        f"{summary['reference_records_with_shared_doi']}"
    )
    print(
        f"{predicted_label} comparable rows with DOI absent in {reference_label}: "
        f"{summary['predicted_records_without_shared_doi']}"
    )
    print(
        f"{reference_label} comparable rows with DOI absent in {predicted_label}: "
        f"{summary['reference_records_without_shared_doi']}"
    )

    predicted_exclusion_reasons = summary.get("predicted_exclusion_reasons", {})
    reference_exclusion_reasons = summary.get("reference_exclusion_reasons", {})
    for reason, count in predicted_exclusion_reasons.items():
        print(f"Excluded from {predicted_label} comparison - {reason}: {count}")
    for reason, count in reference_exclusion_reasons.items():
        print(f"Excluded from {reference_label} comparison - {reason}: {count}")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Exact matches: {summary['exact_matches']}")
    print(f"Partial same-system pairs: {summary['partial_matches']}")
    print(f"Matched rows (exact + partial): {summary['matched_records']}")
    print(
        f"Exact match coverage over shared-DOI {predicted_label} rows: "
        f"{summary['exact_match_rate_predicted_shared_doi_percent']:.2f}%"
    )
    print(
        f"Exact match coverage over shared-DOI {reference_label} rows: "
        f"{summary['exact_match_rate_reference_shared_doi_percent']:.2f}%"
    )
    print(
        f"Paired coverage over shared-DOI {predicted_label} rows: "
        f"{summary['paired_match_rate_predicted_shared_doi_percent']:.2f}%"
    )
    print(
        f"Paired coverage over shared-DOI {reference_label} rows: "
        f"{summary['paired_match_rate_reference_shared_doi_percent']:.2f}%"
    )
    print(f"Missing in prediction: {summary['missing_in_prediction']}")
    print(f"Extra in prediction: {summary['extra_in_prediction']}")


def write_comparison_report(report: dict[str, Any], output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return target_path
