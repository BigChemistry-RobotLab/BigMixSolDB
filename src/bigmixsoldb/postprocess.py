from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from bigmixsoldb.constants import FRACTION_UNITS, MISSING_VALUES, OUTPUT_COLUMNS
from bigmixsoldb.files import normalize_doi_from_stem
from bigmixsoldb.molecules import MoleculeRecord, load_molecule_lookup
from bigmixsoldb.yaml_utils import load_yaml_document

MULTIPLICATION_SYMBOLS = [r"\\times", r"\\cdot", r"\times", r"\cdot", "·", "×", ""]
IONIC_ADDITIVE_KEYWORDS = (
    " chloride",
    " bromide",
    " iodide",
    " nitrate",
    " nitrite",
    " sulfate",
    " sulfite",
    " sulfide",
    " phosphate",
    " perchlorate",
    " carbonate",
    " bicarbonate",
    " hydroxide",
    " ammonium",
    " lithium",
    " sodium",
    " potassium",
    " calcium",
    " magnesium",
    " barium",
    " strontium",
    " zinc",
    " copper",
    " nickel",
    " cobalt",
    " manganese",
    " aluminum",
    " caesium",
    " cesium",
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_VALUES


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def clean_numeric_artifact(value: float, *, significant_digits: int = 15) -> float:
    if not math.isfinite(value):
        return value
    rounded = float(f"{value:.{significant_digits}g}")
    tolerance = 1e-15 * max(1.0, abs(value))
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=tolerance):
        return rounded
    return value


def split_pipe_field(value: Any) -> list[str]:
    if is_missing(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def normalize_unit_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip().lower()
    text = text.replace("$", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("−", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_numeric_value(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return clean_numeric_artifact(float(value))

    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass

    parenthetical_uncertainty = re.fullmatch(
        r"\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\([^)]*\)\s*",
        text,
    )
    if parenthetical_uncertainty:
        try:
            return clean_numeric_artifact(float(parenthetical_uncertainty.group(1)))
        except ValueError:
            pass

    cleaned = text
    if "±" in cleaned:
        cleaned = cleaned.split("±", maxsplit=1)[0]
    if r"\pm" in cleaned:
        cleaned = cleaned.split(r"\pm", maxsplit=1)[0]

    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("(", "")
    cleaned = cleaned.replace(")", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("−", "")
    cleaned = cleaned.replace("\\\\", "\\")

    compact = cleaned.replace(" ", "")

    patterns = [
        r"([+-]?\d*\.?\d+)(?:\\times|\\cdot|·|×)10\^\{?([+-]?\d+)\}?",
        r"([+-]?\d*\.?\d+)e([+-]?\d+)",
        r"([+-]?\d*\.?\d+)10\^\{?([+-]?\d+)\}?",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            continue
        base = float(match.group(1))
        exponent = int(match.group(2))
        return clean_numeric_artifact(base * (10**exponent))

    try:
        return clean_numeric_artifact(float(compact))
    except ValueError:
        return None


def extract_fraction_unit(unit: Any) -> tuple[float, str] | None:
    if is_missing(unit):
        return None

    clean_str = str(unit).replace("$", "").replace(" ", "").replace(",", "").lower()
    clean_str = clean_str.replace("−", "")

    aliases = [
        (r"(x(?:_\{?[a-z0-9]+\}?)?|molefraction|mol%|mole%)", "mole fraction"),
        (r"(w(?:_\{?[a-z0-9]+\}?)?|massfraction|wt%|weight%|mass%|w%)", "mass fraction"),
        (r"(v(?:_\{?[a-z0-9]+\}?)?|volumefraction|vol%|v/v|phi|ϕ)", "volume fraction"),
    ]

    for symbol in MULTIPLICATION_SYMBOLS:
        symbol_pattern = re.escape(symbol)
        for pattern, canonical_unit in aliases:
            match = re.search(rf"10\^{{?([+-]?\d+)}}?{symbol_pattern}{pattern}", clean_str)
            if match:
                return 10 ** abs(int(match.group(1))), canonical_unit

            match = re.search(rf"{pattern}{symbol_pattern}10\^{{?([+-]?\d+)}}?", clean_str)
            if match:
                return 10 ** abs(int(match.group(match.lastindex))), canonical_unit

            match = re.search(rf"(\d+){symbol_pattern}{pattern}", clean_str)
            if match:
                return float(match.group(1)), canonical_unit

            match = re.search(rf"{pattern}{symbol_pattern}(\d+)", clean_str)
            if match:
                return float(match.group(match.lastindex)), canonical_unit

    for pattern, canonical_unit in aliases:
        if re.fullmatch(pattern, clean_str):
            if "%" in clean_str or "percent" in clean_str:
                return 100.0, canonical_unit
            return 1.0, canonical_unit

    return None


def normalize_concentration(value: Any, unit: Any) -> tuple[float | str, str]:
    fraction_info = extract_fraction_unit(unit)
    if fraction_info is None:
        numeric = parse_numeric_value(value)
        if numeric is not None:
            return numeric, clean_text(unit)
        return clean_text(value), clean_text(unit)

    divisor, canonical_unit = fraction_info
    numeric = parse_numeric_value(value)
    if numeric is None:
        return "", canonical_unit
    return clean_numeric_artifact(numeric / divisor), canonical_unit


def normalize_solubility(value: Any, unit: Any) -> tuple[float | str, str]:
    numeric = parse_numeric_value(value)
    unit_text = normalize_unit_text(unit)

    if numeric is None:
        return clean_text(value), ""
    if unit_text is None:
        return clean_numeric_artifact(numeric), ""

    fraction_info = extract_fraction_unit(unit)
    if fraction_info is not None:
        divisor, canonical_unit = fraction_info
        return clean_numeric_artifact(numeric / divisor), canonical_unit

    normalized_unit = unit_text.replace(" ", "")

    if normalized_unit in {"mol/mol", "mole/mole"}:
        return clean_numeric_artifact(numeric), "mole fraction"
    if normalized_unit in {"mmol/mol", "millimol/mol", "millimole/mole"}:
        return clean_numeric_artifact(numeric / 1_000.0), "mole fraction"
    scaled_mol_fraction = re.fullmatch(
        r"10\^(?:([+-]?\d+)|\{([+-]?\d+)\})(?:\\cdot|\\times|·|×)?mol/mol",
        normalized_unit,
    )
    if scaled_mol_fraction:
        exponent = int(scaled_mol_fraction.group(1) or scaled_mol_fraction.group(2))
        return clean_numeric_artifact(numeric * (10**exponent)), "mole fraction"
    if normalized_unit in {"x", "y", "molefraction"}:
        return clean_numeric_artifact(numeric), "mole fraction"

    if normalized_unit in {"g/g", "gg-1", "kg/kg", "kgkg-1"}:
        return clean_numeric_artifact(numeric), "mass fraction"
    if normalized_unit in {"g/100g", "g/100gsolution", "g/100gmixture"}:
        return clean_numeric_artifact(numeric / 100.0), "mass fraction"
    if normalized_unit in {"10g/100g", "10g/100gsolution", "10g/100gmixture"}:
        return clean_numeric_artifact(numeric / 10.0), "mass fraction"
    if normalized_unit in {"mg/g", "mgg-1"}:
        return clean_numeric_artifact(numeric / 1_000.0), "mass fraction"
    if normalized_unit in {"g/kg", "gkg-1"}:
        return clean_numeric_artifact(numeric / 1_000.0), "mass fraction"

    return clean_numeric_artifact(numeric), ""


def normalize_temperature(value: Any, unit: Any) -> tuple[float | str, str]:
    numeric = parse_numeric_value(value)
    unit_text = normalize_unit_text(unit)
    if numeric is None:
        return clean_text(value), clean_text(unit)
    if unit_text is None or unit_text == "":
        if numeric >= 170.0:
            return clean_numeric_artifact(numeric), "K"
        return clean_numeric_artifact(numeric + 273.15), "K"
    compact = unit_text.replace(" ", "")
    if compact in {"k", "kelvin"}:
        return clean_numeric_artifact(numeric), "K"
    if compact in {"c", "celsius", "°c", "degc", r"\circc", r"^\circc"}:
        return clean_numeric_artifact(numeric + 273.15), "K"
    if compact in {"f", "fahrenheit", "°f", "degf", r"\circf", r"^\circf"}:
        return clean_numeric_artifact((numeric - 32.0) * 5.0 / 9.0 + 273.15), "K"
    return clean_numeric_artifact(numeric), clean_text(unit)


def normalize_pressure(value: Any, unit: Any) -> tuple[float | str, str]:
    numeric = parse_numeric_value(value)
    unit_text = normalize_unit_text(unit)
    if numeric is None:
        return clean_text(value), ""
    if unit_text is None or unit_text == "":
        return clean_numeric_artifact(numeric), ""

    compact = unit_text.replace(" ", "")
    compact = compact.replace("(", "").replace(")", "").replace(".", "")
    if compact in {"pa", "pascal", "pascals"}:
        return clean_numeric_artifact(numeric), "Pa"
    if compact in {"hpa", "hectopascal", "hectopascals"}:
        return clean_numeric_artifact(numeric * 100.0), "Pa"
    if compact == "kpa":
        return clean_numeric_artifact(numeric * 1_000.0), "Pa"
    if compact == "mpa":
        return clean_numeric_artifact(numeric * 1_000_000.0), "Pa"
    if compact in {"bar", "barg"}:
        return clean_numeric_artifact(numeric * 100_000.0), "Pa"
    if compact == "mbar":
        return clean_numeric_artifact(numeric * 100.0), "Pa"
    if compact == "atm":
        return clean_numeric_artifact(numeric * 101_325.0), "Pa"
    if compact in {"psi", "psia"}:
        return clean_numeric_artifact(numeric * 6_894.757293168), "Pa"
    if compact in {"torr", "mmhg"}:
        return clean_numeric_artifact(numeric * 133.322368421), "Pa"
    scaled_pascal = re.fullmatch(
        r"(?:\\times|\\cdot|×|·)?10\^(?:([+-]?\d+)|\{([+-]?\d+)\})pa",
        compact,
    )
    if scaled_pascal:
        exponent = int(scaled_pascal.group(1) or scaled_pascal.group(2))
        return clean_numeric_artifact(numeric * (10**exponent)), "Pa"
    return clean_numeric_artifact(numeric), ""


def fill_missing_binary_fraction(row: dict[str, Any]) -> None:
    if row["Concentration Unit"] not in FRACTION_UNITS:
        return

    active_columns = [
        concentration_column
        for solvent_column, concentration_column in (
            ("Solvent 1", "Concentration Solvent 1"),
            ("Solvent 2", "Concentration Solvent 2"),
            ("Solvent 3", "Concentration Solvent 3"),
        )
        if clean_text(row[solvent_column]) != ""
    ]
    if len(active_columns) < 2:
        return

    known_total = 0.0
    missing_columns: list[str] = []
    for concentration_column in active_columns:
        numeric = parse_numeric_value(row[concentration_column])
        if numeric is None:
            missing_columns.append(concentration_column)
            continue
        if not 0.0 <= numeric <= 1.0:
            return
        known_total += numeric

    if len(missing_columns) != 1:
        return

    remainder = 1.0 - known_total
    tolerance = 1e-9
    if -tolerance <= remainder <= 1.0 + tolerance:
        row[missing_columns[0]] = clean_numeric_artifact(min(max(remainder, 0.0), 1.0))


def simplify_pure_solvent_row(row: dict[str, Any]) -> None:
    if row["Concentration Unit"] not in FRACTION_UNITS:
        return

    active_solvents = [
        (solvent_column, concentration_column)
        for solvent_column, concentration_column in (
            ("Solvent 1", "Concentration Solvent 1"),
            ("Solvent 2", "Concentration Solvent 2"),
            ("Solvent 3", "Concentration Solvent 3"),
        )
        if clean_text(row[solvent_column]) != ""
    ]
    if len(active_solvents) < 2:
        return

    tolerance = 1e-9

    concentrations: list[float] = []
    for _, concentration_column in active_solvents:
        numeric = parse_numeric_value(row[concentration_column])
        if numeric is None:
            return
        concentrations.append(numeric)

    pure_index: int | None = None
    for index, numeric in enumerate(concentrations):
        if abs(numeric - 1.0) >= tolerance:
            continue
        if all(abs(other) < tolerance for other_index, other in enumerate(concentrations) if other_index != index):
            pure_index = index
            break

    if pure_index is None:
        return

    pure_solvent_column, _ = active_solvents[pure_index]
    row["Solvent 1"] = row[pure_solvent_column]
    row["Solvent 2"] = ""
    row["Solvent 3"] = ""
    row["Extra Solvents"] = ""
    row["Concentration Solvent 1"] = ""
    row["Concentration Solvent 2"] = ""
    row["Concentration Solvent 3"] = ""
    row["Concentration Unit"] = ""


def simplify_nonfraction_additive_row(row: dict[str, Any]) -> None:
    if row["Concentration Unit"] in FRACTION_UNITS or is_missing(row["Concentration Unit"]):
        return
    if clean_text(row["Solvent 1"]).lower() != "water":
        return

    secondary_solvents = [clean_text(row[column]).lower() for column in ("Solvent 2", "Solvent 3")]
    if not any(
        solvent_text and solvent_text != "-" and any(keyword in f" {solvent_text}" for keyword in IONIC_ADDITIVE_KEYWORDS)
        for solvent_text in secondary_solvents
    ):
        return

    active_solvents = [
        (solvent_column, concentration_column)
        for solvent_column, concentration_column in (
            ("Solvent 1", "Concentration Solvent 1"),
            ("Solvent 2", "Concentration Solvent 2"),
            ("Solvent 3", "Concentration Solvent 3"),
        )
        if clean_text(row[solvent_column]) != "-" and clean_text(row[solvent_column]) != ""
    ]
    if len(active_solvents) < 2:
        return

    present_concentrations = [
        (solvent_column, concentration_column)
        for solvent_column, concentration_column in active_solvents
        if parse_numeric_value(row[concentration_column]) is not None
    ]
    missing_concentrations = [
        (solvent_column, concentration_column)
        for solvent_column, concentration_column in active_solvents
        if parse_numeric_value(row[concentration_column]) is None
    ]

    if len(present_concentrations) != 1 or not missing_concentrations:
        return

    present_solvent_column, _ = present_concentrations[0]
    if present_solvent_column != "Solvent 1":
        return

    base_solvent_column, _ = missing_concentrations[0]
    row["Solvent 1"] = row[base_solvent_column]
    row["Solvent 2"] = ""
    row["Solvent 3"] = ""
    row["Extra Solvents"] = ""
    row["Concentration Solvent 1"] = ""
    row["Concentration Solvent 2"] = ""
    row["Concentration Solvent 3"] = ""
    row["Concentration Unit"] = ""


MoleculeLookup = Mapping[str, MoleculeRecord | str]


def _molecule_key(name: Any) -> str:
    return clean_text(name).lower()


def molecule_record_for_name(name: Any, molecule_lookup: MoleculeLookup) -> MoleculeRecord | None:
    key = _molecule_key(name)
    if not key:
        return None
    record = molecule_lookup.get(key)
    if isinstance(record, str):
        return MoleculeRecord(smiles=record)
    return record


def molecule_smiles(name: Any, molecule_lookup: MoleculeLookup) -> str:
    record = molecule_record_for_name(name, molecule_lookup)
    if record is None or not record.enabled:
        return ""
    return record.smiles


def molecule_is_disabled(name: Any, molecule_lookup: MoleculeLookup) -> bool:
    record = molecule_record_for_name(name, molecule_lookup)
    return record is not None and not record.enabled


def row_has_disabled_molecule(row: dict[str, Any], molecule_lookup: MoleculeLookup) -> bool:
    molecule_names = [
        row["Compound Name"],
        row["Solvent 1"],
        row["Solvent 2"],
        row["Solvent 3"],
        *split_pipe_field(row.get("Extra Solvents")),
    ]
    return any(molecule_is_disabled(name, molecule_lookup) for name in molecule_names)


def classify_row_mixture_type(row: Mapping[str, Any]) -> str:
    if not is_missing(row.get("Extra Solvents")):
        return "extra"
    if (
        not is_missing(row.get("Solvent 1"))
        and not is_missing(row.get("Solvent 2"))
        and not is_missing(row.get("Solvent 3"))
    ):
        return "ternary"
    if not is_missing(row.get("Solvent 1")) and not is_missing(row.get("Solvent 2")):
        return "binary"
    return "single"


def record_disabled_molecule_row(row: Mapping[str, Any], stats: dict[str, Any]) -> None:
    stats["disabled_molecule_rows_removed"] = stats.get("disabled_molecule_rows_removed", 0) + 1
    by_type = stats.setdefault("disabled_molecule_rows_removed_by_type", {})
    mixture_type = classify_row_mixture_type(row)
    by_type[mixture_type] = by_type.get(mixture_type, 0) + 1


def attach_smiles(row: dict[str, Any], molecule_lookup: MoleculeLookup) -> None:
    row["SMILES_Compound"] = molecule_smiles(row["Compound Name"], molecule_lookup)
    row["SMILES_Solvent_1"] = molecule_smiles(row["Solvent 1"], molecule_lookup)
    row["SMILES_Solvent_2"] = molecule_smiles(row["Solvent 2"], molecule_lookup)
    row["SMILES_Solvent_3"] = molecule_smiles(row["Solvent 3"], molecule_lookup)

    for column in ("SMILES_Compound", "SMILES_Solvent_1", "SMILES_Solvent_2", "SMILES_Solvent_3"):
        if clean_text(row[column]) == "":
            row[column] = ""


def make_base_row(compound_name: str, synonyms: str, solvents: list[str], note: Any, doi: str) -> dict[str, Any]:
    extra_solvents = solvents[3:]
    return {
        "Compound Name": compound_name,
        "Synonyms": synonyms,
        "SMILES_Compound": "",
        "Solvent 1": solvents[0] if len(solvents) > 0 else "",
        "SMILES_Solvent_1": "",
        "Solvent 2": solvents[1] if len(solvents) > 1 else "",
        "SMILES_Solvent_2": "",
        "Solvent 3": solvents[2] if len(solvents) > 2 else "",
        "SMILES_Solvent_3": "",
        "Extra Solvents": " | ".join(extra_solvents) if extra_solvents else "",
        "Concentration Solvent 1": "",
        "Concentration Solvent 2": "",
        "Concentration Solvent 3": "",
        "Concentration Unit": "",
        "Solubility": "",
        "Solubility Unit": "",
        "Temperature": "",
        "Temperature Unit": "",
        "Pressure": "",
        "Pressure Unit": "",
        "Notes": clean_text(note),
        "Requires Review": False,
        "doi": doi,
    }


def apply_entry_level_fields(row: dict[str, Any], entry: dict[str, Any]) -> None:
    field_map = {
        "c1": "Concentration Solvent 1",
        "c2": "Concentration Solvent 2",
        "c3": "Concentration Solvent 3",
        "cu": "Concentration Unit",
        "sol": "Solubility",
        "su": "Solubility Unit",
        "temp": "Temperature",
        "tu": "Temperature Unit",
        "pres": "Pressure",
        "pu": "Pressure Unit",
    }
    for input_field, output_field in field_map.items():
        if input_field in entry and not is_missing(entry[input_field]):
            row[output_field] = entry[input_field]


def apply_data_row(row: dict[str, Any], header: list[str], data_row: Any) -> None:
    if isinstance(data_row, list):
        values = [str(value).strip().strip("[]") for value in data_row]
    elif isinstance(data_row, str):
        values = [part.strip().strip("[]") for part in data_row.split(",")]
    else:
        values = [str(data_row).strip()]

    field_map = {
        "solv": lambda value: row.update(_solvent_override(value)),
        "c1": lambda value: row.__setitem__("Concentration Solvent 1", value),
        "c2": lambda value: row.__setitem__("Concentration Solvent 2", value),
        "c3": lambda value: row.__setitem__("Concentration Solvent 3", value),
        "cu": lambda value: row.__setitem__("Concentration Unit", value),
        "sol": lambda value: row.__setitem__("Solubility", value),
        "su": lambda value: row.__setitem__("Solubility Unit", value),
        "temp": lambda value: row.__setitem__("Temperature", value),
        "tu": lambda value: row.__setitem__("Temperature Unit", value),
        "pres": lambda value: row.__setitem__("Pressure", value),
        "pu": lambda value: row.__setitem__("Pressure Unit", value),
    }

    for index, value in enumerate(values):
        if index >= len(header):
            break
        handler = field_map.get(header[index])
        if handler is not None:
            handler(value)


def _solvent_override(value: Any) -> dict[str, Any]:
    solvents = split_pipe_field(value)
    extra_solvents = solvents[3:]
    return {
        "Solvent 1": solvents[0] if len(solvents) > 0 else "",
        "Solvent 2": solvents[1] if len(solvents) > 1 else "",
        "Solvent 3": solvents[2] if len(solvents) > 2 else "",
        "Extra Solvents": " | ".join(extra_solvents) if extra_solvents else "",
    }


def normalize_row(row: dict[str, Any], document_review_required: bool, molecule_lookup: MoleculeLookup) -> dict[str, Any]:
    # inherit any pre-existing flag on the row OR the document-level review flag
    row["Requires Review"] = bool(row.get("Requires Review", False) or document_review_required)

    for column in (
        "Compound Name",
        "Synonyms",
        "Solvent 1",
        "Solvent 2",
        "Solvent 3",
        "Extra Solvents",
        "Notes",
        "doi",
    ):
        row[column] = clean_text(row[column])

    original_concentration_unit = row["Concentration Unit"]
    row["Concentration Solvent 1"], row["Concentration Unit"] = normalize_concentration(
        row["Concentration Solvent 1"], original_concentration_unit
    )
    row["Concentration Solvent 2"], _ = normalize_concentration(
        row["Concentration Solvent 2"], original_concentration_unit
    )
    row["Concentration Solvent 3"], _ = normalize_concentration(
        row["Concentration Solvent 3"], original_concentration_unit
    )

    fill_missing_binary_fraction(row)
    simplify_pure_solvent_row(row)
    simplify_nonfraction_additive_row(row)

    row["Solubility"], row["Solubility Unit"] = normalize_solubility(row["Solubility"], row["Solubility Unit"])
    row["Temperature"], row["Temperature Unit"] = normalize_temperature(row["Temperature"], row["Temperature Unit"])
    row["Pressure"], row["Pressure Unit"] = normalize_pressure(row["Pressure"], row["Pressure Unit"])

    solubility_numeric = parse_numeric_value(row["Solubility"])
    if row["Solubility Unit"] in FRACTION_UNITS and solubility_numeric is not None:
        if solubility_numeric < 0.0 or solubility_numeric > 1.0:
            row["Requires Review"] = True

    if parse_numeric_value(row["Solubility"]) is None:
        row["Requires Review"] = True

    attach_smiles(row, molecule_lookup)

    for column in OUTPUT_COLUMNS:
        if column not in row:
            row[column] = False if column == "Requires Review" else ""
        elif column != "Requires Review" and is_missing(row[column]):
            row[column] = ""

    return row


def flatten_yaml_file(
    path: str | Path,
    molecule_json: str | Path | None = None,
    molecule_lookup: MoleculeLookup | None = None,
    stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    document, review_required = load_yaml_document(path)
    if molecule_lookup is None:
        molecule_lookup = load_molecule_lookup(molecule_json)
    doi = normalize_doi_from_stem(path)
    rows: list[dict[str, Any]] = []
    document_review_required = review_required

    for item in document:
        if item.get("review_required"):
            document_review_required = True
        if item.get("parsing_issues"):
            document_review_required = True
        if "compound" not in item:
            continue

        compound_name = clean_text(item.get("compound"))
        synonyms = "|".join(split_pipe_field(item.get("synonyms"))) or ""
        entries = item.get("entries") or []
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            solvents = split_pipe_field(entry.get("solv"))
            if not solvents:
                continue

            base_row = make_base_row(compound_name, synonyms, solvents, entry.get("note"), doi)
            apply_entry_level_fields(base_row, entry)
            data_block = entry.get("data")

            if isinstance(data_block, list) and len(data_block) >= 2:
                header_raw = data_block[0]
                if isinstance(header_raw, list):
                    header = [str(item).strip() for item in header_raw]
                else:
                    header = [part.strip() for part in str(header_raw).split(",") if part.strip()]

                for data_row in data_block[1:]:
                    row = dict(base_row)
                    apply_data_row(row, header, data_row)
                    if row_has_disabled_molecule(row, molecule_lookup):
                        if stats is not None:
                            record_disabled_molecule_row(row, stats)
                        continue
                    rows.append(normalize_row(row, document_review_required, molecule_lookup))
            else:
                if row_has_disabled_molecule(base_row, molecule_lookup):
                    if stats is not None:
                        record_disabled_molecule_row(base_row, stats)
                    continue
                rows.append(normalize_row(base_row, document_review_required, molecule_lookup))

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=[*OUTPUT_COLUMNS, "Notes"])

    for column in [*OUTPUT_COLUMNS, "Notes"]:
        if column not in dataframe.columns:
            dataframe[column] = False if column == "Requires Review" else ""

    dataframe = dataframe[[*OUTPUT_COLUMNS, "Notes"]]
    return dataframe


def write_postprocessed_csv(
    yaml_path: str | Path,
    output_path: str | Path,
    molecule_json: str | Path | None = None,
) -> Path:
    dataframe = flatten_yaml_file(yaml_path, molecule_json=molecule_json)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(target_path, index=False)
    return target_path


def filter_complete_rows(
    dataframe: pd.DataFrame,
    *,
    require_smiles: bool = False,
    drop_review_required: bool = False,
) -> pd.DataFrame:
    df = dataframe.copy()

    required_columns = [
        "Compound Name",
        "Solvent 1",
        "Solubility",
        "Solubility Unit",
        "doi",
    ]

    mask = pd.Series(True, index=df.index)
    for column in required_columns:
        mask &= ~df[column].map(is_missing)

    mask &= df["Solubility"].map(lambda value: parse_numeric_value(value) is not None)

    if drop_review_required and "Requires Review" in df.columns:
        mask &= ~df["Requires Review"].fillna(False).astype(bool)

    def row_is_complete(row: pd.Series) -> bool:
        active_solvent_2 = not is_missing(row.get("Solvent 2"))
        active_solvent_3 = not is_missing(row.get("Solvent 3"))

        if require_smiles:
            smiles_columns = ["SMILES_Compound", "SMILES_Solvent_1"]
            if active_solvent_2:
                smiles_columns.append("SMILES_Solvent_2")
            if active_solvent_3:
                smiles_columns.append("SMILES_Solvent_3")
            for smiles_column in smiles_columns:
                if is_missing(row.get(smiles_column)):
                    return False

        return True

    mask &= df.apply(row_is_complete, axis=1)

    filtered = df[mask].drop_duplicates().reset_index(drop=True)
    return filtered


def drop_invalid_fraction_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    df = dataframe.copy()
    solubility = pd.to_numeric(df["Solubility"], errors="coerce")
    invalid_fraction_mask = df["Solubility Unit"].isin(FRACTION_UNITS) & (
        (solubility < 0.0) | (solubility > 1.0)
    )
    return df[~invalid_fraction_mask].reset_index(drop=True)


def dedupe_condition_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    df = dataframe.copy().reset_index(drop=True)
    df["_unit_rank"] = (df["Solubility Unit"].astype(str) == "mole fraction").astype(int)
    df = df.sort_values(["_unit_rank"], ascending=False, kind="stable")

    condition_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in {"Solubility", "Solubility Unit", "Requires Review"}
    ]
    deduped = df.drop_duplicates(subset=condition_columns, keep="first")
    return deduped.drop(columns=["_unit_rank"]).reset_index(drop=True)


def merge_dataframes(
    frames: list[pd.DataFrame],
    *,
    require_smiles: bool = False,
    drop_review_required: bool = False,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    for column in OUTPUT_COLUMNS:
        if column not in merged.columns:
            merged[column] = False if column == "Requires Review" else ""

    merged = filter_complete_rows(
        merged,
        require_smiles=require_smiles,
        drop_review_required=drop_review_required,
    )
    merged = drop_invalid_fraction_rows(merged)
    merged = dedupe_condition_rows(merged)
    # preserve any existing Requires Review flags; ensure missing values are False
    if "Requires Review" in merged.columns:
        merged["Requires Review"] = merged["Requires Review"].fillna(False).astype(bool)
    return merged[OUTPUT_COLUMNS]


def merge_csv_files(
    csv_paths: list[Path],
    *,
    require_smiles: bool = False,
    drop_review_required: bool = False,
) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in csv_paths]
    return merge_dataframes(
        frames,
        require_smiles=require_smiles,
        drop_review_required=drop_review_required,
    )
