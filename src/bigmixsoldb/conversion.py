from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

from bigmixsoldb.density import chemical_identity, load_density_manifest, select_exact_density
from bigmixsoldb.postprocess import is_missing, parse_numeric_value
from bigmixsoldb.units import (
    UnitConversion,
    basis_matches_solvent,
    parse_solubility_unit,
)

AUDIT_COLUMNS = ("_Original Solubility", "_Original Solubility Unit")

_DIRECT_FRACTION_METHODS = {
    "canonical_mole_fraction",
    "canonical_mass_fraction",
    "canonical_volume_fraction",
    "percentage",
    "scaled_fraction",
    "mass_fraction",
}
_SINGLE_SOLVENT_METHODS = {"basis_ratio", "molality", "molarity", "mass_volume"}


@lru_cache(maxsize=16384)
def _molar_mass_text(smiles: str) -> float | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    value = float(Descriptors.MolWt(molecule))
    return value if math.isfinite(value) and value > 0 else None


def _molar_mass(smiles: Any) -> float | None:
    return _molar_mass_text(str(smiles or "").strip())


@lru_cache(maxsize=16384)
def _identity_text(smiles: str) -> tuple[str, str] | None:
    with rdBase.BlockLogs():
        return chemical_identity(smiles)


def _identity(smiles: Any) -> tuple[str, str] | None:
    return _identity_text(str(smiles or "").strip())


def _active_solvent_count(row: Mapping[str, Any]) -> int:
    count = sum(
        not is_missing(row.get(column))
        for column in ("Solvent 1", "Solvent 2", "Solvent 3")
    )
    extra = row.get("Extra Solvents")
    if not is_missing(extra):
        count += max(1, len([item for item in str(extra).split("|") if item.strip()]))
    return count


def classify_conversion(unit: Any) -> tuple[str, float, str | None]:
    """Compatibility tuple backed by the shared structural unit specification."""

    parsed = parse_solubility_unit(unit)
    return parsed.method, parsed.factor, parsed.result_unit


def _base_report(
    row: Mapping[str, Any],
    parsed: UnitConversion,
    *,
    row_id: Any,
    original_value: Any,
    original_unit: Any,
    requested_temperature: float | None,
    active_solvents: int,
    solvent_identity: tuple[str, str] | None,
    solute_identity: tuple[str, str] | None,
) -> dict[str, Any]:
    notes = list(parsed.normalization_notes)
    solvent_inchikey = solvent_identity[1] if solvent_identity else ""
    return {
        "row_id": row_id,
        "doi": row.get("doi", ""),
        "compound_name": row.get("Compound Name", ""),
        "solvent_name": row.get("Solvent 1", ""),
        "original_value": original_value,
        "original_unit": original_unit,
        "normalized_unit": parsed.normalized_unit,
        "parsed_notation": parsed.parsed_notation,
        "conversion_method": parsed.method,
        "unit_factor": parsed.factor,
        "divisor": parsed.divisor,
        "original_exponent": parsed.original_exponent,
        "normalized_exponent": parsed.normalized_exponent,
        "exponent_sign_canonicalized": parsed.exponent_sign_canonicalized,
        "normalization_notes": notes,
        "normalization_note": "; ".join(notes),
        "basis": parsed.basis,
        "basis_kind": parsed.basis_kind,
        "active_solvent_count": active_solvents,
        "inchikey": solvent_inchikey,
        "solvent_inchikey": solvent_inchikey,
        "solute_inchikey": solute_identity[1] if solute_identity else "",
        "status": "failed",
        "attempted_result": None,
        "result": None,
        "result_unit": parsed.result_unit,
        "requested_temperature_k": requested_temperature,
        "selected_density_g_cm3": None,
        "density_temperature_k": None,
        "density_temperature_delta_k": None,
        "density_source_type": None,
        "density_source": None,
        "density_citation": None,
        "density_raw_evidence": None,
        "density_unit_explicit": None,
        "density_peer_reviewed_verified": None,
        "failure_reason": "",
    }


def _record_density(
    report: dict[str, Any],
    observation: Mapping[str, Any],
    requested_temperature: float,
) -> None:
    density_temperature = parse_numeric_value(observation.get("temperature_k"))
    report.update(
        {
            "selected_density_g_cm3": observation.get("density_g_cm3"),
            "density_temperature_k": density_temperature,
            "density_temperature_delta_k": (
                density_temperature - requested_temperature
                if density_temperature is not None
                else None
            ),
            "density_source_type": observation.get("source_type"),
            "density_source": observation.get("source"),
            "density_citation": observation.get("citation"),
            "density_raw_evidence": observation.get("raw_evidence"),
            "density_unit_explicit": observation.get(
                "unit_explicit",
                not bool(observation.get("unit_inferred_from_section")),
            ),
            "density_peer_reviewed_verified": observation.get(
                "peer_reviewed_verified",
                observation.get("peer_reviewed"),
            ),
        }
    )


def convert_solubility_row(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    *,
    row_id: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    converted = dict(row)
    original_value = row.get("_Original Solubility", row.get("Solubility"))
    original_unit = row.get("_Original Solubility Unit", row.get("Solubility Unit"))

    # Failed and unsupported attempts must retain the exact extracted representation.
    converted["Solubility"] = original_value
    converted["Solubility Unit"] = original_unit

    value = parse_numeric_value(original_value)
    parsed = parse_solubility_unit(original_unit)
    requested_temperature = parse_numeric_value(row.get("Temperature"))
    active_solvents = _active_solvent_count(row)
    solvent_identity = _identity(row.get("SMILES_Solvent_1"))
    solute_identity = _identity(row.get("SMILES_Compound"))
    report = _base_report(
        row,
        parsed,
        row_id=row_id,
        original_value=original_value,
        original_unit=original_unit,
        requested_temperature=requested_temperature,
        active_solvents=active_solvents,
        solvent_identity=solvent_identity,
        solute_identity=solute_identity,
    )

    if value is None or not math.isfinite(value):
        report["failure_reason"] = "invalid_solubility_value"
        return converted, report
    if not parsed.supported:
        report["status"] = "unsupported"
        report["failure_reason"] = parsed.failure_reason
        return converted, report
    if parsed.method in _SINGLE_SOLVENT_METHODS and active_solvents != 1:
        report["failure_reason"] = "not_single_solvent"
        return converted, report

    result: float | None = None
    if parsed.method in _DIRECT_FRACTION_METHODS:
        result = value * parsed.factor
    elif parsed.method == "basis_ratio":
        if not basis_matches_solvent(parsed, row.get("Solvent 1")):
            report["failure_reason"] = "basis_solvent_mismatch"
            return converted, report
        ratio = value * parsed.factor
        result = ratio / (1.0 + ratio) if ratio != -1.0 else math.nan
    elif parsed.method == "molality":
        solvent_mw = _molar_mass(row.get("SMILES_Solvent_1"))
        if solvent_mw is None:
            report["failure_reason"] = "invalid_solvent_smiles"
            return converted, report
        molality = value * parsed.factor
        result = molality / (molality + 1000.0 / solvent_mw)
    else:
        solute_mw = (
            _molar_mass(row.get("SMILES_Compound"))
            if parsed.method == "mass_volume"
            else None
        )
        if parsed.method == "mass_volume" and solute_mw is None:
            report["failure_reason"] = "invalid_solute_smiles"
            return converted, report
        solvent_mw = _molar_mass(row.get("SMILES_Solvent_1"))
        if solvent_mw is None or solvent_identity is None:
            report["failure_reason"] = "invalid_solvent_smiles"
            return converted, report
        if requested_temperature is None:
            report["failure_reason"] = "missing_temperature"
            return converted, report
        density_observation = select_exact_density(
            manifest or {"observations": []},
            solvent_identity[1],
            requested_temperature,
        )
        if density_observation is None:
            report["failure_reason"] = "missing_exact_density"
            return converted, report
        _record_density(report, density_observation, requested_temperature)
        density = parse_numeric_value(density_observation.get("density_g_cm3"))
        if density is None or not 0 < density < 25:
            report["failure_reason"] = "invalid_density_observation"
            return converted, report
        solvent_moles_l = density * 1000.0 / solvent_mw
        if parsed.method == "molarity":
            solute_moles_l = value * parsed.factor
        else:
            assert solute_mw is not None
            solute_moles_l = value * parsed.factor / solute_mw
        denominator = solute_moles_l + solvent_moles_l
        result = solute_moles_l / denominator if denominator != 0 else math.nan

    report["attempted_result"] = result
    if result is None or not math.isfinite(result) or not 0.0 <= result <= 1.0:
        report["failure_reason"] = "invalid_conversion_result"
        return converted, report

    converted["Solubility"] = result
    converted["Solubility Unit"] = parsed.result_unit
    report.update(
        {
            "status": "converted",
            "result": result,
            "failure_reason": "",
        }
    )
    return converted, report


def convert_dataframe(
    dataframe: pd.DataFrame,
    manifest: Mapping[str, Any] | str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    loaded = (
        load_density_manifest(manifest)
        if isinstance(manifest, (str, Path))
        else (manifest or {"observations": []})
    )
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for position, (index, row) in enumerate(dataframe.iterrows()):
        converted, report = convert_solubility_row(row.to_dict(), loaded, row_id=index)
        report["row_position"] = position
        rows.append(converted)
        reports.append(report)
    return pd.DataFrame(rows, columns=dataframe.columns), reports


def write_conversion_report(
    reports: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        pd.DataFrame(reports).to_csv(target, index=False)
    return target
