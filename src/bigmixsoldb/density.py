from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem, rdBase

MANIFEST_SCHEMA_VERSION = 2
EXACT_TEMPERATURE_TOLERANCE_K = 0.01
PUBCHEM_DELAY_SECONDS = 0.21


def _decimal(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace("\u2212", "-")
    if not text:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def chemical_identity(smiles: Any) -> tuple[str, str] | None:
    text = str(smiles or "").strip()
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    with rdBase.BlockLogs():
        inchikey = Chem.MolToInchiKey(molecule)
    return (canonical_smiles, inchikey) if inchikey else None


def empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requests": [],
        "observations": [],
    }


def validate_density_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Density manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported density manifest schema: "
            f"{payload.get('schema_version')!r}; expected {MANIFEST_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("generated_at"), str) or not payload["generated_at"].strip():
        raise ValueError("Density manifest generated_at must be a non-empty string")
    if not isinstance(payload.get("requests"), list) or not isinstance(
        payload.get("observations"), list
    ):
        raise ValueError("Density manifest requests and observations must be arrays")
    if any(not isinstance(item, dict) for item in payload["requests"]):
        raise ValueError("Every density manifest request must be an object")
    if any(not isinstance(item, dict) for item in payload["observations"]):
        raise ValueError("Every density manifest observation must be an object")
    return payload


def load_density_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return empty_manifest()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_density_manifest(payload)


def write_density_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = validate_density_manifest(dict(manifest))
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return target


def seed_local_density_csv(path: str | Path) -> list[dict[str, Any]]:
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"Temperature_K", "Density_g/cm^3", "SMILES"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Density CSV is missing columns: {', '.join(sorted(missing))}")

    observations: list[dict[str, Any]] = []
    for file_order, row in dataframe.iterrows():
        identity = chemical_identity(row.get("SMILES"))
        temperature = _decimal(row.get("Temperature_K"))
        density = _decimal(row.get("Density_g/cm^3"))
        eligible = bool(
            identity
            and temperature is not None
            and density is not None
            and temperature > 0
            and 0 < density < 25
        )
        canonical_smiles, inchikey = identity or ("", "")
        observations.append(
            {
                "inchikey": inchikey,
                "canonical_smiles": canonical_smiles,
                "temperature_k": temperature,
                "density_g_cm3": density,
                "source_type": "local",
                "source": str(row.get("Source", "")).strip(),
                "citation": str(row.get("Source", "")).strip(),
                "raw_evidence": {
                    column: str(row.get(column, "")) for column in dataframe.columns
                },
                "eligible": eligible,
                "failure_reason": "" if eligible else "invalid_local_observation",
                "unit_inferred_from_section": False,
                "unit_explicit": True,
                "peer_reviewed": bool(str(row.get("Source", "")).strip()),
                "peer_reviewed_verified": bool(str(row.get("Source", "")).strip()),
                "file_order": int(file_order),
                "response_order": int(file_order),
            }
        )
    return observations


def select_exact_density(
    manifest: Mapping[str, Any],
    inchikey: str,
    temperature_k: float,
    *,
    tolerance_k: float = EXACT_TEMPERATURE_TOLERANCE_K,
) -> dict[str, Any] | None:
    local_candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    pubchem_candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for position, raw in enumerate(manifest.get("observations", [])):
        if not isinstance(raw, dict) or not raw.get("eligible", True):
            continue
        if raw.get("inchikey") != inchikey:
            continue
        actual_temperature = _decimal(raw.get("temperature_k"))
        density = _decimal(raw.get("density_g_cm3"))
        if actual_temperature is None or density is None or not 0 < density < 25:
            continue
        if abs(actual_temperature - temperature_k) > tolerance_k + 1e-12:
            continue
        source_type = str(raw.get("source_type", ""))
        if source_type == "local":
            file_order = int(raw.get("file_order", raw.get("response_order", position)))
            local_candidates.append(((file_order, position), dict(raw)))
            continue
        explicit_rank = 0 if raw.get(
            "unit_explicit", not bool(raw.get("unit_inferred_from_section"))
        ) else 1
        verified_peer_rank = 0 if raw.get(
            "peer_reviewed_verified", raw.get("peer_reviewed", False)
        ) else 1
        order = int(raw.get("response_order", position))
        pubchem_candidates.append(
            ((explicit_rank, verified_peer_rank, order, position), dict(raw))
        )
    if local_candidates:
        return min(local_candidates, key=lambda item: item[0])[1]
    if pubchem_candidates:
        return min(pubchem_candidates, key=lambda item: item[0])[1]
    return None


_NON_DENSITY_PROPERTY = re.compile(
    r"\b(?:specific\s+gravity|relative\s+density|bulk\s+density|"
    r"critical\s+density|denser|lighter)\b",
    re.IGNORECASE,
)
_MIXTURE_TERMS = re.compile(
    r"\b(?:mixture|solution|aqueous|binary\s+(?:mixture|system)|"
    r"multi-?component|suspension|emulsion)\b",
    re.IGNORECASE,
)
_INAPPROPRIATE_PHASE = re.compile(
    r"\b(?:vapo[u]?r|gas(?:eous)?|air|ice|solid|crystal(?:line)?|"
    r"supercritical|sublim(?:ed|ation)|saturated\s+air)\b",
    re.IGNORECASE,
)
_RANGE_OR_INEQUALITY = re.compile(
    r"(?:[<>≤≥~]|\b(?:less|greater|range|between|from|about|approx(?:imately)?)\b|"
    r"\d\s*(?:-|–|—)\s*\d)",
    re.IGNORECASE,
)
_RATIO_TEMPERATURE = re.compile(r"\d+(?:[.,]\d+)?\s*°?\s*[CFK]\s*/\s*\d", re.I)
_PRESSURE = re.compile(
    r"\b([+-]?\d+(?:[.,]\d+)?)\s*"
    r"(Pa|kPa|MPa|GPa|mbar|bar|atm|mmHg|torr|psi)\b",
    re.I,
)
_NUMBER = r"([+-]?\d+(?:[.,]\d+)?)"
_UNIT = (
    r"(g\s*/\s*(?:L|l|mL|ml|cm(?:\^?3|³)|cc|cu\s*cm)|"
    r"kg\s*/\s*(?:L|l|m(?:\^?3|³)|dm(?:\^?3|³)))"
)
_TEMP = r"([+-]?\d+(?:[.,]\d+)?)\s*(?:°|deg(?:rees?)?\s*)?(C|F|K)\b"


def _temperature_kelvin(value: str, scale: str) -> float:
    numeric = float(value.replace(",", "."))
    if scale.upper() == "C":
        return numeric + 273.15
    if scale.upper() == "F":
        return (numeric - 32.0) * 5.0 / 9.0 + 273.15
    return numeric


def _density_g_cm3(value: str, unit: str) -> float:
    numeric = float(value.replace(",", "."))
    compact = re.sub(r"\s+", "", unit).lower()
    if compact in {"g/l", "kg/m3", "kg/m^3", "kg/m³"}:
        return numeric / 1000.0
    return numeric


def _has_nonambient_pressure(text: str) -> bool:
    ambient_pa = 101_325.0
    multipliers = {
        "pa": 1.0,
        "kpa": 1e3,
        "mpa": 1e6,
        "gpa": 1e9,
        "mbar": 1e2,
        "bar": 1e5,
        "atm": ambient_pa,
        "mmhg": ambient_pa / 760.0,
        "torr": ambient_pa / 760.0,
        "psi": 6_894.757293168,
    }
    for match in _PRESSURE.finditer(text):
        value = float(match.group(1).replace(",", "."))
        pressure_pa = value * multipliers[match.group(2).lower()]
        if not 0.8 * ambient_pa <= pressure_pa <= 1.2 * ambient_pa:
            return True
    return False


def _rejection_reason(contextual_text: str, text: str) -> str:
    if "[Binary attachment]" in text:
        return "binary_attachment"
    if _RANGE_OR_INEQUALITY.search(text):
        return "range_or_approximate_density"
    if _RATIO_TEMPERATURE.search(text):
        return "reference_temperature_density_ratio"
    if _NON_DENSITY_PROPERTY.search(contextual_text):
        return "not_absolute_density"
    if _MIXTURE_TERMS.search(contextual_text):
        return "mixture_density_evidence"
    if _INAPPROPRIATE_PHASE.search(contextual_text):
        return "inappropriate_phase"
    if _has_nonambient_pressure(text):
        return "nonambient_pressure"
    return ""


def _text_values(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        if "Binary" in node:
            yield "[Binary attachment]"
        if "StringWithMarkup" in node:
            value = node["StringWithMarkup"]
            if isinstance(value, list):
                parts = [str(item.get("String", "")) for item in value if isinstance(item, dict)]
                text = " ".join(part for part in parts if part)
                if text:
                    yield text
        if "Number" in node and isinstance(node["Number"], list):
            unit = str(node.get("Unit", ""))
            for number in node["Number"]:
                yield f"{number} {unit}".strip()
        for key, value in node.items():
            if key not in {"Binary", "StringWithMarkup", "Number"}:
                yield from _text_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _text_values(value)


def _reference_metadata(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in payload.get("Record", {}).get("Reference", []) or []:
        if not isinstance(item, dict):
            continue
        number = item.get("ReferenceNumber")
        if isinstance(number, int):
            source = str(item.get("SourceName", ""))
            citation = str(item.get("Citation", "") or item.get("Description", ""))
            verified = bool(item.get("DOI"))
            result[number] = {
                "source": source,
                "citation": citation,
                "peer_reviewed": bool(
                    verified or re.search(r"\bjournal\b|\b\d{4}\b", citation, re.I)
                ),
                "peer_reviewed_verified": verified,
            }
    return result


def _information_nodes(
    node: Any,
    headings: tuple[str, ...] = (),
) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
    if isinstance(node, dict):
        heading = str(node.get("TOCHeading", "")).strip()
        context = (*headings, heading) if heading else headings
        if isinstance(node.get("Information"), list):
            for item in node["Information"]:
                if isinstance(item, dict):
                    yield item, context
        for key, value in node.items():
            if key != "Information":
                yield from _information_nodes(value, context)
    elif isinstance(node, list):
        for value in node:
            yield from _information_nodes(value, headings)


def _unitless_pairs(text: str) -> tuple[list[tuple[float, float, bool]], str]:
    segments = [segment.strip() for segment in text.split(";") if segment.strip()]

    def segment_pair(segment: str) -> tuple[float, float, bool] | None:
        temperatures = list(re.finditer(_TEMP, segment, re.I))
        scrubbed = re.sub(_TEMP, " ", segment, flags=re.I)
        scrubbed = _PRESSURE.sub(" ", scrubbed)
        numbers = [
            float(value.replace(",", "."))
            for value in re.findall(_NUMBER, scrubbed)
            if 0.1 <= float(value.replace(",", ".")) <= 5.0
        ]
        if len(temperatures) != 1 or len(numbers) != 1:
            return None
        temperature = temperatures[0]
        return (
            numbers[0],
            _temperature_kelvin(temperature.group(1), temperature.group(2)),
            True,
        )

    if len(segments) > 1:
        paired = [segment_pair(segment) for segment in segments]
        if all(pair is not None for pair in paired):
            return [pair for pair in paired if pair is not None], ""

        if len(segments) % 2 == 0:
            alternating: list[tuple[float, float, bool]] = []
            for value_segment, temperature_segment in zip(
                segments[::2], segments[1::2]
            ):
                value_match = re.fullmatch(r"\s*" + _NUMBER + r"\s*", value_segment)
                temperature_match = re.fullmatch(
                    r"\s*" + _TEMP + r"\s*",
                    temperature_segment,
                    re.I,
                )
                if not value_match or not temperature_match:
                    alternating = []
                    break
                numeric = float(value_match.group(1).replace(",", "."))
                if not 0.1 <= numeric <= 5.0:
                    alternating = []
                    break
                alternating.append(
                    (
                        numeric,
                        _temperature_kelvin(
                            temperature_match.group(1),
                            temperature_match.group(2),
                        ),
                        True,
                    )
                )
            if alternating:
                return alternating, ""
        return [], "ambiguous_unitless_density"

    pair = segment_pair(text)
    if pair is not None:
        return [pair], ""
    if re.search(_TEMP, text, re.I) or re.search(_NUMBER, text):
        return [], "ambiguous_unitless_density"
    return [], "missing_density_value"


def parse_pubchem_density(
    payload: Mapping[str, Any],
    *,
    inchikey: str = "",
    cid: int | None = None,
) -> list[dict[str, Any]]:
    """Parse conservative pure-liquid density evidence from a PUG-View Density response."""
    references = _reference_metadata(payload)
    observations: list[dict[str, Any]] = []
    response_order = 0

    for information, headings in _information_nodes(payload.get("Record", payload)):
        reference_numbers = information.get("ReferenceNumber", [])
        if isinstance(reference_numbers, int):
            reference_numbers = [reference_numbers]
        reference = next(
            (references[number] for number in reference_numbers if number in references),
            {
                "source": "PubChem",
                "citation": "",
                "peer_reviewed": False,
                "peer_reviewed_verified": False,
            },
        )
        for raw_text in _text_values(information.get("Value", information)):
            text = " ".join(raw_text.split())
            contextual_text = " ".join((*headings, text))
            rejection_reason = _rejection_reason(contextual_text, text)
            pairs: list[tuple[float, float, bool]] = []

            # Explicit density followed by a bound temperature.
            explicit = re.compile(
                rf"{_NUMBER}\s*{_UNIT}\s*(?:at|@|,|;|\(\s*(?:at\s*)?)\s*{_TEMP}",
                re.I,
            )
            for match in explicit.finditer(text):
                pairs.append(
                    (
                        _density_g_cm3(match.group(1), match.group(2)),
                        _temperature_kelvin(match.group(3), match.group(4)),
                        False,
                    )
                )

            # "Density (at 20 C): 0.8 g/mL" and similar reversed forms.
            reversed_explicit = re.compile(
                rf"density\s*(?:\(\s*)?(?:at|@)?\s*{_TEMP}(?:\s*\))?\s*[:;,=-]+\s*{_NUMBER}\s*{_UNIT}",
                re.I,
            )
            for match in reversed_explicit.finditer(text):
                pairs.append(
                    (
                        _density_g_cm3(match.group(3), match.group(4)),
                        _temperature_kelvin(match.group(1), match.group(2)),
                        False,
                    )
                )

            if pairs and not rejection_reason and ";" in text:
                for segment in (item.strip() for item in text.split(";") if item.strip()):
                    if explicit.search(segment) or reversed_explicit.search(segment):
                        continue
                    scrubbed = re.sub(_TEMP, " ", segment, flags=re.I)
                    scrubbed = _PRESSURE.sub(" ", scrubbed)
                    plausible_numbers = [
                        float(value.replace(",", "."))
                        for value in re.findall(_NUMBER, scrubbed)
                        if 0.1 <= float(value.replace(",", ".")) <= 5.0
                    ]
                    if plausible_numbers or re.search(_TEMP, segment, re.I):
                        rejection_reason = "ambiguous_semicolon_density"
                        break

            if not pairs and not rejection_reason:
                pairs, rejection_reason = _unitless_pairs(text)

            if not pairs:
                observations.append(
                    {
                        "inchikey": inchikey,
                        "cid": cid,
                        "temperature_k": None,
                        "density_g_cm3": None,
                        "source_type": "pubchem",
                        "source": reference["source"],
                        "citation": reference["citation"],
                        "raw_evidence": text,
                        "eligible": False,
                        "failure_reason": rejection_reason or "ineligible_density_evidence",
                        "unit_inferred_from_section": False,
                        "unit_explicit": False,
                        "peer_reviewed": reference["peer_reviewed"],
                        "peer_reviewed_verified": reference[
                            "peer_reviewed_verified"
                        ],
                        "response_order": response_order,
                    }
                )
                response_order += 1
                continue

            for density, temperature, inferred in pairs:
                plausible = 0 < density < 25 and temperature > 0
                failure_reason = rejection_reason
                if not plausible:
                    failure_reason = "implausible_density_value"
                observations.append(
                    {
                        "inchikey": inchikey,
                        "cid": cid,
                        "temperature_k": temperature,
                        "density_g_cm3": density,
                        "source_type": "pubchem",
                        "source": reference["source"],
                        "citation": reference["citation"],
                        "raw_evidence": text,
                        "eligible": bool(plausible and not rejection_reason),
                        "failure_reason": failure_reason,
                        "unit_inferred_from_section": inferred,
                        "unit_explicit": not inferred,
                        "peer_reviewed": reference["peer_reviewed"],
                        "peer_reviewed_verified": reference[
                            "peer_reviewed_verified"
                        ],
                        "response_order": response_order,
                    }
                )
                response_order += 1
    return observations


class PubChemDensityError(RuntimeError):
    pass


class PubChemRecordMissing(PubChemDensityError):
    pass


class PubChemPermanentError(PubChemDensityError):
    pass


class PubChemTransientError(PubChemDensityError):
    pass


@dataclass
class PubChemDensityClient:
    session: requests.Session
    retries: int = 3
    timeout: float = 30.0

    def _get_json(self, url: str) -> dict[str, Any]:
        last_error = ""
        attempts = max(1, self.retries)
        for attempt in range(attempts):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"transport error: {exc}"
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
                    continue
                raise PubChemTransientError(
                    f"PubChem request failed after {attempts} attempts: {last_error}"
                ) from exc

            status = int(response.status_code)
            if status == 404:
                raise PubChemRecordMissing(f"PubChem record not found: {url}")
            if status == 429 or 500 <= status <= 599:
                last_error = f"transient HTTP status {status}"
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
                    continue
                raise PubChemTransientError(
                    f"PubChem request failed after {attempts} attempts: {last_error}"
                )
            if 400 <= status <= 499:
                raise PubChemPermanentError(
                    f"Permanent PubChem HTTP status {status}: {url}"
                )
            if not 200 <= status <= 299:
                raise PubChemPermanentError(
                    f"Unexpected PubChem HTTP status {status}: {url}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise PubChemPermanentError(
                    "PubChem returned invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise PubChemPermanentError(
                    "PubChem returned a non-object JSON payload"
                )
            time.sleep(PUBCHEM_DELAY_SECONDS)
            return payload
        raise PubChemTransientError(f"PubChem request failed: {last_error}")

    def resolve_cid(self, inchikey: str) -> int:
        payload = self._get_json(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
            f"{quote(inchikey, safe='')}/cids/JSON"
        )
        cids = payload.get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise PubChemRecordMissing(f"No PubChem CID for {inchikey}")
        return int(cids[0])

    def fetch_density(self, cid: int, inchikey: str) -> list[dict[str, Any]]:
        payload = self._get_json(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
            f"{cid}/JSON?heading=Density"
        )
        return parse_pubchem_density(payload, inchikey=inchikey, cid=cid)
