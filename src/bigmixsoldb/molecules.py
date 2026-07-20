from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, rdBase

from bigmixsoldb.constants import CAS_PATTERN, GENERIC_NAME_PATTERNS
from bigmixsoldb.doi import normalize_doi
from bigmixsoldb.yaml_utils import load_yaml_document


def is_generic_name(name: str) -> bool:
    text = name.strip()
    if len(text) <= 2:
        return True
    return any(pattern.match(text) for pattern in GENERIC_NAME_PATTERNS)


def is_cas_number(name: str) -> tuple[bool, str]:
    match = CAS_PATTERN.search(name.strip())
    if match:
        return True, match.group(1)
    return False, ""


def order_molecule_group(names: list[str]) -> list[str]:
    cas_numbers: list[str] = []
    other_names: list[str] = []
    for name in names:
        is_cas, cas_value = is_cas_number(name)
        if is_cas:
            cas_numbers.append(cas_value)
        else:
            other_names.append(name)
    return cas_numbers + other_names


def _normalized_synonyms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split("|")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item and not is_generic_name(item)]


def extract_molecule_placeholders(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped_solutes: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    grouped_solvents: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    for path in paths:
        try:
            document, _ = load_yaml_document(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        source = path.stem
        for item in document:
            compound = str(item.get("compound", "")).strip()
            if compound and not is_generic_name(compound):
                solute_group = order_molecule_group([compound, *_normalized_synonyms(item.get("synonyms"))])
                if solute_group:
                    key = (solute_group[0], tuple(sorted(set(solute_group[1:]))))
                    entry = grouped_solutes.setdefault(
                        key,
                        {
                            "name": solute_group[0],
                            "synonyms": list(dict.fromkeys(solute_group[1:])),
                            "smiles": "",
                            "sources": [],
                            "molecule_type": "solute",
                            "enabled": True,
                        },
                    )
                    if source not in entry["sources"]:
                        entry["sources"].append(source)

            for record in item.get("entries", []):
                if not isinstance(record, dict):
                    continue
                solv = str(record.get("solv", "")).strip()
                if not solv:
                    continue
                for solvent_name in [part.strip() for part in solv.split("|") if part.strip()]:
                    if is_generic_name(solvent_name):
                        continue
                    key = (solvent_name, tuple())
                    entry = grouped_solvents.setdefault(
                        key,
                        {
                            "name": solvent_name,
                            "synonyms": [],
                            "smiles": "",
                            "sources": [],
                            "molecule_type": "solvent",
                            "enabled": True,
                        },
                    )
                    if source not in entry["sources"]:
                        entry["sources"].append(source)

    return {
        "solutes": sorted(grouped_solutes.values(), key=lambda entry: entry["name"].lower()),
        "solvents": sorted(grouped_solvents.values(), key=lambda entry: entry["name"].lower()),
    }


def write_molecule_placeholders(paths: list[Path], output_path: str | Path) -> Path:
    content = extract_molecule_placeholders(paths)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_file


@dataclass(frozen=True, slots=True)
class MoleculeRecord:
    smiles: str
    enabled: bool = True
    ambiguous: bool = False


def _molecule_lookup_key(name: str) -> str:
    return name.strip().lower()


MoleculeLookupKey = str | tuple[str, str] | tuple[str, str, str]


def _doi_aliases(value: Any) -> list[str]:
    text = str(value or "").strip()
    lowered = text.lower()
    for suffix in (".yaml", ".yml", ".pdf"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            break

    normalized = normalize_doi(text)
    if not normalized:
        return []

    # A DOI that already has its registrant separator can contain legitimate
    # underscores in the suffix. Only treat underscores as filename-encoded
    # slashes when that required separator is absent (for example,
    # ``10.1000_article`` from ``10.1000/article.yml``).
    _, separator, _ = normalized.partition("/")
    if separator or "_" not in normalized:
        return [normalized]
    registrant = normalized.split("_", 1)[0]
    if registrant.startswith("10.") and registrant[3:].isdigit():
        return [normalize_doi(normalized.replace("_", "/"))]
    return [normalized]


def normalize_molecule_dois(value: Any) -> list[str]:
    """Normalize DOI provenance while accepting DOI and filename-style sources."""
    if value is None:
        return []
    raw_values = value if isinstance(value, (list, tuple, set)) else [value]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for part in str(raw_value).split(";"):
            for doi in _doi_aliases(part):
                if doi not in seen:
                    seen.add(doi)
                    normalized.append(doi)
    return normalized


@lru_cache(maxsize=None)
def _molecule_identity(smiles: str) -> tuple[str, str] | None:
    text = smiles.strip()
    if not text:
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return None
        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        inchikey = Chem.MolToInchiKey(molecule)
    if inchikey:
        return "inchikey", inchikey
    return ("canonical_smiles", canonical_smiles) if canonical_smiles else None


def combine_molecule_records(
    records: Iterable[MoleculeRecord],
) -> MoleculeRecord | None:
    """Collapse equivalent candidates and return a blank ambiguity sentinel on conflict."""
    combined: MoleculeRecord | None = None
    for record in records:
        if combined is None:
            combined = record
            continue
        if combined.ambiguous or record.ambiguous:
            combined = MoleculeRecord(smiles="", enabled=True, ambiguous=True)
            continue

        combined_identity = _molecule_identity(combined.smiles)
        record_identity = _molecule_identity(record.smiles)
        same_structure = (
            combined_identity is not None
            and record_identity is not None
            and combined_identity == record_identity
        )
        if not same_structure and combined.smiles.strip() == record.smiles.strip():
            same_structure = True

        if not same_structure:
            combined = MoleculeRecord(smiles="", enabled=True, ambiguous=True)
            continue

        combined = MoleculeRecord(
            smiles=combined.smiles or record.smiles,
            enabled=combined.enabled and record.enabled,
        )
    return combined


def load_molecule_lookup(path: str | Path | None) -> dict[MoleculeLookupKey, MoleculeRecord]:
    if path is None:
        return {}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: dict[MoleculeLookupKey, MoleculeRecord] = {}

    def register_record(key: MoleculeLookupKey, record: MoleculeRecord) -> None:
        existing = lookup.get(key)
        combined = combine_molecule_records(
            [candidate for candidate in (existing, record) if candidate is not None]
        )
        if combined is not None:
            lookup[key] = combined

    def register_entry(entry: dict[str, Any], group: str | None = None) -> None:
        smiles = str(entry.get("smiles", "")).strip()
        enabled = bool(entry.get("enabled", True))
        molecule_type = str(entry.get("molecule_type") or group or "").strip().lower()
        if molecule_type.endswith("s"):
            molecule_type = molecule_type[:-1]
        if molecule_type not in {"solute", "solvent"}:
            molecule_type = ""

        names: list[str] = []
        primary_name = entry.get("name") or entry.get("primary_name") or entry.get("compound")
        if primary_name:
            names.append(str(primary_name).strip())

        synonyms = entry.get("synonyms", [])
        if isinstance(synonyms, str):
            names.extend(part.strip() for part in synonyms.split("|") if part.strip())
        elif isinstance(synonyms, list):
            names.extend(str(part).strip() for part in synonyms if str(part).strip())

        source_backed = "sources" in entry
        source_dois = normalize_molecule_dois(entry.get("sources")) if source_backed else []
        record = MoleculeRecord(smiles=smiles, enabled=enabled)
        for name in names:
            key = _molecule_lookup_key(name)
            if not key:
                continue
            if source_backed:
                for doi in source_dois:
                    register_record((molecule_type, key, doi), record)
            else:
                register_record((molecule_type, key) if molecule_type else key, record)

    if isinstance(payload, dict):
        if all(isinstance(value, str) for value in payload.values()):
            for name, smiles in payload.items():
                key = _molecule_lookup_key(name)
                if key:
                    register_record(key, MoleculeRecord(smiles=smiles.strip()))
            return lookup

        for key in ("name_to_smiles", "molecules"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                for name, smiles in candidate.items():
                    lookup_key = _molecule_lookup_key(name)
                    if isinstance(smiles, str) and lookup_key:
                        register_record(
                            lookup_key,
                            MoleculeRecord(smiles=smiles.strip()),
                        )

        for key in ("solutes", "solvents", "molecules"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                for entry in candidate:
                    if isinstance(entry, dict):
                        register_entry(entry, key)
        return lookup

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                register_entry(entry)

    return lookup


def load_name_to_smiles_map(path: str | Path | None) -> dict[str, str]:
    """Return only unambiguous, source-less legacy mappings.

    DOI-scoped records are intentionally omitted because flattening them back to
    a global name map would reintroduce cross-source structure inheritance.
    """
    records_by_name: dict[str, MoleculeRecord] = {}
    for key, record in load_molecule_lookup(path).items():
        if isinstance(key, tuple) and len(key) == 3:
            continue
        name = key[1] if isinstance(key, tuple) else key
        combined = combine_molecule_records(
            [
                candidate
                for candidate in (records_by_name.get(name), record)
                if candidate is not None
            ]
        )
        if combined is not None:
            records_by_name[name] = combined
    return {
        name: record.smiles
        for name, record in records_by_name.items()
        if record.enabled and record.smiles and not record.ambiguous
    }
