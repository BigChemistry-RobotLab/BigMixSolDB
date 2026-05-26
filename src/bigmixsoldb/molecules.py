from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigmixsoldb.constants import CAS_PATTERN, GENERIC_NAME_PATTERNS
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


def _molecule_lookup_key(name: str) -> str:
    return name.strip().lower()


def load_molecule_lookup(path: str | Path | None) -> dict[str, MoleculeRecord]:
    if path is None:
        return {}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: dict[str, MoleculeRecord] = {}

    def register_entry(entry: dict[str, Any]) -> None:
        smiles = str(entry.get("smiles", "")).strip()
        enabled = bool(entry.get("enabled", True))

        names: list[str] = []
        primary_name = entry.get("name") or entry.get("primary_name") or entry.get("compound")
        if primary_name:
            names.append(str(primary_name).strip())

        synonyms = entry.get("synonyms", [])
        if isinstance(synonyms, str):
            names.extend(part.strip() for part in synonyms.split("|") if part.strip())
        elif isinstance(synonyms, list):
            names.extend(str(part).strip() for part in synonyms if str(part).strip())

        for name in names:
            key = _molecule_lookup_key(name)
            if key:
                lookup[key] = MoleculeRecord(smiles=smiles, enabled=enabled)

    if isinstance(payload, dict):
        if all(isinstance(value, str) for value in payload.values()):
            for name, smiles in payload.items():
                key = _molecule_lookup_key(name)
                if key:
                    lookup[key] = MoleculeRecord(smiles=smiles.strip())
            return lookup

        for key in ("name_to_smiles", "molecules"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                for name, smiles in candidate.items():
                    lookup_key = _molecule_lookup_key(name)
                    if isinstance(smiles, str) and lookup_key:
                        lookup[lookup_key] = MoleculeRecord(smiles=smiles.strip())

        for key in ("solutes", "solvents", "molecules"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                for entry in candidate:
                    if isinstance(entry, dict):
                        register_entry(entry)
        return lookup

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                register_entry(entry)

    return lookup


def load_name_to_smiles_map(path: str | Path | None) -> dict[str, str]:
    return {
        name: record.smiles
        for name, record in load_molecule_lookup(path).items()
        if record.enabled and record.smiles
    }
