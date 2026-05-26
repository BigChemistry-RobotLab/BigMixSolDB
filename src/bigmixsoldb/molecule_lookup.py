from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from rdkit import Chem
from requests import Session
from tqdm import tqdm

CHEMSCRIPT_AVAILABLE = find_spec("ChemScript") is not None

if CHEMSCRIPT_AVAILABLE:
    import ChemScript as _ChemScript  # type: ignore[import]
else:
    _ChemScript = None

LATEX_REPLACEMENTS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\omicron": "omicron",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}

SAFE_ADDITIVE_FRAGMENTS = {
    "trihydrochloride": ("Cl", "Cl", "Cl"),
    "dihydrochloride": ("Cl", "Cl"),
    "monohydrochloride": ("Cl",),
    "hydrochloride": ("Cl",),
    "trihydrate": ("O", "O", "O"),
    "dihydrate": ("O", "O"),
    "monohydrate": ("O",),
    "hydrate": ("O",),
}

SAFE_SOLVATE_FRAGMENTS = {
    "water-ethanol": ("O", "CCO"),
    "water": ("O",),
    "methanol": ("CO",),
    "ethanol": ("CCO",),
    "isopropanol": ("CC(C)O",),
    "acetonitrile": ("CC#N",),
    "dimethyl sulfoxide": ("CS(C)=O",),
    "dimethylsulfoxide": ("CS(C)=O",),
    "dimethyl formamide": ("CN(C)C=O",),
    "dimethylformamide": ("CN(C)C=O",),
}

NAME_ALIASES = {
    "i-butanol": ["isobutanol", "2-methyl-1-propanol"],
    "i-propanol": ["isopropanol", "2-propanol"],
    "s-butanol": ["sec-butanol", "2-butanol"],
    "n-butanol": ["1-butanol"],
    "n-pentanol": ["1-pentanol"],
    "n-propanol": ["1-propanol"],
    "t-butanol": ["tert-butanol", "2-methyl-2-propanol"],
    "ethyl ethanoate": ["ethyl acetate"],
    "methyl ethanoate": ["methyl acetate"],
    "propyl ethanoate": ["propyl acetate"],
    "butyl ethanoate": ["butyl acetate"],
    "trichloromethane": ["chloroform"],
    "tetrachloromethane": ["carbon tetrachloride"],
    "propanone": ["acetone"],
}

FORM_SUFFIX_PATTERN = re.compile(r"(?:,?\s+form\s+[a-z0-9-]+)$", re.IGNORECASE)
BRACED_TEXT_PATTERN = re.compile(r"\{([^{}]+)\}")
WHITESPACE_PATTERN = re.compile(r"\s+")

LOOKUP_METADATA_FIELDS = (
    "smiles_source",
    "smiles_query",
    "smiles_match_name",
    "smiles_exact_match",
    "smiles_url",
    "smiles_note",
    "smiles_status",
    "retrieval_metadata",
)

RETRIEVAL_METADATA_FIELDS = (
    "smiles_source",
    "smiles_query",
    "smiles_status",
    "smiles_match_name",
    "smiles_exact_match",
    "smiles_url",
    "smiles_note",
    "retrieved_smiles",
    "human_validated_smiles",
    "human_validation_method",
)


@dataclass(frozen=True, slots=True)
class LookupCandidate:
    query: str
    additives: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True, slots=True)
class LookupResult:
    smiles: str
    source: str
    query: str
    matched_name: str | None = None
    exact_match: bool | None = None
    url: str | None = None
    note: str | None = None


def canonicalize_smiles(smiles: str) -> str | None:
    text = smiles.strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _clear_lookup_metadata(entry: dict[str, Any]) -> None:
    for field in LOOKUP_METADATA_FIELDS:
        entry.pop(field, None)


def _molecule_type_from_group(group: str, entry: dict[str, Any]) -> str:
    existing = str(entry.get("molecule_type", "")).strip()
    if existing:
        return existing
    if group == "solutes":
        return "solute"
    if group == "solvents":
        return "solvent"
    return "molecule"


def _metadata_payload(
    *,
    smiles_source: str = "",
    smiles_query: str = "",
    smiles_status: str = "",
    smiles_match_name: str = "",
    smiles_exact_match: bool | str = "",
    smiles_url: str = "",
    smiles_note: str = "",
    retrieved_smiles: str = "",
    human_validated_smiles: str = "",
    human_validation_method: str = "",
) -> dict[str, Any]:
    return {
        "smiles_source": smiles_source,
        "smiles_query": smiles_query,
        "smiles_status": smiles_status,
        "smiles_match_name": smiles_match_name,
        "smiles_exact_match": smiles_exact_match,
        "smiles_url": smiles_url,
        "smiles_note": smiles_note,
        "retrieved_smiles": retrieved_smiles,
        "human_validated_smiles": human_validated_smiles,
        "human_validation_method": human_validation_method,
    }


def _existing_retrieval_metadata(entry: dict[str, Any], existing_smiles: str) -> dict[str, Any]:
    metadata = entry.get("retrieval_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    merged = {
        "smiles_source": metadata.get("smiles_source", entry.get("smiles_source", "")),
        "smiles_query": metadata.get("smiles_query", entry.get("smiles_query", "")),
        "smiles_status": metadata.get("smiles_status", entry.get("smiles_status", "existing")),
        "smiles_match_name": metadata.get("smiles_match_name", entry.get("smiles_match_name", "")),
        "smiles_exact_match": metadata.get("smiles_exact_match", entry.get("smiles_exact_match", "")),
        "smiles_url": metadata.get("smiles_url", entry.get("smiles_url", "")),
        "smiles_note": metadata.get("smiles_note", entry.get("smiles_note", "")),
        "retrieved_smiles": metadata.get("retrieved_smiles", entry.get("retrieved_smiles", existing_smiles)),
        "human_validated_smiles": metadata.get(
            "human_validated_smiles",
            entry.get("human_validated_smiles", ""),
        ),
        "human_validation_method": metadata.get(
            "human_validation_method",
            entry.get("human_validation_method", ""),
        ),
    }
    return {field: merged.get(field, "") for field in RETRIEVAL_METADATA_FIELDS}


def _shape_resolved_entry(
    entry: dict[str, Any],
    *,
    group: str,
    smiles: str,
    metadata: dict[str, Any],
) -> None:
    name = str(entry.get("name", "")).strip()
    synonyms = _extract_synonyms(entry.get("synonyms"))
    sources = entry.get("sources", [])
    if not isinstance(sources, list):
        sources = [str(sources)] if sources else []

    molecule_type = _molecule_type_from_group(group, entry)
    enabled = entry.get("enabled", True)

    entry.clear()
    entry.update(
        {
            "name": name,
            "synonyms": synonyms,
            "smiles": smiles,
            "sources": sources,
            "molecule_type": molecule_type,
            "enabled": bool(enabled),
            "retrieval_metadata": {field: metadata.get(field, "") for field in RETRIEVAL_METADATA_FIELDS},
        }
    )


def _dedupe_preserve_order(values: list[LookupCandidate]) -> list[LookupCandidate]:
    deduped: list[LookupCandidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for value in values:
        key = (value.query.lower(), value.additives)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_query_text(text: str) -> str:
    normalized = text.strip()
    for source, target in LATEX_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")
    normalized = normalized.replace("\\", "")
    normalized = normalized.replace("{", " ")
    normalized = normalized.replace("}", " ")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip(" ,;:-")


def _apply_name_aliases(name: str) -> list[str]:
    lowered = name.lower()
    aliases = list(NAME_ALIASES.get(lowered, []))
    if lowered.endswith(" ethanoate"):
        aliases.append(re.sub(r"ethanoate$", "acetate", lowered))
    return [alias.strip() for alias in aliases if alias.strip()]


def _remove_case_insensitive(text: str, fragment: str) -> str:
    return re.sub(re.escape(fragment), "", text, flags=re.IGNORECASE)


def _strip_additives_for_lookup(name: str) -> tuple[str, tuple[str, ...]]:
    cleaned = name
    additives: list[str] = []
    lowered = cleaned.lower()

    for token, fragments in sorted(SAFE_ADDITIVE_FRAGMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        if token in lowered:
            cleaned = _remove_case_insensitive(cleaned, token)
            lowered = cleaned.lower()
            additives.extend(fragments)

    if "solvate" in lowered:
        cleaned = re.sub(r"\bsolvate\b", "", cleaned, flags=re.IGNORECASE)
        lowered = cleaned.lower()
        for token, fragments in sorted(SAFE_SOLVATE_FRAGMENTS.items(), key=lambda item: len(item[0]), reverse=True):
            if token in lowered:
                cleaned = _remove_case_insensitive(cleaned, token)
                lowered = cleaned.lower()
                additives.extend(fragments)

    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip(" ,;:-")
    return cleaned, tuple(additives)


def _extract_synonyms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def build_lookup_candidates(entry: dict[str, Any]) -> list[LookupCandidate]:
    raw_names = [str(entry.get("name", "")).strip(), *_extract_synonyms(entry.get("synonyms"))]
    candidates: list[LookupCandidate] = []

    def add_candidate(query: str, additives: tuple[str, ...] = (), note: str | None = None) -> None:
        normalized = _normalize_query_text(query)
        if normalized:
            candidates.append(LookupCandidate(normalized, additives, note))

    for raw_name in raw_names:
        if not raw_name:
            continue

        brace_contents = BRACED_TEXT_PATTERN.findall(raw_name)
        brace_removed = BRACED_TEXT_PATTERN.sub("", raw_name).strip()
        if brace_contents and brace_removed:
            add_candidate(brace_removed, note="without-braced-text")
        else:
            add_candidate(raw_name)

        for brace_content in brace_contents:
            add_candidate(brace_content, note="braced-text")

        normalized = _normalize_query_text(raw_name)
        form_stripped = FORM_SUFFIX_PATTERN.sub("", normalized).strip(" ,;:-")
        if form_stripped and form_stripped != normalized:
            add_candidate(form_stripped, note="without-form")

        for alias in _apply_name_aliases(normalized):
            add_candidate(alias, note="alias")
        for alias in _apply_name_aliases(form_stripped):
            add_candidate(alias, note="alias-without-form")

        additive_stripped, additives = _strip_additives_for_lookup(form_stripped)
        if additive_stripped and additive_stripped != form_stripped:
            add_candidate(additive_stripped, additives=additives, note="without-additives")
        for alias in _apply_name_aliases(additive_stripped):
            add_candidate(alias, additives=additives, note="alias-without-additives")

    return _dedupe_preserve_order(candidates)


def _combine_smiles(base_smiles: str, additives: tuple[str, ...]) -> str | None:
    if not additives:
        return base_smiles
    combined = ".".join([base_smiles, *additives])
    return canonicalize_smiles(combined) or combined


def _resolve_with_chemscript(
    candidate: LookupCandidate,
    cache: dict[str, str | None],
) -> LookupResult | None:
    if not CHEMSCRIPT_AVAILABLE or _ChemScript is None:
        return None

    key = candidate.query.lower()
    if key not in cache:
        structure = _ChemScript.StructureData()
        try:
            structure.ReadData(candidate.query, "name")
        except Exception:
            cache[key] = None
        else:
            if getattr(structure, "IsEmpty", True):
                cache[key] = None
            else:
                smiles_data = structure.WriteData("smiles")
                if isinstance(smiles_data, bytes):
                    smiles_data = smiles_data.decode("utf-8")
                cache[key] = canonicalize_smiles(str(smiles_data))

    base_smiles = cache.get(key)
    if not base_smiles:
        return None

    combined = _combine_smiles(base_smiles, candidate.additives)
    if not combined:
        return None
    return LookupResult(
        smiles=combined,
        source="chemscript",
        query=candidate.query,
        note=candidate.note,
    )


def _lookup_pubchem_name(
    query: str,
    session: Session,
    *,
    timeout: int,
) -> tuple[str, str | None] | None:
    def get_with_retry(url: str) -> requests.Response:
        last_response: requests.Response | None = None
        for attempt in range(3):
            response = session.get(url, timeout=timeout)
            last_response = response
            if response.status_code not in {429, 503}:
                return response
            if attempt < 2:
                time.sleep(1.0 + attempt)
        assert last_response is not None
        return last_response

    encoded = quote(query)
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/JSON"
    response = get_with_retry(cid_url)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    cids = response.json().get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None

    cid = cids[0]
    property_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"
    )
    property_response = get_with_retry(property_url)
    if property_response.status_code == 404:
        return None
    property_response.raise_for_status()

    properties = property_response.json().get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return None

    property_record = properties[0]
    smiles = (
        property_record.get("IsomericSMILES")
        or property_record.get("CanonicalSMILES")
        or property_record.get("SMILES")
        or property_record.get("ConnectivitySMILES")
    )
    canonical = canonicalize_smiles(str(smiles or ""))
    if not canonical:
        return None

    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    return canonical, pubchem_url


def _resolve_with_pubchem(
    candidate: LookupCandidate,
    session: Session,
    cache: dict[str, LookupResult | None],
    *,
    timeout: int,
    autocomplete_limit: int,
) -> LookupResult | None:
    key = candidate.query.lower()
    if key not in cache:
        direct_match = _lookup_pubchem_name(candidate.query, session, timeout=timeout)
        if direct_match is not None:
            cache[key] = LookupResult(
                smiles=direct_match[0],
                source="pubchem",
                query=candidate.query,
                matched_name=candidate.query,
                exact_match=True,
                url=direct_match[1],
            )
        else:
            encoded = quote(candidate.query)
            autocomplete_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/"
                f"{encoded}/json?limit={autocomplete_limit}"
            )
            response = session.get(autocomplete_url, timeout=timeout)
            if response.status_code == 404:
                cache[key] = None
            else:
                response.raise_for_status()
                names = response.json().get("dictionary_terms", {}).get("compound", [])
                resolved: LookupResult | None = None
                for matched_name in names[:autocomplete_limit]:
                    matched = _lookup_pubchem_name(matched_name, session, timeout=timeout)
                    if matched is None:
                        continue
                    resolved = LookupResult(
                        smiles=matched[0],
                        source="pubchem",
                        query=candidate.query,
                        matched_name=matched_name,
                        exact_match=matched_name.lower() == candidate.query.lower(),
                        url=matched[1],
                    )
                    break
                cache[key] = resolved

    base_result = cache.get(key)
    if base_result is None:
        return None

    combined = _combine_smiles(base_result.smiles, candidate.additives)
    if not combined:
        return None
    return LookupResult(
        smiles=combined,
        source=base_result.source,
        query=base_result.query,
        matched_name=base_result.matched_name,
        exact_match=base_result.exact_match,
        url=base_result.url,
        note=candidate.note,
    )


def resolve_entry_smiles(
    entry: dict[str, Any],
    session: Session,
    *,
    pubchem_timeout: int = 15,
    pubchem_limit: int = 5,
    chemscript_cache: dict[str, str | None] | None = None,
    pubchem_cache: dict[str, LookupResult | None] | None = None,
) -> LookupResult | None:
    candidates = build_lookup_candidates(entry)
    if not candidates:
        return None

    if chemscript_cache is None:
        chemscript_cache = {}
    if pubchem_cache is None:
        pubchem_cache = {}

    if CHEMSCRIPT_AVAILABLE:
        for candidate in candidates:
            result = _resolve_with_chemscript(candidate, chemscript_cache)
            if result is not None:
                return result

    for candidate in candidates:
        try:
            result = _resolve_with_pubchem(
                candidate,
                session,
                pubchem_cache,
                timeout=pubchem_timeout,
                autocomplete_limit=pubchem_limit,
            )
        except requests.RequestException:
            continue
        if result is not None:
            return result
    return None


def _iter_payload_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, list):
        return [("molecules", entry) for entry in payload if isinstance(entry, dict)]

    if not isinstance(payload, dict):
        raise ValueError("Expected a molecule JSON payload containing a dict or list of molecule entries")

    entries: list[tuple[str, dict[str, Any]]] = []
    for key in ("solutes", "solvents", "molecules"):
        group = payload.get(key)
        if isinstance(group, list):
            entries.extend((key, entry) for entry in group if isinstance(entry, dict))

    if not entries:
        raise ValueError("No molecule entries were found in the provided JSON payload")

    return entries


def resolve_molecule_json_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    replace_existing: bool = False,
    limit: int | None = None,
    pubchem_timeout: int = 15,
    pubchem_limit: int = 5,
) -> dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    entries = _iter_payload_entries(payload)
    if limit is not None:
        entries = entries[:limit]

    summary: dict[str, Any] = {
        "total": len(entries),
        "resolved": 0,
        "skipped": 0,
        "unresolved": 0,
        "chemscript": 0,
        "pubchem": 0,
        "chemscript_available": CHEMSCRIPT_AVAILABLE,
    }

    chemscript_cache: dict[str, str | None] = {}
    pubchem_cache: dict[str, LookupResult | None] = {}

    with requests.Session() as session:
        for group, entry in tqdm(entries, desc="Resolving molecule SMILES"):
            existing_smiles = str(entry.get("smiles", "")).strip()
            if existing_smiles and not replace_existing:
                summary["skipped"] += 1
                metadata = _existing_retrieval_metadata(entry, existing_smiles)
                _clear_lookup_metadata(entry)
                _shape_resolved_entry(
                    entry,
                    group=group,
                    smiles=existing_smiles,
                    metadata=metadata,
                )
                continue

            _clear_lookup_metadata(entry)
            result = resolve_entry_smiles(
                entry,
                session,
                pubchem_timeout=pubchem_timeout,
                pubchem_limit=pubchem_limit,
                chemscript_cache=chemscript_cache,
                pubchem_cache=pubchem_cache,
            )

            if result is None:
                _shape_resolved_entry(
                    entry,
                    group=group,
                    smiles="",
                    metadata=_metadata_payload(
                        smiles_status="unresolved",
                        smiles_note="ChemScript and PubChem lookup failed",
                    ),
                )
                summary["unresolved"] += 1
                continue

            _shape_resolved_entry(
                entry,
                group=group,
                smiles=result.smiles,
                metadata=_metadata_payload(
                    smiles_source=result.source,
                    smiles_query=result.query,
                    smiles_status="resolved",
                    smiles_match_name=result.matched_name or "",
                    smiles_exact_match=result.exact_match if result.exact_match is not None else "",
                    smiles_url=result.url or "",
                    smiles_note=result.note or "",
                    retrieved_smiles=result.smiles,
                    human_validated_smiles="",
                    human_validation_method="",
                ),
            )

            summary["resolved"] += 1
            summary[result.source] += 1

    resolved_entries = [entry for _, entry in entries]

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(resolved_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
