from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from rdkit import Chem


SOLVENT_COLUMNS = (
    ("Solvent 1", "SMILES_Solvent_1"),
    ("Solvent 2", "SMILES_Solvent_2"),
    ("Solvent 3", "SMILES_Solvent_3"),
)


CURATED_SOLVENT_NAMES = {
    "O": "Water",
    "CCO": "Ethanol",
    "CO": "Methanol",
    "CC(C)O": "Isopropanol",
    "CCOC(C)=O": "Ethyl acetate",
    "CC(C)=O": "Acetone",
    "CCCO": "n-Propanol",
    "CC#N": "Acetonitrile",
    "CN(C)C=O": "N,N-Dimethylformamide",
    "CCCCO": "n-Butanol",
    "C1COCCO1": "1,4-Dioxane",
    "CN1CCCC1=O": "N-Methyl-2-pyrrolidone",
    "CCCCCC": "n-Hexane",
    "C1CCOC1": "Tetrahydrofuran",
    "C1CCCCC1": "Cyclohexane",
    "Cc1ccccc1": "Toluene",
    "CS(C)=O": "Dimethyl sulfoxide",
    "OCCO": "Ethylene glycol",
    "CC(O)CO": "Propylene glycol",
    "CC(C)CO": "Isobutanol",
    "CCCCCCC": "Heptane",
    "CC(=O)O": "Acetic acid",
    "COC(C)=O": "Methyl acetate",
    "ClC(Cl)Cl": "Chloroform",
    "CCC(C)O": "2-Butanol",
    "CCCCCCCCO": "n-Octanol",
    "CCC(C)=O": "2-Butanone",
    "CCCCCO": "n-Pentanol",
    "CC(=O)N(C)C": "N,N-Dimethylacetamide",
    "ClCCl": "Dichloromethane",
    "O=C1CCCCC1": "Cyclohexanone",
    "CCCCOC(C)=O": "Butyl acetate",
    "O=C=O": "Carbon dioxide",
    "CCOCCOCCO": "Diethylene glycol monoethyl ether",
    "COCCO": "2-Methoxyethanol",
    "c1ccccc1": "Benzene",
    "CCCOC(C)=O": "n-Propyl acetate",
    "CC(C)CCO": "3-Methyl-1-butanol",
    "ClCCCl": "1,2-Dichloroethane",
    "CCC(=O)O": "Propionic acid",
    "CC(C)CC(C)(C)C": "2,2,4-Trimethylpentane",
    "CCOCCO": "2-Ethoxyethanol",
    "COC(C)(C)C": "Methyl tert-butyl ether",
    "Cc1ccccc1C": "o-Xylene",
    "CCCCCCO": "n-Hexanol",
    "CCCCCCCO": "n-Heptanol",
    "CCCCCCCCCO": "n-Nonanol",
    "CCCCCCCCCCO": "n-Decanol",
    "CCCCCCCCCCCO": "n-Undecanol",
    "CCCCCCCCCCCCO": "n-Dodecanol",
    "O=CO": "Formic acid",
    "ClC(Cl)(Cl)Cl": "Carbon tetrachloride",
    "CC1CCCCC1": "Methylcyclohexane",
    "CCCCCCCC": "Octane",
    "CCc1ccccc1": "Ethylbenzene",
    "CCCCC(CC)CO": "2-Ethyl-1-hexanol",
    "CCCCOCCO": "2-Butoxyethanol",
    "CC(=O)OC(C)C": "Isopropyl acetate",
    "CC(=O)CC(C)C": "Methyl isobutyl ketone",
    "CCCCOCCCC": "Dibutyl ether",
    "CCOC=O": "Ethyl formate",
    "CCCCCC(C)C": "Isooctane",
    "CCCOCCO": "2-Propoxyethanol",
    "Cc1ccc(C)cc1": "p-Xylene",
    "OC1CCCCC1": "Cyclohexanol",
    "O=C1CCCCCO1": "epsilon-Caprolactone",
}


@dataclass(frozen=True)
class StandardizationResult:
    input_path: Path
    output_path: Path
    audit_path: Path
    rows_read: int
    solvent_cells: int
    changed_cells: int
    canonical_solvents: int
    original_name_variants: int


def default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_standardized_solvents{path.suffix}")


def default_audit_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_solvent_name_map.csv")


def molecule_from_smiles(smiles: str) -> Chem.Mol | None:
    text = smiles.strip()
    if not text:
        return None
    return Chem.MolFromSmiles(text)


def canonicalize_smiles(smiles: str) -> str:
    text = smiles.strip()
    molecule = molecule_from_smiles(text)
    if molecule is None:
        return text
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def inchikey_from_smiles(smiles: str) -> str:
    text = smiles.strip()
    molecule = molecule_from_smiles(text)
    if molecule is None:
        return text
    return Chem.MolToInchiKey(molecule)


def canonicalized_curated_names() -> dict[str, str]:
    return {inchikey_from_smiles(smiles): name for smiles, name in CURATED_SOLVENT_NAMES.items()}


def collect_solvent_observations(
    csv_path: str | Path,
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]], int, int]:
    name_counts: dict[str, Counter[str]] = defaultdict(Counter)
    smiles_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows_read = 0
    solvent_cells = 0

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_read += 1
            for name_column, smiles_column in SOLVENT_COLUMNS:
                name = (row.get(name_column) or "").strip()
                smiles = (row.get(smiles_column) or "").strip()
                if not name:
                    continue
                solvent_cells += 1
                if smiles:
                    inchikey = inchikey_from_smiles(smiles)
                    name_counts[inchikey][name] += 1
                    smiles_counts[inchikey][smiles] += 1

    return dict(name_counts), dict(smiles_counts), rows_read, solvent_cells


def choose_frequency_name(counter: Counter[str]) -> str:
    max_count = max(counter.values())
    candidates = [name for name, count in counter.items() if count == max_count]
    return sorted(candidates, key=lambda value: value.lower())[0]


def build_canonical_name_map(
    name_counts: Mapping[str, Counter[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    curated = canonicalized_curated_names()
    canonical_names: dict[str, str] = {}
    sources: dict[str, str] = {}

    for smiles, counter in name_counts.items():
        if smiles in curated:
            canonical_names[smiles] = curated[smiles]
            sources[smiles] = "curated"
        else:
            canonical_names[smiles] = choose_frequency_name(counter)
            sources[smiles] = "frequency_fallback"

    return canonical_names, sources


def build_audit_rows(
    name_counts: Mapping[str, Counter[str]],
    smiles_counts: Mapping[str, Counter[str]],
    canonical_names: Mapping[str, str],
    sources: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for inchikey in sorted(name_counts):
        canonical_name = canonical_names[inchikey]
        observed_smiles = format_counter_values(smiles_counts.get(inchikey, Counter()))
        for original_name, original_count in sorted(name_counts[inchikey].items()):
            rows.append(
                {
                    "inchikey": inchikey,
                    "observed_smiles": observed_smiles,
                    "canonical_name": canonical_name,
                    "source": sources[inchikey],
                    "original_name": original_name,
                    "original_count": original_count,
                    "changed": original_name != canonical_name,
                }
            )
    return rows


def format_counter_values(counter: Counter[str]) -> str:
    return " | ".join(
        value for value, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def all_names_for_identity(
    name_counts: Mapping[str, Counter[str]],
    identity: str,
) -> Iterable[str]:
    return name_counts.get(identity, Counter()).keys()


def standardize_solvent_names_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> StandardizationResult:
    source_path = Path(input_path)
    target_path = Path(output_path) if output_path else default_output_path(source_path)
    audit_target_path = Path(audit_path) if audit_path else default_audit_path(source_path)

    name_counts, smiles_counts, rows_read, solvent_cells = collect_solvent_observations(source_path)
    canonical_names, sources = build_canonical_name_map(name_counts)

    changed_cells = 0
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(newline="", encoding="utf-8-sig") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source_path} does not contain a CSV header")

        with target_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                for name_column, smiles_column in SOLVENT_COLUMNS:
                    name = (row.get(name_column) or "").strip()
                    smiles = (row.get(smiles_column) or "").strip()
                    if not name or not smiles:
                        continue
                    canonical_name = canonical_names.get(inchikey_from_smiles(smiles), name)
                    if row[name_column] != canonical_name:
                        changed_cells += 1
                    row[name_column] = canonical_name
                writer.writerow(row)

    audit_target_path.parent.mkdir(parents=True, exist_ok=True)
    audit_rows = build_audit_rows(name_counts, smiles_counts, canonical_names, sources)
    with audit_target_path.open("w", newline="", encoding="utf-8") as audit_handle:
        fieldnames = [
            "inchikey",
            "observed_smiles",
            "canonical_name",
            "source",
            "original_name",
            "original_count",
            "changed",
        ]
        writer = csv.DictWriter(audit_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    return StandardizationResult(
        input_path=source_path,
        output_path=target_path,
        audit_path=audit_target_path,
        rows_read=rows_read,
        solvent_cells=solvent_cells,
        changed_cells=changed_cells,
        canonical_solvents=len(canonical_names),
        original_name_variants=len(audit_rows),
    )
