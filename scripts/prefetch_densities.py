from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from bigmixsoldb.conversion import classify_conversion
from bigmixsoldb.density import (
    PubChemDensityClient,
    PubChemPermanentError,
    PubChemRecordMissing,
    PubChemTransientError,
    chemical_identity,
    empty_manifest,
    seed_local_density_csv,
    select_exact_density,
    write_density_manifest,
)
from bigmixsoldb.files import collect_input_files
from bigmixsoldb.molecules import load_molecule_lookup
from bigmixsoldb.postprocess import flatten_yaml_file, is_missing, parse_numeric_value


def _single_solvent(row: pd.Series) -> bool:
    return (
        not is_missing(row.get("Solvent 1"))
        and is_missing(row.get("Solvent 2"))
        and is_missing(row.get("Solvent 3"))
        and is_missing(row.get("Extra Solvents"))
    )


def density_requests(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    requests_to_make: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for position, row in dataframe.iterrows():
        original_unit = row.get("_Original Solubility Unit", row.get("Solubility Unit"))
        method, _, _ = classify_conversion(original_unit)
        if method not in {"molarity", "mass_volume"} or not _single_solvent(row):
            continue
        identity = chemical_identity(row.get("SMILES_Solvent_1"))
        temperature = parse_numeric_value(row.get("Temperature"))
        inchikey = identity[1] if identity else ""
        key = (inchikey or f"invalid:{position}", temperature or float("nan"))
        if identity and temperature is not None and key in seen:
            continue
        seen.add(key)
        requests_to_make.append(
            {
                "row_id": int(position) if isinstance(position, int) else str(position),
                "doi": row.get("doi", ""),
                "solvent_name": row.get("Solvent 1", ""),
                "smiles": row.get("SMILES_Solvent_1", ""),
                "canonical_smiles": identity[0] if identity else "",
                "inchikey": inchikey,
                "temperature_k": temperature,
                "original_unit": original_unit,
                "conversion_method": method,
                "status": "pending",
                "failure_reason": "",
            }
        )
    return requests_to_make


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prefetch exact-temperature solvent densities into an auditable manifest. "
            "This is the only density workflow that accesses PubChem."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Extracted YAML files/directories or a CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output density manifest JSON.")
    parser.add_argument("--molecules", help="Molecule-name to SMILES JSON used for YAML inputs.")
    parser.add_argument(
        "--local-density-csv",
        type=Path,
        default=Path("data/bigsoldb_densities.csv"),
        help="Local density source (default: data/bigsoldb_densities.csv when present).",
    )
    parser.add_argument(
        "--no-pubchem",
        action="store_true",
        help="Seed local observations and requests without querying PubChem.",
    )
    args = parser.parse_args()
    input_errors: dict[str, str] = {}
    empty_files: list[str] = []

    if len(args.inputs) == 1 and Path(args.inputs[0]).suffix.lower() == ".csv":
        dataframe = pd.read_csv(args.inputs[0], low_memory=False)
    else:
        paths = collect_input_files(args.inputs, suffixes={".yml", ".yaml"})
        if not paths:
            raise SystemExit("No YAML files were found in the provided inputs.")
        lookup = load_molecule_lookup(args.molecules)
        blocked_rows: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []
        for path in tqdm(paths, desc="Scan density requests"):
            if not path.read_text(encoding="utf-8").strip():
                empty_files.append(str(path))
                continue
            try:
                frames.append(
                    flatten_yaml_file(
                        path,
                        molecule_lookup=lookup,
                        blocked_rows=blocked_rows,
                    )
                )
            except Exception as exc:
                input_errors[str(path)] = str(exc)
        if blocked_rows:
            frames.append(pd.DataFrame(blocked_rows))
        dataframe = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    manifest = empty_manifest()
    manifest["input_errors"] = input_errors
    manifest["empty_input_files"] = empty_files
    if args.local_density_csv.exists():
        manifest["observations"].extend(seed_local_density_csv(args.local_density_csv))
    manifest["requests"] = density_requests(dataframe)

    missing_identities: dict[str, list[dict[str, Any]]] = {}
    for request in manifest["requests"]:
        inchikey = request["inchikey"]
        temperature = request["temperature_k"]
        if not inchikey:
            request.update(status="failed", failure_reason="invalid_solvent_smiles")
        elif temperature is None:
            request.update(status="failed", failure_reason="missing_temperature")
        elif select_exact_density(manifest, inchikey, temperature) is not None:
            request.update(status="satisfied_local", failure_reason="")
        else:
            missing_identities.setdefault(inchikey, []).append(request)

    if not args.no_pubchem:
        session = requests.Session()
        session.headers["User-Agent"] = "bigmixsoldb-density-prefetch/1.0"
        client = PubChemDensityClient(session)
        for inchikey, identity_requests in tqdm(
            missing_identities.items(), desc="Prefetch PubChem Density"
        ):
            try:
                cid = client.resolve_cid(inchikey)
                observations = client.fetch_density(cid, inchikey)
                manifest["observations"].extend(observations)
                for request in identity_requests:
                    selected = select_exact_density(
                        manifest, inchikey, float(request["temperature_k"])
                    )
                    if selected is None:
                        request.update(
                            status="failed",
                            failure_reason="missing_exact_density",
                            cid=cid,
                        )
                    else:
                        request.update(
                            status="satisfied_pubchem",
                            failure_reason="",
                            cid=cid,
                        )
            except PubChemRecordMissing as exc:
                for request in identity_requests:
                    request.update(
                        status="failed",
                        failure_reason="pubchem_record_missing",
                        error=str(exc),
                    )
            except PubChemPermanentError as exc:
                for request in identity_requests:
                    request.update(
                        status="failed",
                        failure_reason="pubchem_permanent_failure",
                        error=str(exc),
                    )
            except PubChemTransientError as exc:
                for request in identity_requests:
                    request.update(
                        status="failed",
                        failure_reason="pubchem_transient_failure",
                        error=str(exc),
                    )
            except Exception as exc:  # defensive: every failure remains explicit
                for request in identity_requests:
                    request.update(
                        status="failed",
                        failure_reason="pubchem_unexpected_failure",
                        error=str(exc),
                    )
    else:
        for requests_for_identity in missing_identities.values():
            for request in requests_for_identity:
                request.update(status="failed", failure_reason="missing_exact_density")

    write_density_manifest(manifest, args.output)
    satisfied = sum(str(item["status"]).startswith("satisfied") for item in manifest["requests"])
    print(
        f"Wrote {args.output}: {len(manifest['requests']):,} requests, "
        f"{len(manifest['observations']):,} retained observations, {satisfied:,} satisfied."
    )
    if empty_files:
        print(f"Skipped {len(empty_files):,} empty YAML files.", file=sys.stderr)
    if input_errors:
        print(
            f"Skipped {len(input_errors):,} malformed YAML files during density prefetch:",
            file=sys.stderr,
        )
        for path, error in sorted(input_errors.items()):
            print(f"  {path}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
