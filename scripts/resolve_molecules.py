from __future__ import annotations

import argparse
from pathlib import Path

from bigmixsoldb.molecule_lookup import CHEMSCRIPT_AVAILABLE, resolve_molecule_json_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill SMILES fields in a molecule JSON file using ChemScript with PubChem fallback."
    )
    parser.add_argument("input", help="Path to the molecule JSON file produced by extract_molecule_names.py.")
    parser.add_argument(
        "--output",
        help="Path to write the updated JSON file. Defaults to <input_stem>_resolved.json.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Re-resolve entries that already contain a SMILES value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many entries. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--pubchem-timeout",
        type=int,
        default=15,
        help="Timeout in seconds for each PubChem request.",
    )
    parser.add_argument(
        "--pubchem-limit",
        type=int,
        default=5,
        help="Number of PubChem autocomplete candidates to inspect per query.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_resolved.json")

    summary = resolve_molecule_json_file(
        input_path,
        output_path,
        replace_existing=args.replace_existing,
        limit=args.limit,
        pubchem_timeout=args.pubchem_timeout,
        pubchem_limit=args.pubchem_limit,
    )

    print(f"ChemScript available: {'yes' if CHEMSCRIPT_AVAILABLE else 'no'}")
    print(f"Processed entries: {summary['total']}")
    print(f"Resolved entries: {summary['resolved']}")
    print(f"Skipped entries: {summary['skipped']}")
    print(f"Unresolved entries: {summary['unresolved']}")
    print(f"Resolved with ChemScript: {summary['chemscript']}")
    print(f"Resolved with PubChem: {summary['pubchem']}")
    print(f"Wrote updated molecule JSON to: {output_path}")


if __name__ == "__main__":
    main()