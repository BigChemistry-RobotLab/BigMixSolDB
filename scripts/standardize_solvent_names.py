from __future__ import annotations

import argparse

from bigmixsoldb.solvent_standardization import standardize_solvent_names_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize solvent display names in a BigMixSolDB CSV."
    )
    parser.add_argument("input_csv", help="Input CSV with Solvent 1/2/3 and SMILES_Solvent_* columns.")
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to <input>_standardized_solvents.csv.",
    )
    parser.add_argument(
        "--audit-output",
        help="Audit mapping CSV path. Defaults to <input>_solvent_name_map.csv.",
    )
    args = parser.parse_args()

    result = standardize_solvent_names_csv(
        args.input_csv,
        output_path=args.output,
        audit_path=args.audit_output,
    )

    print(f"Rows read: {result.rows_read:,}")
    print(f"Solvent cells: {result.solvent_cells:,}")
    print(f"Changed solvent cells: {result.changed_cells:,}")
    print(f"Canonical solvents: {result.canonical_solvents:,}")
    print(f"Original name variants: {result.original_name_variants:,}")
    print(f"Wrote standardized CSV: {result.output_path}")
    print(f"Wrote audit map: {result.audit_path}")


if __name__ == "__main__":
    main()
