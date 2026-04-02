from __future__ import annotations

import argparse

from bigmixsoldb.files import collect_input_files
from bigmixsoldb.molecules import write_molecule_placeholders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract molecule names from YAML files and write a JSON file with SMILES placeholders." \
        "This can be used as input to convert_molecule_smiles.py to fill in the SMILES strings.",
    )
    parser.add_argument("inputs", nargs="+", help="YAML files or directories containing YAML files.")
    parser.add_argument("--output", required=True, help="Path of the JSON file to write.")
    args = parser.parse_args()

    inputs = collect_input_files(args.inputs, suffixes={".yml", ".yaml"})
    if not inputs:
        raise SystemExit("No YAML files were found in the provided inputs.")

    write_molecule_placeholders(inputs, args.output)


if __name__ == "__main__":
    main()
