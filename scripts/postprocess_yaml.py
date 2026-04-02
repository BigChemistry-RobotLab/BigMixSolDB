from __future__ import annotations

import argparse

from tqdm import tqdm

from bigmixsoldb.files import build_output_path, collect_input_files
from bigmixsoldb.postprocess import write_postprocessed_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extracted YAML files into standardized CSV files.")
    parser.add_argument("inputs", nargs="+", help="YAML files or directories containing YAML files.")
    parser.add_argument("--output-dir", required=True, help="Directory where CSV files will be written.")
    parser.add_argument(
        "--molecules",
        help="Optional JSON file mapping molecule names to SMILES, such as the output of convert_molecule_smiles.py.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files.")
    args = parser.parse_args()

    inputs = collect_input_files(args.inputs, suffixes={".yml", ".yaml"})
    if not inputs:
        raise SystemExit("No YAML files were found in the provided inputs.")

    for yaml_path in tqdm(inputs, desc="Postprocess"):
        output_path = build_output_path(yaml_path, args.output_dir, ".csv")
        if output_path.exists() and not args.force:
            continue
        write_postprocessed_csv(yaml_path, output_path, molecule_json=args.molecules)


if __name__ == "__main__":
    main()
