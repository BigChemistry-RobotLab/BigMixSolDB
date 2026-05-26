from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from bigmixsoldb.constants import OUTPUT_COLUMNS
from bigmixsoldb.files import collect_input_files
from bigmixsoldb.filtering import filter_entries_like_reference
from bigmixsoldb.molecules import load_molecule_lookup
from bigmixsoldb.postprocess import flatten_yaml_file, merge_dataframes


def format_count(value: int) -> str:
    return f"{value:,}"


def ordered_mixture_types(values: Iterable[str]) -> list[str]:
    order = ["single", "binary", "ternary", "extra"]
    present = set(values)
    return [mixture_type for mixture_type in order if mixture_type in present]


def displayed_before_count(stats: dict[str, int]) -> int:
    return stats["before"] + stats.get("blocked_smiles_removed", 0)


def print_prefilter_summary(filter_stats: dict[str, dict[str, int]]) -> None:
    print("System counts before filtering:")
    for mixture_type in ordered_mixture_types(filter_stats):
        before = displayed_before_count(filter_stats[mixture_type])
        print(f"  {mixture_type:8s}: {format_count(before)}")


def print_filter_summary(filter_stats: dict[str, dict[str, int]]) -> None:
    print("Filtering summary:")
    for mixture_type in ordered_mixture_types(filter_stats):
        stats = filter_stats[mixture_type]
        before = displayed_before_count(stats)
        removed = before - stats["after"]
        print(
            f"[{mixture_type}] {before:>7,} rows -> {stats['after']:>7,} kept "
            f"({removed:,} removed)"
        )
        print(f"  Missing SMILES removed    : {stats['smiles_removed']:,}")
        print(f"  Blocked SMILES removed    : {stats.get('blocked_smiles_removed', 0):,}")
        print(f"  Unit and values removed   : {stats['unit_removed']:,}")
        print(f"  Concentration sum removed : {stats.get('concentration_sum_removed', 0):,}")
        print(f"  Temperature removed       : {stats['temperature_removed']:,}")
        print(f"  Both filters removed      : {stats['both_removed']:,}")


def add_blocked_smiles_stats(
    filter_stats: dict[str, dict[str, int]],
    build_stats: dict[str, Any],
) -> None:
    blocked_by_type = build_stats.get("disabled_molecule_rows_removed_by_type", {})
    for mixture_type in ordered_mixture_types([*filter_stats.keys(), *blocked_by_type.keys()]):
        stats = filter_stats.setdefault(
            mixture_type,
            {
                "before": 0,
                "after": 0,
                "smiles_removed": 0,
                "unit_removed": 0,
                "concentration_sum_removed": 0,
                "temperature_removed": 0,
                "both_removed": 0,
            },
        )
        stats["blocked_smiles_removed"] = int(blocked_by_type.get(mixture_type, 0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standardized filtered CSV directly from extracted YAML input."
    )
    parser.add_argument("inputs", nargs="+", help="YAML file(s) or directories containing YAML files.")
    parser.add_argument("--output", required=True, help="Filtered CSV output path.")
    parser.add_argument(
        "--molecules",
        help="Optional JSON file mapping molecule names to SMILES, such as data/name_to_smiles.json.",
    )
    args = parser.parse_args()

    input_paths = collect_input_files(args.inputs, suffixes={".yml", ".yaml"})
    if not input_paths:
        raise SystemExit("No YAML files were found in the provided inputs.")

    print(f"Found {len(input_paths):,} YAML input files.")
    molecule_lookup = load_molecule_lookup(args.molecules)
    if args.molecules:
        enabled_count = sum(1 for record in molecule_lookup.values() if record.enabled)
        disabled_count = sum(1 for record in molecule_lookup.values() if not record.enabled)
        print(
            f"Loaded {len(molecule_lookup):,} molecule lookup names "
            f"({enabled_count:,} enabled, {disabled_count:,} disabled)."
        )

    frames: list[pd.DataFrame] = []
    empty_files: list[str] = []
    errored_files: dict[str, str] = {}
    build_stats: dict[str, Any] = {
        "disabled_molecule_rows_removed": 0,
        "disabled_molecule_rows_removed_by_type": {},
    }

    for yaml_path in tqdm(input_paths, desc="Build filtered CSV"):
        if yaml_path.read_text(encoding="utf-8").strip() == "":
            empty_files.append(str(yaml_path))
            continue
        try:
            frames.append(flatten_yaml_file(yaml_path, molecule_lookup=molecule_lookup, stats=build_stats))
        except Exception as exc:  # pragma: no cover - CLI reporting path
            errored_files[str(yaml_path)] = str(exc)

    merged = merge_dataframes(frames)

    total_before_filtering = len(merged) + build_stats["disabled_molecule_rows_removed"]
    print(f"Compiled {total_before_filtering:,} standardized rows from YAML files before filtering.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path.with_stem(output_path.stem + "_unfiltered"), index=False)

    filtered, filter_stats = filter_entries_like_reference(merged)
    add_blocked_smiles_stats(filter_stats, build_stats)
    filtered = filtered[[column for column in OUTPUT_COLUMNS if column in filtered.columns]]

    # print(f"Compiled {len(merged):,} standardized rows before filtering.")
    print_prefilter_summary(filter_stats)
    print_filter_summary(filter_stats)

    extra_after = filter_stats.get("extra", {}).get("after", 0)
    if extra_after:
        print(f"Excluded {extra_after:,} filtered extra-system rows from the final CSV.")

    if filtered["Extra Solvents"].isna().all():
        filtered = filtered.drop(columns=["Extra Solvents"])
    else:
        print(
            "Warning: 'Extra Solvents' column contains non-empty values. "
            "These rows will be included in the final CSV, but may require special handling."
        )
    filtered.to_csv(output_path, index=False)

    print(f"Built filtered dataset: {output_path} ({len(filtered)} rows)")
    if empty_files:
        print(f"Skipped {len(empty_files)} empty YAML files.")
    if errored_files:
        print(f"Encountered {len(errored_files)} non-empty YAML processing errors.")
        for path, error in sorted(errored_files.items()):
            print(f"  {path}: {error}")
    else:
        print("Processed all non-empty YAML files without runtime errors.")


if __name__ == "__main__":
    main()
