from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from bigmixsoldb.constants import OUTPUT_COLUMNS
from bigmixsoldb.conversion import convert_dataframe, write_conversion_report
from bigmixsoldb.files import collect_input_files
from bigmixsoldb.filtering import (
    classify_mixture_type,
    filter_entries_like_reference,
    filter_supported_solubility_units,
    filter_valid_solubility_results,
)
from bigmixsoldb.molecules import load_molecule_lookup
from bigmixsoldb.postprocess import (
    dedupe_condition_rows,
    filter_complete_rows,
    flatten_yaml_file,
    is_missing,
)

MIXTURE_TYPES = ("single", "binary", "ternary", "extra")
BLOCKED_COLUMN = "_Blocked Molecule"


def format_count(value: int) -> str:
    return f"{value:,}"


def ordered_mixture_types(values: Iterable[str]) -> list[str]:
    order = ["single", "binary", "ternary", "extra"]
    present = set(values)
    return [mixture_type for mixture_type in order if mixture_type in present]


def displayed_before_count(stats: dict[str, int]) -> int:
    return stats["before"] + stats.get("blocked_smiles_removed", 0)


def print_prefilter_summary(
    filter_stats: dict[str, dict[str, int]],
    before_by_type: dict[str, int] | None = None,
) -> None:
    print("System counts before filtering:")
    counts = before_by_type or {
        mixture_type: displayed_before_count(stats)
        for mixture_type, stats in filter_stats.items()
    }
    for mixture_type in ordered_mixture_types(counts):
        before = counts[mixture_type]
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
        print(f"  Solubility unit removed   : {stats.get('solubility_unit_removed', 0):,}")
        print(f"  Concentration sum removed : {stats.get('concentration_sum_removed', 0):,}")
        print(f"  Temperature removed       : {stats['temperature_removed']:,}")
        print(f"  Extra-system rows removed : {stats.get('extra_scope_removed', 0):,}")


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
                "solubility_unit_removed": 0,
                "concentration_sum_removed": 0,
                "temperature_removed": 0,
                "extra_scope_removed": 0,
                "both_removed": 0,
            },
        )
        stats["blocked_smiles_removed"] = int(blocked_by_type.get(mixture_type, 0))


def mixture_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    if dataframe.empty:
        return {mixture_type: 0 for mixture_type in MIXTURE_TYPES}
    counts = dataframe.apply(classify_mixture_type, axis=1).value_counts()
    return {
        mixture_type: int(counts.get(mixture_type, 0))
        for mixture_type in MIXTURE_TYPES
    }


def add_counts(*counts: dict[str, int]) -> dict[str, int]:
    return {
        mixture_type: sum(item.get(mixture_type, 0) for item in counts)
        for mixture_type in MIXTURE_TYPES
    }


def filtering_counts(
    filter_stats: dict[str, dict[str, int]],
    key: str,
) -> dict[str, int]:
    return {
        mixture_type: int(filter_stats.get(mixture_type, {}).get(key, 0))
        for mixture_type in MIXTURE_TYPES
    }


def subtract_counts(
    before_by_type: dict[str, int],
    removed_by_type: dict[str, int],
    *,
    stage: str,
) -> dict[str, int]:
    after_by_type = {
        mixture_type: before_by_type.get(mixture_type, 0)
        - removed_by_type.get(mixture_type, 0)
        for mixture_type in MIXTURE_TYPES
    }
    if any(value < 0 for value in after_by_type.values()):
        raise ValueError(f"Filtering removals exceed available rows at {stage}")
    return after_by_type


def attrition_stage(
    name: str,
    before_by_type: dict[str, int],
    after_by_type: dict[str, int],
) -> dict[str, Any]:
    removed_by_type = {
        mixture_type: before_by_type.get(mixture_type, 0)
        - after_by_type.get(mixture_type, 0)
        for mixture_type in MIXTURE_TYPES
    }
    if any(value < 0 for value in removed_by_type.values()):
        raise ValueError(f"Non-sequential attrition counts for {name}")
    return {
        "stage": name,
        "before": int(sum(before_by_type.values())),
        "after": int(sum(after_by_type.values())),
        "removed": int(sum(removed_by_type.values())),
        "before_by_type": before_by_type,
        "after_by_type": after_by_type,
        "removed_by_type": removed_by_type,
    }


def print_sequential_attrition(stages: dict[str, dict[str, Any]]) -> None:
    print("Sequential attrition:")
    for stage in stages.values():
        print(
            f"  {stage['stage']:24s}: {stage['before']:>7,} -> "
            f"{stage['after']:>7,} ({stage['removed']:,} removed)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standardized filtered CSV directly from extracted YAML input."
    )
    parser.add_argument("inputs", nargs="+", help="YAML file(s) or directories containing YAML files.")
    parser.add_argument("--output", required=True, help="Filtered CSV output path.")
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Optional JSON path for machine-readable filtering diagnostics.",
    )
    parser.add_argument(
        "--molecules",
        help="Optional JSON file mapping molecule names to SMILES, such as data/name_to_smiles.json.",
    )
    parser.add_argument(
        "--density-manifest",
        type=Path,
        help="Optional offline exact-temperature density manifest.",
    )
    parser.add_argument(
        "--conversion-report",
        type=Path,
        help="Optional CSV or JSON audit report for every solubility conversion attempt.",
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
    blocked_rows: list[dict[str, Any]] = []
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
            frames.append(
                flatten_yaml_file(
                    yaml_path,
                    molecule_lookup=molecule_lookup,
                    stats=build_stats,
                    blocked_rows=blocked_rows,
                )
            )
        except Exception as exc:  # pragma: no cover - CLI reporting path
            errored_files[str(yaml_path)] = str(exc)

    if frames:
        standardized = pd.concat(frames, ignore_index=True, sort=False)
    else:
        standardized = pd.DataFrame(columns=OUTPUT_COLUMNS)
    standardized[BLOCKED_COLUMN] = False
    blocked_frame = pd.DataFrame(blocked_rows)
    if not blocked_frame.empty:
        blocked_frame[BLOCKED_COLUMN] = True
    all_rows = pd.concat(
        [standardized, blocked_frame],
        ignore_index=True,
        sort=False,
    )
    if BLOCKED_COLUMN not in all_rows:
        all_rows[BLOCKED_COLUMN] = False
    all_rows[BLOCKED_COLUMN] = all_rows[BLOCKED_COLUMN].fillna(False).astype(bool)
    converted_rows, conversion_reports = convert_dataframe(
        all_rows,
        args.density_manifest,
    )
    converted_blocked = converted_rows[converted_rows[BLOCKED_COLUMN]].copy()
    build_stats["disabled_molecule_rows_removed_by_type"] = {
        str(mixture_type): int(count)
        for mixture_type, count in converted_blocked.apply(
            classify_mixture_type, axis=1
        ).value_counts().items()
    }
    before_by_type = mixture_counts(converted_rows)
    blocked_by_type = mixture_counts(converted_blocked)
    merged = converted_rows[~converted_rows[BLOCKED_COLUMN]].copy()

    total_before_filtering = len(all_rows)
    print(f"Compiled {total_before_filtering:,} standardized rows from YAML files before filtering.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # This is intentionally before completeness filtering and condition deduplication.
    unfiltered = converted_rows[[column for column in OUTPUT_COLUMNS if column in converted_rows]]
    unfiltered.to_csv(output_path.with_stem(output_path.stem + "_unfiltered"), index=False)

    complete = filter_complete_rows(merged, require_numeric_solubility=False)
    after_complete_by_type = add_counts(mixture_counts(complete), blocked_by_type)

    valid_results = filter_valid_solubility_results(complete)
    after_invalid_by_type = add_counts(mixture_counts(valid_results), blocked_by_type)

    reference_filtered, filter_stats = filter_entries_like_reference(valid_results)
    add_blocked_smiles_stats(filter_stats, build_stats)
    after_molecular_scope_by_type = mixture_counts(valid_results)
    after_missing_smiles_by_type = subtract_counts(
        after_molecular_scope_by_type,
        filtering_counts(filter_stats, "smiles_removed"),
        stage="missing SMILES",
    )
    after_unit_value_by_type = subtract_counts(
        after_missing_smiles_by_type,
        filtering_counts(filter_stats, "unit_removed"),
        stage="unit or value errors",
    )
    after_concentration_sum_by_type = subtract_counts(
        after_unit_value_by_type,
        filtering_counts(filter_stats, "concentration_sum_removed"),
        stage="concentration sum",
    )
    after_temperature_by_type = subtract_counts(
        after_concentration_sum_by_type,
        filtering_counts(filter_stats, "temperature_removed"),
        stage="temperature",
    )
    after_extra_scope_by_type = subtract_counts(
        after_temperature_by_type,
        filtering_counts(filter_stats, "extra_scope_removed"),
        stage="extra-solvent scope",
    )
    after_reference_by_type = mixture_counts(reference_filtered)
    if after_extra_scope_by_type != after_reference_by_type:
        raise ValueError(
            "Disjoint reference-filter counts do not reconcile with retained rows"
        )

    supported_units = filter_supported_solubility_units(reference_filtered)
    after_unit_by_type = mixture_counts(supported_units)
    for mixture_type in MIXTURE_TYPES:
        if (
            mixture_type not in filter_stats
            and blocked_by_type.get(mixture_type, 0) == 0
            and after_reference_by_type.get(mixture_type, 0) == 0
        ):
            continue
        stats = filter_stats.setdefault(
            mixture_type,
            {
                "before": 0,
                "after": 0,
                "smiles_removed": 0,
                "unit_removed": 0,
                "solubility_unit_removed": 0,
                "concentration_sum_removed": 0,
                "temperature_removed": 0,
                "extra_scope_removed": 0,
                "both_removed": 0,
                "blocked_smiles_removed": blocked_by_type.get(mixture_type, 0),
            },
        )
        stats["reference_after"] = int(after_reference_by_type[mixture_type])
        stats["solubility_unit_removed"] = int(
            after_reference_by_type[mixture_type] - after_unit_by_type[mixture_type]
        )
        stats["after"] = int(after_unit_by_type[mixture_type])

    filtered = dedupe_condition_rows(supported_units)
    after_dedup_by_type = mixture_counts(filtered)
    filtered = filtered[[column for column in OUTPUT_COLUMNS if column in filtered.columns]]

    stages = {
        "completeness": attrition_stage(
            "completeness",
            before_by_type,
            after_complete_by_type,
        ),
        "invalid_result": attrition_stage(
            "invalid result",
            after_complete_by_type,
            after_invalid_by_type,
        ),
        "outside_molecular_scope": attrition_stage(
            "outside molecular scope",
            after_invalid_by_type,
            after_molecular_scope_by_type,
        ),
        "missing_smiles": attrition_stage(
            "missing SMILES",
            after_molecular_scope_by_type,
            after_missing_smiles_by_type,
        ),
        "unit_or_value": attrition_stage(
            "unit or value errors",
            after_missing_smiles_by_type,
            after_unit_value_by_type,
        ),
        "concentration_sum": attrition_stage(
            "concentrations not summing to 1.0",
            after_unit_value_by_type,
            after_concentration_sum_by_type,
        ),
        "temperature": attrition_stage(
            "lack of temperature values",
            after_concentration_sum_by_type,
            after_temperature_by_type,
        ),
        "extra_scope": attrition_stage(
            "contains more than 3 solvents",
            after_temperature_by_type,
            after_extra_scope_by_type,
        ),
        "solubility_unit": attrition_stage(
            "solubility unit",
            after_reference_by_type,
            after_unit_by_type,
        ),
        "deduplication": attrition_stage(
            "condition deduplication",
            after_unit_by_type,
            after_dedup_by_type,
        ),
    }

    # print(f"Compiled {len(merged):,} standardized rows before filtering.")
    print_prefilter_summary(filter_stats, before_by_type)
    print_filter_summary(filter_stats)
    print_sequential_attrition(stages)

    extra_removed = filter_stats.get("extra", {}).get("extra_scope_removed", 0)
    if extra_removed:
        print(f"Excluded {extra_removed:,} extra-system rows from the final CSV.")

    if filtered.empty or filtered["Extra Solvents"].map(is_missing).all():
        filtered = filtered.drop(columns=["Extra Solvents"])
    else:
        print(
            "Warning: 'Extra Solvents' column contains non-empty values. "
            "These rows will be included in the final CSV, but may require special handling."
        )
    filtered.to_csv(output_path, index=False)

    if args.conversion_report:
        write_conversion_report(conversion_reports, args.conversion_report)
        print(f"Wrote conversion audit: {args.conversion_report}")

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "standardized_rows_before_filtering": total_before_filtering,
            "standardized_rows_before_filtering_by_type": before_by_type,
            "rows_removed_before_reference_filter": {
                mixture_type: before_by_type.get(mixture_type, 0)
                - after_invalid_by_type.get(mixture_type, 0)
                for mixture_type in MIXTURE_TYPES
            },
            "filtering": filter_stats,
            "sequential_attrition": stages,
            "attrition_stages": list(stages.values()),
        }
        args.stats_output.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote filtering diagnostics: {args.stats_output}")

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
