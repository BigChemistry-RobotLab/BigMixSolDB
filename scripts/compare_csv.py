from __future__ import annotations

import argparse

from bigmixsoldb.compare import compare_csv_files, print_comparison_summary, write_comparison_report


def label_for_format(fmt: str, role: str) -> str:
    normalized = fmt.lower()
    if normalized == "mixturesoldb":
        return "MixtureSolDB"
    if normalized == "bigsoldb":
        return "BigSolDB"
    if role == "predicted":
        return "BigMixSolDB"
    return "Reference CSV"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a predicted CSV against a reference dataset.")
    parser.add_argument("--predicted", required=True, help="Predicted CSV path.")
    parser.add_argument("--reference", required=True, help="Reference CSV path.")
    parser.add_argument(
        "--reference-format",
        choices=["standardized", "mixturesoldb", "bigsoldb"],
        required=True,
        help="Reference dataset format.",
    )
    parser.add_argument("--output", required=True, help="JSON path for the comparison report.")
    args = parser.parse_args()

    report = compare_csv_files(
        args.predicted,
        args.reference,
        reference_format=args.reference_format,
    )
    print_comparison_summary(
        report,
        predicted_label=label_for_format("standardized", "predicted"),
        reference_label=label_for_format(args.reference_format, "reference"),
    )
    target_path = write_comparison_report(report, args.output)
    print(f"Saved comparison report to {target_path}")


if __name__ == "__main__":
    main()