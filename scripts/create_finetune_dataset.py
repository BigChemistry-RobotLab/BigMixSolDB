from __future__ import annotations

import argparse
import logging

from bigmixsoldb.finetune_dataset import create_finetune_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSONL finetuning dataset by pairing Markdown files with YAML labels."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Markdown files or directories containing Markdown files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="YAML files or directories containing YAML labels.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSONL dataset that will be written.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/extract_yaml_prompt.txt",
        help="Prompt file used for the system message in each training example.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an input Markdown file has no matching YAML label or if a file is empty.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    written_records = create_finetune_dataset(
        markdown_inputs=args.inputs,
        label_inputs=args.labels,
        output_path=args.output,
        prompt_file=args.prompt_file,
        strict=args.strict,
    )
    print(f"Wrote {written_records} finetuning examples to {args.output}")


if __name__ == "__main__":
    main()