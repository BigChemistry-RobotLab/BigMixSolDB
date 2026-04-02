from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from bigmixsoldb.files import collect_input_files, normalize_doi_from_stem, read_text
from bigmixsoldb.yaml_utils import load_prompt

logger = logging.getLogger(__name__)


def index_files_by_stem(files: Sequence[Path]) -> dict[str, Path]:
    indexed_files: dict[str, Path] = {}
    duplicate_stems: set[str] = set()

    for file_path in files:
        stem = file_path.stem
        if stem in indexed_files:
            duplicate_stems.add(stem)
            continue
        indexed_files[stem] = file_path

    if duplicate_stems:
        duplicates = ", ".join(sorted(duplicate_stems))
        raise ValueError(f"Duplicate file stems found: {duplicates}")

    return indexed_files


def collect_matching_pairs(
    markdown_inputs: Sequence[str | Path],
    label_inputs: Sequence[str | Path],
    *,
    strict: bool = False,
) -> list[tuple[Path, Path]]:
    markdown_files = collect_input_files(markdown_inputs, suffixes={".md", ".markdown"})
    if not markdown_files:
        raise ValueError("No Markdown inputs were found.")

    label_files = collect_input_files(label_inputs, suffixes={".yml", ".yaml"})
    if not label_files:
        raise ValueError("No YAML labels were found.")

    label_index = index_files_by_stem(label_files)

    matched_pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    matched_stems: set[str] = set()

    for markdown_path in markdown_files:
        label_path = label_index.get(markdown_path.stem)
        if label_path is None:
            missing_labels.append(markdown_path)
            continue

        matched_pairs.append((markdown_path, label_path))
        matched_stems.add(markdown_path.stem)

    if missing_labels:
        preview = ", ".join(path.name for path in missing_labels[:10])
        suffix = "..." if len(missing_labels) > 10 else ""
        message = (
            f"No YAML label found for {len(missing_labels)} Markdown files: {preview}{suffix}"
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    if not matched_pairs:
        raise ValueError("No matching Markdown/YAML pairs were found.")

    unused_labels = [label_path for label_path in label_files if label_path.stem not in matched_stems]
    if unused_labels:
        logger.info("Ignoring %s YAML labels without matching Markdown input.", len(unused_labels))

    return matched_pairs


def create_finetune_dataset(
    markdown_inputs: Sequence[str | Path],
    label_inputs: Sequence[str | Path],
    output_path: str | Path,
    *,
    prompt_file: str | Path = "prompts/extract_yaml_prompt.txt",
    strict: bool = False,
) -> int:
    matched_pairs = collect_matching_pairs(markdown_inputs, label_inputs, strict=strict)
    prompt = load_prompt(prompt_file)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    written_records = 0
    with destination.open("w", encoding="utf-8") as handle:
        for markdown_path, label_path in matched_pairs:
            markdown_text = read_text(markdown_path).strip()
            label_text = read_text(label_path).strip()

            if not markdown_text or not label_text:
                message = f"Skipping empty pair: {markdown_path.name} / {label_path.name}"
                if strict:
                    raise ValueError(message)
                logger.warning(message)
                continue

            record = {
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": markdown_text},
                    {"role": "assistant", "content": label_text},
                ],
                "doi": normalize_doi_from_stem(markdown_path),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_records += 1

    return written_records