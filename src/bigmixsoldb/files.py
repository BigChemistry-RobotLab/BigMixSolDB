from __future__ import annotations

from pathlib import Path
from typing import Sequence


def collect_input_files(inputs: Sequence[str | Path], suffixes: set[str] | None = None) -> list[Path]:
    files: list[Path] = []
    normalized_suffixes = {suffix.lower() for suffix in suffixes} if suffixes else None

    for raw_path in inputs:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if normalized_suffixes and path.suffix.lower() not in normalized_suffixes:
                continue
            files.append(path)
            continue

        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            if normalized_suffixes and child.suffix.lower() not in normalized_suffixes:
                continue
            files.append(child)

    seen: set[Path] = set()
    unique_files: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)

    return unique_files


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_output_path(input_path: str | Path, output_dir: str | Path, suffix: str) -> Path:
    input_file = Path(input_path)
    target_dir = ensure_directory(output_dir)
    return target_dir / f"{input_file.stem}{suffix}"


def normalize_doi_from_stem(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.replace("_", "/").replace("", ":")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
