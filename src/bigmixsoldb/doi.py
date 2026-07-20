from __future__ import annotations

import math
from typing import Any, Iterable


DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def normalize_doi(value: Any) -> str:
    """Return a normalized individual DOI, without changing valid DOI characters."""
    if _is_missing(value):
        return ""

    doi = str(value).strip().replace("", ":").lower()
    for prefix in DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi.removeprefix(prefix).strip()
            break
    return doi.rstrip(".,;").strip()


def split_dois(value: Any) -> list[str]:
    """Split and normalize a semicolon-separated DOI provenance cell."""
    if _is_missing(value):
        return []

    dois: list[str] = []
    seen: set[str] = set()
    for part in str(value).split(";"):
        doi = normalize_doi(part)
        if doi and doi not in seen:
            seen.add(doi)
            dois.append(doi)
    return dois


def unique_dois(values: Iterable[Any]) -> set[str]:
    return {doi for value in values for doi in split_dois(value)}
