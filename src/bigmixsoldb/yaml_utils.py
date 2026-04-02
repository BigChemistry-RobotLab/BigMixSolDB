from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.search(r"```(?:yaml|yml|markdown|md)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    stripped = stripped.replace("```yaml", "").replace("```yml", "")
    stripped = stripped.replace("```markdown", "").replace("```md", "")
    return stripped.replace("```", "").strip()


def clean_model_response(text: str) -> str:
    cleaned = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return strip_code_fences(cleaned)


def sanitize_yaml_text(yaml_text: str) -> str:
    def yaml_quote(value: str) -> str:
        trimmed = value.strip()
        if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
            trimmed = trimmed[1:-1]
        return "'" + trimmed.replace("'", "''") + "'"

    scalar_keys = {
        "compound",
        "synonyms",
        "solv",
        "c1",
        "c2",
        "c3",
        "cu",
        "sol",
        "su",
        "temp",
        "tu",
        "pres",
        "pu",
        "note",
    }
    scalar_list_keys = {"parsing_issues", "extraction_notes"}
    allowed_keys = scalar_keys | scalar_list_keys | {"entries", "data", "review_required"}
    key_value_pattern = re.compile(r"^(?P<indent>\s*)(?P<prefix>-\s*)?(?P<key>[^:]+):\s*(?P<value>.*)$")
    list_context: str | None = None
    root_is_list = False
    sanitized_lines: list[str] = []

    for raw_line in yaml_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        match = key_value_pattern.match(line)
        if match is not None and match.group("key").strip().lower() not in allowed_keys:
            match = None
        if match is None:
            stripped = line.strip()
            indent = line[: len(line) - len(line.lstrip())]
            if not indent and stripped.startswith("-"):
                root_is_list = True
            if list_context == "data" and stripped.startswith("-"):
                value = stripped[1:].strip().replace("[", "").replace("]", "")
                sanitized_lines.append(f"{indent}- {yaml_quote(value)}")
            elif list_context in scalar_list_keys and stripped.startswith("-"):
                value = stripped[1:].strip()
                sanitized_lines.append(f"{indent}- {yaml_quote(value)}")
            else:
                sanitized_lines.append(line)
            continue

        indent = match.group("indent")
        prefix = match.group("prefix") or ""
        key = match.group("key").strip().lower()
        value = match.group("value").strip()

        if not indent and not prefix and root_is_list and key in scalar_list_keys | {"review_required"}:
            prefix = "- "

        if not indent and prefix:
            root_is_list = True

        if key == "data":
            list_context = "data"
            sanitized_lines.append(f"{indent}{prefix}{key}:")
            continue

        if key in scalar_list_keys:
            list_context = key
            sanitized_lines.append(f"{indent}{prefix}{key}:")
            continue

        list_context = None

        if key in scalar_keys and value:
            value = yaml_quote(value)

        if value:
            sanitized_lines.append(f"{indent}{prefix}{key}: {value}")
        else:
            sanitized_lines.append(f"{indent}{prefix}{key}:")

    return "\n".join(sanitized_lines).strip()


def load_yaml_document(path: str | Path) -> tuple[list[dict[str, Any]], bool]:
    yaml_text = clean_model_response(Path(path).read_text(encoding="utf-8"))

    try:
        data = yaml.safe_load(yaml_text)
    except Exception:
        sanitized = sanitize_yaml_text(yaml_text)
        data = yaml.safe_load(sanitized)

    if data is None:
        return [], False

    if isinstance(data, dict):
        candidate = data.get("records") or data.get("data") or data.get("entries")
        if isinstance(candidate, list):
            data = candidate
        elif any(key in data for key in ("compound", "entries")):
            data = [data]
        elif set(data).issubset({"review_required", "parsing_issues", "extraction_notes"}):
            data = []
        else:
            raise ValueError("Expected the YAML document root to be a list of compound entries")

    if not isinstance(data, list):
        raise ValueError("Expected the YAML document root to be a list of compound entries")

    review_required = any(
        isinstance(item, dict) and bool(item.get("review_required"))
        for item in data
    ) or bool(re.search(r"(?im)^review_required:\s*true\s*$", yaml_text))

    return [item for item in data if isinstance(item, dict)], review_required


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()
