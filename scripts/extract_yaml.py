from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from bigmixsoldb.files import build_output_path, collect_input_files, read_text, write_text
from bigmixsoldb.llm import LLMClient, ProviderConfig
from bigmixsoldb.yaml_utils import clean_model_response, load_prompt


def resolve_api_key(provider: str, explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key
    raise SystemExit(f"Missing API key. Set {env_var} or pass --api-key.")


def resolve_api_base(provider: str, explicit_api_base: str | None) -> str | None:
    if explicit_api_base:
        return explicit_api_base

    env_vars = (
        ("GEMINI_API_BASE", "GEMINI_BASE_URL")
        if provider == "gemini"
        else ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_URL")
    )
    for env_var in env_vars:
        api_base = os.environ.get(env_var)
        if api_base:
            return api_base
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YAML from Markdown files with an LLM.")
    parser.add_argument(
        "inputs", nargs="+", help="Markdown files or directories containing Markdown files."
    )
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", help="Base URL for OpenAI-compatible or Gemini API requests.")
    parser.add_argument(
        "--api-key", help="API key. Falls back to OPENAI_API_KEY or GEMINI_API_KEY."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory where YAML files will be written."
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/extract_yaml_prompt.txt",
        help="Prompt file used for Markdown to YAML extraction.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature for LLM generation." \
            " A value of 0.0 typically results in more deterministic" \
            " output, while higher values can produce more varied and" \
            " creative responses. Follow the guidelines for your chosen" \
            " model and provider when selecting a temperature.",
    )
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true", help="Overwrite existing YAML files.")
    args = parser.parse_args()

    inputs = collect_input_files(args.inputs, suffixes={".md", ".markdown"})
    if not inputs:
        raise SystemExit("No Markdown files were found in the provided inputs.")

    prompt = load_prompt(args.prompt_file)
    client = LLMClient(
        ProviderConfig(
            provider=args.provider,
            model=args.model,
            api_key=resolve_api_key(args.provider, args.api_key),
            api_base=resolve_api_base(args.provider, args.api_base),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    )

    for markdown_path in tqdm(inputs, desc="Extract"):
        output_path = build_output_path(markdown_path, args.output_dir, ".yml")
        if output_path.exists() and not args.force:
            continue
        markdown_text = read_text(markdown_path)
        response = client.generate_text(prompt, markdown_text)
        write_text(output_path, clean_model_response(response).strip() + "\n")


if __name__ == "__main__":
    main()
