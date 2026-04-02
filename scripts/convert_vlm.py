from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from bigmixsoldb.files import build_output_path, collect_input_files
from bigmixsoldb.llm import LLMClient, ProviderConfig
from bigmixsoldb.vlm_parser import parse_image_with_vlm, parse_pdf_with_vlm
from bigmixsoldb.yaml_utils import load_prompt


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
    parser = argparse.ArgumentParser(description="Parse PDFs or images to Markdown with a VLM.")
    parser.add_argument("inputs", nargs="+", help="PDF or image files, or directories containing them.")
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", required=True, help="Model name for the selected provider.")
    parser.add_argument("--api-base", help="Base URL for OpenAI-compatible or Gemini API requests.")
    parser.add_argument("--api-key", help="API key. Falls back to OPENAI_API_KEY or GEMINI_API_KEY.")
    parser.add_argument("--output-dir", required=True, help="Directory where Markdown files will be written.")
    parser.add_argument("--prompt-file", help="Prompt file used for page transcription.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--scale", type=float, default=2.0, help="PDF rendering scale for page images.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Markdown files.")
    args = parser.parse_args()

    inputs = collect_input_files(args.inputs, suffixes={".pdf", ".png", ".jpg", ".jpeg", ".webp"})
    if not inputs:
        raise SystemExit("No PDF or image files were found in the provided inputs.")

    prompt = load_prompt(args.prompt_file) if args.prompt_file else ""
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

    for input_path in tqdm(inputs, desc="VLM"):
        output_path = build_output_path(input_path, args.output_dir, ".md")
        if output_path.exists() and not args.force:
            continue
        if input_path.suffix.lower() == ".pdf":
            parse_pdf_with_vlm(input_path, output_path, client, prompt, scale=args.scale)
        else:
            parse_image_with_vlm(input_path, output_path, client, prompt)


if __name__ == "__main__":
    main()
