from __future__ import annotations

import argparse
from tqdm import tqdm

from bigmixsoldb.docling_parser import convert_pdf_to_markdown
from bigmixsoldb.files import build_output_path, collect_input_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDF files to Markdown with Docling.")
    parser.add_argument("inputs", nargs="+", help="PDF files or directories containing PDFs.")
    parser.add_argument("--output-dir", required=True, help="Directory where Markdown files will be written.")
    parser.add_argument(
        "--tableformer-mode",
        choices=["accurate", "fast"],
        default="accurate",
        help="Docling table structure mode.",
    )
    parser.add_argument("--use-tesseract", action="store_true", help="Use the Tesseract OCR backend.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Markdown files.")
    args = parser.parse_args()

    inputs = collect_input_files(args.inputs, suffixes={".pdf"})
    if not inputs:
        raise SystemExit("No PDF files were found in the provided inputs.")

    for pdf_path in tqdm(inputs, desc="Docling"):
        output_path = build_output_path(pdf_path, args.output_dir, ".md")
        if output_path.exists() and not args.force:
            continue
        convert_pdf_to_markdown(
            pdf_path,
            output_path,
            use_tesseract=args.use_tesseract,
            tableformer_mode=args.tableformer_mode,
        )


if __name__ == "__main__":
    main()
