from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium

from bigmixsoldb.llm import LLMClient
from bigmixsoldb.yaml_utils import strip_code_fences


def render_pdf_pages(pdf_path: str | Path, scale: float = 2.0) -> list[tuple[int, bytes, str]]:
    pdf_file = Path(pdf_path)
    document = pdfium.PdfDocument(str(pdf_file))
    pages: list[tuple[int, bytes, str]] = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append((page_index + 1, buffer.getvalue(), "image/png"))
    finally:
        document.close()
    return pages


def read_image_bytes(image_path: str | Path) -> tuple[bytes, str]:
    path = Path(image_path)
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    return path.read_bytes(), mime_type


def parse_pdf_with_vlm(
    pdf_path: str | Path,
    output_path: str | Path,
    client: LLMClient,
    prompt: str,
    *,
    scale: float = 2.0,
) -> Path:
    pages = render_pdf_pages(pdf_path, scale=scale)
    page_markdown: list[str] = []
    for page_number, image_bytes, mime_type in pages:
        markdown = client.generate_from_image(
            system_prompt=prompt,
            user_text=f"Transcribe page {page_number} to Markdown.",
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
        cleaned = strip_code_fences(markdown).strip()
        page_markdown.append(f"<!-- page {page_number} -->\n\n{cleaned}")

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n\n".join(page_markdown).strip() + "\n", encoding="utf-8")
    return target_path


def parse_image_with_vlm(
    image_path: str | Path,
    output_path: str | Path,
    client: LLMClient,
    prompt: str,
) -> Path:
    image_bytes, mime_type = read_image_bytes(image_path)
    markdown = client.generate_from_image(
        system_prompt=prompt,
        user_text="Transcribe this page or figure to Markdown.",
        image_bytes=image_bytes,
        mime_type=mime_type,
    )
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(strip_code_fences(markdown).strip() + "\n", encoding="utf-8")
    return target_path
