from __future__ import annotations

from pathlib import Path
from typing import Literal

def clean_repeated_sequences(text, threshold=4, max_seq_len=32):
    """Remove repeated sequences in text that are likely due to OCR errors.
    Args:
        text: The input text to clean.
        threshold: Minimum number of repetitions to consider a sequence as repeated.
        max_seq_len: Maximum length of the sequence to check for repetitions.
    Returns:
        Cleaned text with repeated sequences removed.
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        found = False
        # Try all possible sequence lengths from largest to smallest
        for k in range(min(max_seq_len, n - i), 0, -1):
            pattern = text[i : i + k]
            # Check how many times this pattern repeats starting at i
            m = 1
            while (i + m * k <= n) and (text[i : i + m * k] == pattern * m):
                m += 1
            m -= 1  # now m is the actual count of repetitions
            if m >= threshold:
                # Found a repeated pattern of length k, repeated m times
                result.append(pattern)
                i += m * k
                found = True
                break
        if not found:
            result.append(text[i])
            i += 1
    return "".join(result)

def convert_pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    repetition_threshold: int = 4,
    repetition_max_seq_len: int = 32,
    use_tesseract: bool = False,
    tableformer_mode: Literal["accurate", "fast"] = "accurate",
) -> Path:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_file = Path(pdf_path)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 4.0
    pipeline_options.do_ocr = True
    if use_tesseract:
        pipeline_options.ocr_options = TesseractCliOcrOptions()
        pipeline_options.ocr_options.lang = ["eng"]
    else:
        pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.ocr_options.force_full_page_ocr = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if tableformer_mode == "accurate" else TableFormerMode.FAST
    )
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.AUTO,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(pdf_file))
    doc = result.document

    # Avoid repeated sequences in formulas due to OCR errors
    for txt in doc.texts:
        if txt.label == "formula":
            txt.text = clean_repeated_sequences(
                txt.text,
                threshold=repetition_threshold,
                max_seq_len=repetition_max_seq_len,
            )

    markdown = result.document.export_to_markdown(image_placeholder="")
    target_path.write_text(markdown, encoding="utf-8")
    return target_path
