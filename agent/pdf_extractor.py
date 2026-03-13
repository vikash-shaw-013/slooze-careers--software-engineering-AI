"""PDF text extraction using pdfplumber."""

from __future__ import annotations

import io
import pdfplumber


def extract_text_from_path(pdf_path: str) -> tuple[str, int]:
    """Extract all text from a PDF file on disk.

    Returns:
        A tuple of (full_text, page_count).
    """
    text_parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts), page_count


def extract_text_from_bytes(file_bytes: bytes) -> tuple[str, int]:
    """Extract all text from PDF bytes (e.g. from a Streamlit upload).

    Returns:
        A tuple of (full_text, page_count).
    """
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts), page_count
