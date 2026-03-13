"""Text chunking with overlap for RAG ingestion."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split *text* into overlapping chunks using recursive character splitting.

    The splitter tries paragraph → sentence → word boundaries in order,
    falling back as needed to honour *chunk_size*.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks
