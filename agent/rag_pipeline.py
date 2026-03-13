"""RAG Pipeline — orchestrates PDF ingestion, retrieval, and generation."""

from __future__ import annotations

from agent.config import Config
from agent.pdf_extractor import extract_text_from_bytes, extract_text_from_path
from agent.chunker import chunk_text
from agent.embeddings import EmbeddingModel
from agent.vector_store import VectorStore
from agent.llm import LLMClient


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Typical usage::

        pipeline = RAGPipeline()
        pipeline.ingest_bytes(uploaded_file.read())
        summary = pipeline.summarize()
        answer  = pipeline.query("What methodology was used?")
    """

    def __init__(self) -> None:
        self._embedding_model = EmbeddingModel(Config.EMBEDDING_MODEL)
        self._vector_store = VectorStore(self._embedding_model.dimension)
        self._llm = LLMClient()
        self._chunks: list[str] = []
        self._full_text: str = ""
        self._page_count: int = 0
        self._is_ingested: bool = False

    # ── Ingestion ─────────────────────────────────────────────────

    def ingest_bytes(self, file_bytes: bytes) -> dict:
        """Ingest a PDF from raw bytes (Streamlit upload).

        Returns:
            A dict with ingestion statistics.
        """
        self._vector_store.reset()
        self._full_text, self._page_count = extract_text_from_bytes(
            file_bytes
        )
        return self._process_text()

    def ingest_path(self, pdf_path: str) -> dict:
        """Ingest a PDF from a file path.

        Returns:
            A dict with ingestion statistics.
        """
        self._vector_store.reset()
        self._full_text, self._page_count = extract_text_from_path(pdf_path)
        return self._process_text()

    def _process_text(self) -> dict:
        """Chunk → embed → store."""
        if not self._full_text.strip():
            raise ValueError(
                "No text could be extracted from the PDF. "
                "The file may be image-based or empty."
            )

        self._chunks = chunk_text(
            self._full_text,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        embeddings = self._embedding_model.encode(self._chunks)
        self._vector_store.add_documents(self._chunks, embeddings)
        self._is_ingested = True

        return {
            "pages": self._page_count,
            "characters": len(self._full_text),
            "chunks": len(self._chunks),
        }

    # ── Query ─────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """Retrieve relevant chunks and generate an answer.

        Returns:
            A dict with ``answer`` and ``sources`` (the retrieved chunks).
        """
        self._ensure_ingested()

        query_emb = self._embedding_model.encode_query(question)
        results = self._vector_store.search(query_emb, top_k=Config.TOP_K)

        context = "\n\n---\n\n".join(r["chunk"] for r in results)
        answer = self._llm.answer(question, context)

        return {
            "answer": answer,
            "sources": results,
        }

    # ── Summarization ─────────────────────────────────────────────

    def summarize(self) -> str:
        """Generate a comprehensive summary of the ingested document."""
        self._ensure_ingested()

        # Use a selection of chunks that covers the document evenly
        if len(self._chunks) <= 15:
            selected = self._chunks
        else:
            # Take first 5, middle 5, last 5 for broad coverage
            n = len(self._chunks)
            mid = n // 2
            selected = (
                self._chunks[:5]
                + self._chunks[mid - 2 : mid + 3]
                + self._chunks[-5:]
            )

        context = "\n\n---\n\n".join(selected)
        return self._llm.summarize(context)

    # ── Helpers ───────────────────────────────────────────────────

    def _ensure_ingested(self) -> None:
        if not self._is_ingested:
            raise RuntimeError(
                "No document has been ingested yet. "
                "Call ingest_bytes() or ingest_path() first."
            )

    @property
    def is_ready(self) -> bool:
        return self._is_ingested

    @property
    def document_stats(self) -> dict:
        return {
            "pages": self._page_count,
            "characters": len(self._full_text),
            "chunks": len(self._chunks),
            "model": self._llm.model_name,
        }
