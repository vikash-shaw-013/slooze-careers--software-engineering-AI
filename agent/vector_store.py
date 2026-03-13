"""FAISS-backed vector store for document chunk retrieval."""

from __future__ import annotations

import faiss
import numpy as np


class VectorStore:
    """In-memory FAISS index that maps embeddings back to text chunks."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        # Inner-product index (works like cosine similarity with normalised vecs)
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[str] = []

    # ── Mutation ──────────────────────────────────────────────────

    def add_documents(
        self, chunks: list[str], embeddings: np.ndarray
    ) -> None:
        """Add document chunks and their embeddings to the index.

        Args:
            chunks: Text chunks corresponding to each embedding row.
            embeddings: Float32 array of shape ``(n, dimension)``.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs "
                f"{embeddings.shape[0]} embeddings."
            )
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def reset(self) -> None:
        """Clear the index and all stored chunks."""
        self._index.reset()
        self._chunks.clear()

    # ── Query ─────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[dict]:
        """Return the *top_k* most similar chunks.

        Args:
            query_embedding: Float32 array of shape ``(1, dimension)``.
            top_k: Number of results to return.

        Returns:
            A list of dicts with keys ``chunk``, ``score``, ``index``.
        """
        if self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "chunk": self._chunks[idx],
                    "score": float(score),
                    "index": int(idx),
                }
            )
        return results

    # ── Info ──────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently stored."""
        return self._index.ntotal
