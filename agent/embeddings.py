"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a sentence-transformers model for encoding text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into normalised float32 embeddings.

        Args:
            texts: Strings to embed.
            batch_size: Batch size for the model.

        Returns:
            A numpy array of shape ``(len(texts), dimension)``.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Returns:
            A numpy array of shape ``(1, dimension)``.
        """
        return self.encode([query])
