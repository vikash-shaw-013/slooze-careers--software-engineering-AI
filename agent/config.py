"""Configuration management for the PDF RAG Agent."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for the RAG pipeline."""

    # ── LLM Provider ──────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Gemini model
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # OpenAI model
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Chunking ──────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # ── Embeddings ────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # ── Retrieval ─────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of configuration warnings/errors."""
        issues: list[str] = []
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            issues.append(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        elif cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is not set.")
        return issues
