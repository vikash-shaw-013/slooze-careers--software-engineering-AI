"""LLM abstraction layer supporting Google Gemini and OpenAI."""

from __future__ import annotations

from agent.config import Config


# ── Prompt Templates ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert document analyst. Answer the user's question "
    "using ONLY the provided context excerpts from the document. "
    "If the context does not contain enough information to answer, "
    "say so clearly. Always cite which part of the context supports "
    "your answer. Be thorough yet concise."
)

_QA_TEMPLATE = """Context from the document:
\"\"\"
{context}
\"\"\"

Question: {question}

Provide a detailed, well-structured answer based solely on the context above."""

_SUMMARY_TEMPLATE = """You are given the full text of a document split into chunks.
Produce a comprehensive summary that covers:
1. The main topic and purpose of the document
2. Key findings, arguments, or conclusions
3. Methodology or approach (if applicable)
4. Important details or data points

Document chunks:
\"\"\"
{context}
\"\"\"

Write a clear, well-organized summary in 3-5 paragraphs."""


# ── LLM Client ───────────────────────────────────────────────────

class LLMClient:
    """Unified interface for LLM generation."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self._client = None
        self._model_name: str = ""
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self._model_name = Config.GEMINI_MODEL
            self._client = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=_SYSTEM_PROMPT,
            )
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self._model_name = Config.OPENAI_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # ── Public API ────────────────────────────────────────────────

    def answer(self, question: str, context: str) -> str:
        """Generate an answer grounded in *context*."""
        prompt = _QA_TEMPLATE.format(context=context, question=question)
        return self._generate(prompt)

    def summarize(self, context: str) -> str:
        """Generate a summary of the document from its chunks."""
        prompt = _SUMMARY_TEMPLATE.format(context=context)
        return self._generate(prompt)

    # ── Private ───────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        if self.provider == "gemini":
            response = self._client.generate_content(prompt)
            return response.text

        # OpenAI
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model_name
