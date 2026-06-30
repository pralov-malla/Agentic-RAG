"""Embedding model wrapper and factory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if TYPE_CHECKING:
    from rag_chatbot.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService(Embeddings):
    """Wrapper around Gemini embeddings with logging."""

    def __init__(self, settings: Settings):
        self._dimensions = settings.EMBEDDING_DIMENSIONS
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.GEMINI_EMBEDDING_MODEL}",
            google_api_key=settings.GEMINI_API_KEY.get_secret_value(),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Embedding %d documents", len(texts))
        return self._embeddings.embed_documents(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        logger.debug("Embedding query (%d characters)", len(text))
        return self._embeddings.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._dimensions,
        )


def build_embeddings(settings: Settings) -> EmbeddingService:
    """Create Gemini embeddings from app settings."""
    return EmbeddingService(settings)
