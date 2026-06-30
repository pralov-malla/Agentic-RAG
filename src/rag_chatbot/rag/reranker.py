"""Cohere reranking with vector-search fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import cohere

from rag_chatbot.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        ...


class NoOpReranker(Reranker):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        _ = query
        return chunks


class CohereReranker(Reranker):
    def __init__(
        self,
        api_key: str,
        model: str,
        top_n: int,
    ) -> None:
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model
        self._top_n = top_n

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks or not query.strip():
            return chunks

        documents = [chunk.content for chunk in chunks]
        top_n = min(max(1, self._top_n), len(documents))

        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        except Exception as exc:
            logger.warning(
                "Cohere reranking failed; falling back to retrieval order. Error: %s",
                exc,
            )
            return chunks

        if not response.results:
            logger.warning(
                "Cohere reranking returned no results; falling back to retrieval order."
            )
            return chunks

        reranked: list[RetrievedChunk] = []
        for result in response.results:
            idx = result.index
            if not isinstance(idx, int) or idx < 0 or idx >= len(chunks):
                continue

            chunk = chunks[idx]
            chunk.rerank_score = result.relevance_score
            reranked.append(chunk)

        if not reranked:
            logger.warning(
                "Cohere reranking returned no usable results; falling back to retrieval order."
            )
            return chunks

        return reranked


@lru_cache(maxsize=1)
def build_reranker() -> Reranker:
    from rag_chatbot.config import settings

    if not settings.has_cohere:
        logger.info("No COHERE_API_KEY; using NoOpReranker.")
        return NoOpReranker()

    return CohereReranker(
        api_key=settings.COHERE_API_KEY.get_secret_value(),
        model=settings.COHERE_RERANK_MODEL,
        top_n=settings.RERANK_TOP_N,
    )
