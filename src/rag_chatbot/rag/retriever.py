"""Relevant-document retrieval."""

from __future__ import annotations

import logging

from rag_chatbot.rag.types import ChunkMetadata, RetrievedChunk
from rag_chatbot.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    *,
    vector_store: VectorStoreManager,
    k: int,
) -> list[RetrievedChunk]:
    """Retrieve top-k candidate chunks from the active Chroma collection."""
    collection = vector_store.get_collection()

    results = collection.similarity_search_with_relevance_scores(
        query=query,
        k=k,
    )

    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                metadata=ChunkMetadata(
                    document_id=meta.get("document_id", ""),
                    title=meta.get("title", ""),
                    source=meta.get("source", ""),
                    source_url=meta.get("source_url", ""),
                    page_number=meta.get("page_number", 0),
                    page_label=meta.get("page_label", ""),
                    section=meta.get("section", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    content_hash=meta.get("content_hash", ""),
                    document_type=meta.get("document_type", ""),
                ),
                retrieval_score=score,
            )
        )

    logger.info("Retrieved %d chunks for query: %.80s", len(chunks), query)
    return chunks
