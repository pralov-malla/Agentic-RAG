"""Retrieval and index metadata types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkMetadata:
    """Metadata attached to every indexed chunk."""

    document_id: str
    title: str
    source: str
    source_url: str
    page_number: int
    page_label: str
    section: str
    chunk_index: int
    content_hash: str
    document_type: str


@dataclass
class RetrievedChunk:
    """A single chunk returned by retrieval and optionally reranked."""

    content: str
    metadata: ChunkMetadata
    retrieval_score: float
    rerank_score: float | None = None

    @property
    def source_label(self) -> str:
        return f"p.{self.metadata.page_label}"


@dataclass
class IndexManifest:
    """Active knowledge-base index state."""

    document_id: str
    title: str
    source_url: str
    collection_name: str
    chunk_count: int
    document_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
