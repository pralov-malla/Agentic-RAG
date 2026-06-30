"""Vector-store construction and persistence."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_chatbot.rag.types import IndexManifest

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage Chroma collections through the separate ChromaDB service."""

    def __init__(
        self,
        chroma_host: str,
        chroma_port: int,
        chroma_ssl: bool,
        manifest_path: Path,
        embeddings: Embeddings,
    ) -> None:
        self._client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            ssl=chroma_ssl,
        )
        self._embeddings = embeddings
        self._manifest_path = manifest_path
        self._active_collection: Chroma | None = None
        self._manifest: IndexManifest | None = None
        self._load_manifest()

    @property
    def manifest(self) -> IndexManifest | None:
        return self._manifest

    @property
    def is_ready(self) -> bool:
        if self._manifest is None:
            return False
        try:
            return self.get_active_chunk_count() == self._manifest.chunk_count
        except Exception:
            return False

    def get_active_chunk_count(self) -> int:
        """Return the actual number of chunks stored in the active collection."""
        if self._manifest is None:
            return 0
        return self._client.get_collection(self._manifest.collection_name).count()

    def get_collection(self) -> Chroma:
        """Return the active Chroma collection."""
        if self._active_collection is None:
            if self._manifest is None:
                raise RuntimeError("No active index. Ingest a document first.")
            self._active_collection = Chroma(
                client=self._client,
                collection_name=self._manifest.collection_name,
                embedding_function=self._embeddings,
                create_collection_if_not_exists=False,
            )
        return self._active_collection

    def create_index(
        self,
        chunks: list[Document],
        *,
        document_id: str,
        title: str,
        source_url: str,
        document_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> IndexManifest:
        """Build a versioned collection and switch the active manifest."""
        if not chunks:
            raise ValueError("Cannot create an index without document chunks.")

        previous_collection_name = (
            self._manifest.collection_name if self._manifest is not None else None
        )
        collection_name = f"idx_{document_id}_{time.time_ns()}"
        logger.info("Creating index '%s' with %d chunks", collection_name, len(chunks))

        collection = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            client=self._client,
            collection_name=collection_name,
        )

        count = self._client.get_collection(collection_name).count()
        if count != len(chunks):
            raise RuntimeError(
                f"Index verification failed: expected {len(chunks)} chunks, got {count}"
            )

        manifest = IndexManifest(
            document_id=document_id,
            title=title,
            source_url=source_url,
            collection_name=collection_name,
            chunk_count=count,
            document_type=document_type,
            metadata=metadata or {},
        )
        self._save_manifest(manifest)
        self._manifest = manifest
        self._active_collection = collection

        logger.info("Index '%s' active: %d chunks indexed", collection_name, count)
        if previous_collection_name and previous_collection_name != collection_name:
            self._delete_collection(previous_collection_name)
        return manifest

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            return

        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            self._manifest = IndexManifest(**data)
            logger.info("Loaded manifest: %s", self._manifest.collection_name)
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Failed to load index manifest; starting fresh.")
            self._manifest = None

    def _save_manifest(self, manifest: IndexManifest) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2),
            encoding="utf-8",
        )
        logger.info("Saved manifest: %s", manifest.collection_name)

    def _delete_collection(self, collection_name: str) -> None:
        """Delete an inactive collection without failing successful ingestion."""
        try:
            self._client.delete_collection(collection_name)
            logger.info("Deleted previous collection: %s", collection_name)
        except Exception:
            logger.warning(
                "Could not delete previous collection: %s",
                collection_name,
            )
