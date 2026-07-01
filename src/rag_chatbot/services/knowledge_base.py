"""Knowledge-base ingestion and index replacement orchestration."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from rag_chatbot.api.schemas.knowledge_base import (
    KnowledgeBaseStatusResponse,
    UploadResponse,
)
from rag_chatbot.config import Settings
from rag_chatbot.core.exceptions import APIException, ErrorCode
from rag_chatbot.rag.ingestion import (
    chunk_documents,
    generate_document_id,
    load_document,
)
from rag_chatbot.rag.types import IndexManifest
from rag_chatbot.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Orchestrates document ingestion, indexing, and knowledge-base lifecycle."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        settings: Settings,
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings

    @property
    def manifest(self) -> IndexManifest | None:
        return self._vector_store.manifest

    def get_status(self) -> KnowledgeBaseStatusResponse:
        """Return status with a live chunk count from ChromaDB."""
        manifest = self._vector_store.manifest
        if manifest is None:
            return KnowledgeBaseStatusResponse(ready=False)

        try:
            chunk_count = self._vector_store.get_active_chunk_count()
        except Exception:
            logger.warning("Active ChromaDB collection is unavailable.")
            chunk_count = 0

        return KnowledgeBaseStatusResponse(
            ready=chunk_count > 0 and chunk_count == manifest.chunk_count,
            document_id=manifest.document_id,
            title=manifest.title,
            source=manifest.metadata.get("source", ""),
            source_url=manifest.source_url,
            chunk_count=chunk_count,
            document_type=manifest.document_type,
        )

    def upload(self, file_content: bytes, filename: str) -> UploadResponse:
        """Ingest an uploaded document, replacing the active knowledge base."""
        safe_filename = Path(filename.replace("\\", "/")).name
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            raise APIException(
                status_code=422,
                message=f"Unsupported file type: {suffix}. Only PDF and TXT are supported.",
                error_code=ErrorCode.INVALID_DOCUMENT,
            )

        if len(file_content) > self._settings.max_upload_bytes:
            max_mb = self._settings.MAX_UPLOAD_MB
            raise APIException(
                status_code=422,
                message=f"File too large. Maximum size is {max_mb}MB.",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)

        try:
            return self._ingest_file(
                tmp_path,
                safe_filename,
                document_type=suffix.lstrip("."),
                source_name=safe_filename,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def ingest_default(self) -> UploadResponse:
        """Ingest the bundled default document and make it active."""
        default_path = self._settings.DEFAULT_DOCUMENT_PATH
        if not default_path.exists():
            raise APIException(
                status_code=500,
                message=f"Default document not found: {default_path.name}",
                error_code=ErrorCode.INTERNAL_ERROR,
            )

        return self._ingest_file(
            default_path,
            default_path.name,
            document_type=default_path.suffix.lstrip("."),
            source_name=default_path.name,
            source_url=self._settings.DEFAULT_DOCUMENT_URL,
        )

    def _ingest_file(
        self,
        path: Path,
        title: str,
        document_type: str,
        source_name: str,
        source_url: str = "",
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> UploadResponse:
        """Load, chunk, and index a document file."""
        document_id = generate_document_id(path)

        logger.info("Ingesting document: %s (id=%s)", title, document_id)

        pages = load_document(path, start_page=start_page, end_page=end_page)
        for page in pages:
            page.metadata["source"] = source_name

        chunks = chunk_documents(
            pages,
            chunk_size=self._settings.CHUNK_SIZE,
            chunk_overlap=self._settings.CHUNK_OVERLAP,
            document_id=document_id,
            title=title,
            source_url=source_url,
            document_type=document_type,
        )

        manifest = self._vector_store.create_index(
            chunks,
            document_id=document_id,
            title=title,
            source_url=source_url,
            document_type=document_type,
            metadata={"source": source_name},
        )

        logger.info("Ingestion complete: %s (%d chunks)", title, manifest.chunk_count)

        return UploadResponse(
            success=True,
            document_id=manifest.document_id,
            title=manifest.title,
            chunk_count=manifest.chunk_count,
            message=f"Successfully indexed '{title}' with {manifest.chunk_count} chunks.",
        )
