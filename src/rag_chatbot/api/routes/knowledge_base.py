"""Knowledge-base status and ingestion endpoints."""

from fastapi import APIRouter, UploadFile

from rag_chatbot.api.dependencies import KBService
from rag_chatbot.api.schemas.knowledge_base import (
    KnowledgeBaseStatusResponse,
    UploadResponse,
)

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])


@router.get("", response_model=KnowledgeBaseStatusResponse)
async def get_status(kb: KBService) -> KnowledgeBaseStatusResponse:
    """Return the current knowledge-base status."""
    return kb.get_status()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    kb: KBService,
) -> UploadResponse:
    """Upload a PDF or TXT document to replace the active knowledge base."""
    content = await file.read()
    return kb.upload(content, file.filename or "untitled")


@router.post("/default", response_model=UploadResponse)
async def ingest_default_document(kb: KBService) -> UploadResponse:
    """Ingest the bundled default document and make it active."""
    return kb.ingest_default()
