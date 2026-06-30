"""Knowledge-base HTTP request and response schemas."""

from pydantic import BaseModel, Field


class KnowledgeBaseStatusResponse(BaseModel):
    """Response for GET /api/v1/knowledge-base."""

    ready: bool = Field(description="Whether an index is available for queries")
    document_id: str | None = Field(default=None, description="Active document ID")
    title: str | None = Field(default=None, description="Document title")
    source: str | None = Field(default=None, description="Original filename")
    source_url: str | None = Field(default=None, description="Document source URL")
    chunk_count: int = Field(default=0, description="Number of indexed chunks")
    document_type: str | None = Field(default=None, description="Document type (pdf/txt)")


class UploadResponse(BaseModel):
    """Response after ingesting a knowledge-base document."""

    success: bool = Field(description="Whether ingestion succeeded")
    document_id: str = Field(description="ID of the ingested document")
    title: str = Field(description="Document title")
    chunk_count: int = Field(description="Number of chunks indexed")
    message: str = Field(description="Human-readable status message")
