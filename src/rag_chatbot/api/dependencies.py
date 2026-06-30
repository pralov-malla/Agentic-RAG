"""FastAPI dependency-injection helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from rag_chatbot.services.knowledge_base import KnowledgeBaseService


def _get_knowledge_base_service(
    request: Request,
) -> KnowledgeBaseService:
    service: KnowledgeBaseService | None = getattr(
        request.app.state, "knowledge_base_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The service is unavailable. Please retry.",
        )
    return service


KBService = Annotated[KnowledgeBaseService, Depends(_get_knowledge_base_service)]
