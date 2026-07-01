"""FastAPI dependency-injection helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from rag_chatbot.services.chat import ChatService
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


def _get_chat_service(request: Request) -> ChatService:
    service: ChatService | None = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The chat service is unavailable. Please retry.",
        )
    return service


CurrentChatService = Annotated[ChatService, Depends(_get_chat_service)]
