"""Chat and conversation endpoints."""

from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import Response

from rag_chatbot.api.dependencies import CurrentChatService
from rag_chatbot.api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    chat_service: CurrentChatService,
) -> ChatResponse:
    """Send a message to the chatbot."""
    result = await chat_service.chat(
        message=request.message,
        thread_id=str(request.thread_id),
    )
    return ChatResponse(**result)


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: UUID,
    chat_service: CurrentChatService,
) -> Response:
    """Clear conversation memory for a thread."""
    await chat_service.delete_thread(str(thread_id))
    return Response(status_code=204)
