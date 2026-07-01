"""Chat API schemas."""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's message content.")
    thread_id: UUID = Field(
        default_factory=uuid4,
        description="Conversation thread ID.",
    )


class ChatResponse(BaseModel):
    thread_id: str = Field(description="Conversation thread ID")
    answer: str = Field(description="The chatbot's response")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents cited in the answer",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal execution metadata",
    )
