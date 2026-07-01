"""Shared state passed between graph nodes."""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from rag_chatbot.rag.types import RetrievedChunk


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    current_question: str

    intent: Literal["casual_chat", "document_query", "unsupported_action"]
    intent_confidence: float
    intent_fast_path: bool

    standalone_query: str
    documents: list[RetrievedChunk]
    context_sufficient: bool

    retrieval_attempts: int
    reranker_used: bool
    knowledge_base_id: str
    sources: list[RetrievedChunk]
