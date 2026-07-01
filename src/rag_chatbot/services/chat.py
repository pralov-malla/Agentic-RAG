"""Chat and conversational workflow service."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from rag_chatbot.config import Settings
from rag_chatbot.core.exceptions import APIException, ErrorCode

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, graph: CompiledStateGraph, settings: Settings):
        self._graph = graph
        self._settings = settings

    async def chat(self, message: str, thread_id: str) -> dict[str, Any]:
        logger.info("Processing chat for thread %s", thread_id)

        if not message.strip():
            raise APIException(
                message="Message cannot be empty.",
                error_code=ErrorCode.VALIDATION_ERROR,
                status_code=422,
            )

        config = {"configurable": {"thread_id": thread_id}}

        input_state = {
            "messages": [HumanMessage(content=message)],
            "current_question": message,
            "intent_confidence": 0.0,
            "intent_fast_path": False,
            "standalone_query": "",
            "documents": [],
            "context_sufficient": False,
            "retrieval_attempts": 0,
            "reranker_used": False,
            "knowledge_base_id": "",
            "sources": [],
        }

        try:
            result = await self._graph.ainvoke(input_state, config=config)

            final_messages = result.get("messages", [])
            answer = final_messages[-1].content if final_messages else "No response generated."

            sources = result.get("sources", [])
            sources_dict = [
                {
                    "id": f"S{index}",
                    "title": source.metadata.title,
                    "page": source.metadata.page_label,
                    "section": source.metadata.section,
                    "source_url": source.metadata.source_url,
                }
                for index, source in enumerate(sources, start=1)
            ]

            meta = {
                "intent": result.get("intent", "unknown"),
                "standalone_query": result.get("standalone_query"),
                "retrieval_attempts": result.get("retrieval_attempts", 0),
                "reranker_used": result.get("reranker_used", False),
                "knowledge_base_id": result.get("knowledge_base_id"),
            }

            return {
                "thread_id": thread_id,
                "answer": answer,
                "sources": sources_dict,
                "meta": meta,
            }

        except Exception as exc:
            logger.exception("Graph execution failed")
            raise APIException(
                message="An internal error occurred during chat processing.",
                error_code=ErrorCode.INTERNAL_ERROR,
                status_code=500,
            ) from exc

    async def delete_thread(self, thread_id: str) -> None:
        logger.info("Deleting thread %s", thread_id)
        try:
            if self._graph.checkpointer is not None:
                await self._graph.checkpointer.adelete_thread(thread_id)
        except Exception as exc:
            logger.warning("Could not delete thread %s: %s", thread_id, exc)
