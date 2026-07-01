"""LangGraph node implementations."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from rag_chatbot.config import settings
from rag_chatbot.graph.prompts import (
    ANSWER_GENERATOR_PROMPT,
    CASUAL_RESPONDER_PROMPT,
    CONTEXT_GRADER_PROMPT,
    CONTEXTUALIZER_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    QUERY_REWRITER_PROMPT,
)
from rag_chatbot.graph.state import GraphState
from rag_chatbot.rag.retriever import retrieve as retrieve_chunks

logger = logging.getLogger(__name__)

chat_model = ChatGoogleGenerativeAI(
    model=settings.GEMINI_CHAT_MODEL,
    google_api_key=settings.GEMINI_API_KEY.get_secret_value(),
    temperature=0.2,
)


class IntentDecision(BaseModel):
    intent: Literal["casual_chat", "document_query", "unsupported_action"] = Field(
        description="The classified intent of the user's message."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reason: str = Field(description="Brief reasoning for the classification.")


def classify_obvious_casual_message(text: str) -> str | None:
    normalized = text.strip().lower()
    casual_exact_matches = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "ok",
        "okay",
    }
    if normalized in casual_exact_matches:
        return "casual_chat"
    return None


async def classify_intent(state: GraphState) -> dict[str, object]:
    question = state["current_question"]

    obvious_intent = classify_obvious_casual_message(question)
    if obvious_intent is not None:
        logger.info("Intent fast-path matched: %s", obvious_intent)
        return {"intent": obvious_intent, "intent_fast_path": True}

    logger.info("Classifying intent via LLM...")
    try:
        structured_llm = chat_model.with_structured_output(IntentDecision)

        recent_history = state.get("messages", [])[-settings.MAX_HISTORY_MESSAGES : -1]
        history_text = "\n".join(f"{message.type}: {message.content}" for message in recent_history)

        prompt = (
            f"{INTENT_CLASSIFIER_PROMPT}\n\n"
            f"History:\n{history_text}\n\n"
            f"Current Message:\n{question}"
        )

        decision = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        if decision.confidence < settings.INTENT_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Intent confidence %.2f is too low; using document_query",
                decision.confidence,
            )
            return {"intent": "document_query", "intent_fast_path": False}

        logger.info("Intent classified: %s (%.2f)", decision.intent, decision.confidence)
        return {
            "intent": decision.intent,
            "intent_confidence": decision.confidence,
            "intent_fast_path": False,
        }

    except Exception as exc:
        logger.error("Intent classification failed, falling back to document_query: %s", exc)
        return {"intent": "document_query", "intent_fast_path": False}


async def respond_casually(state: GraphState) -> dict[str, object]:
    logger.info("Generating casual response...")
    recent_history = state.get("messages", [])[-settings.MAX_HISTORY_MESSAGES :]

    messages = [
        {"role": "system", "content": CASUAL_RESPONDER_PROMPT},
        *recent_history,
    ]

    response = await chat_model.ainvoke(messages)
    return {
        "messages": [response],
        "retrieval_attempts": 0,
        "reranker_used": False,
        "sources": [],
    }


async def respond_with_scope(state: GraphState) -> dict[str, object]:
    logger.info("Generating unsupported scope response...")
    msg = (
        "I can chat with you and answer questions using the active document. "
        "I cannot browse the web, run code, or perform external actions, but "
        "I can explain, summarize, and find information in the document."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "retrieval_attempts": 0,
        "reranker_used": False,
        "sources": [],
    }


async def contextualize(state: GraphState) -> dict[str, object]:
    question = state["current_question"]
    history = state.get("messages", [])

    if len(history) <= 1:
        return {"standalone_query": question}

    logger.info("Contextualizing query...")
    recent_history = history[-settings.MAX_HISTORY_MESSAGES : -1]
    history_text = "\n".join(f"{message.type}: {message.content}" for message in recent_history)

    prompt = f"{CONTEXTUALIZER_PROMPT}\n\nHistory:\n{history_text}\n\nQuestion:\n{question}"
    response = await chat_model.ainvoke([HumanMessage(content=prompt)])

    standalone = response.content.strip()
    logger.info("Contextualized query created")
    return {"standalone_query": standalone}


async def retrieve(state: GraphState, *, vector_store) -> dict[str, object]:
    query = state.get("standalone_query", state["current_question"])
    attempts = state.get("retrieval_attempts", 0)

    logger.info("Retrieving chunks (attempt %d)", attempts + 1)
    chunks = await retrieve_chunks(
        query,
        vector_store=vector_store,
        k=settings.RETRIEVAL_K,
    )

    return {
        "documents": chunks,
        "retrieval_attempts": attempts + 1,
        "knowledge_base_id": (
            vector_store.manifest.document_id if vector_store.manifest else "unknown"
        ),
    }


async def rerank(state: GraphState, *, reranker) -> dict[str, object]:
    query = state.get("standalone_query", state["current_question"])
    docs = state.get("documents", [])

    if not docs:
        return {"documents": [], "reranker_used": False}

    logger.info("Reranking %d chunks...", len(docs))
    reranked_docs = await reranker.rerank(query, docs)
    reranked_docs = reranked_docs[: settings.RERANK_TOP_N]

    return {
        "documents": reranked_docs,
        "reranker_used": any(doc.rerank_score is not None for doc in reranked_docs),
    }


class GradeDecision(BaseModel):
    sufficient: bool = Field(description="True if the context helps answer the question")
    reason: str = Field(description="Reasoning for the grade")


async def grade_context(state: GraphState) -> dict[str, object]:
    query = state.get("standalone_query", state["current_question"])
    docs = state.get("documents", [])

    if not docs:
        return {"context_sufficient": False}

    logger.info("Grading %d chunks for relevance...", len(docs))
    structured_llm = chat_model.with_structured_output(GradeDecision)

    context_text = "\n\n".join(doc.content for doc in docs)
    prompt = f"{CONTEXT_GRADER_PROMPT}\n\nQuestion:\n{query}\n\nContext:\n{context_text}"

    try:
        decision = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("Context sufficient: %s", decision.sufficient)
        return {"context_sufficient": decision.sufficient}
    except Exception as exc:
        logger.error("Grading failed, assuming insufficient: %s", exc)
        return {"context_sufficient": False}


async def rewrite_query(state: GraphState) -> dict[str, object]:
    query = state.get("standalone_query", state["current_question"])
    logger.info("Rewriting retrieval query")

    prompt = f"{QUERY_REWRITER_PROMPT}\n\nOriginal Question:\n{query}"
    response = await chat_model.ainvoke([HumanMessage(content=prompt)])

    rewritten = response.content.strip()
    return {"standalone_query": rewritten}


async def generate_answer(state: GraphState) -> dict[str, object]:
    query = state.get("standalone_query", state["current_question"])
    docs = state.get("documents", [])

    logger.info("Generating grounded answer from %d chunks...", len(docs))

    context_blocks = []
    for index, doc in enumerate(docs, start=1):
        context_blocks.append(f"ID: [S{index}]\nContent: {doc.content}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"{ANSWER_GENERATOR_PROMPT}\n\nQuestion:\n{query}"
    prompt = prompt.replace("{context}", context_text)

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])

    return {
        "messages": [response],
        "sources": docs,
    }


async def abstain(state: GraphState) -> dict[str, object]:
    logger.info("Abstaining (retries exhausted)...")
    msg = "I'm sorry, but I cannot find information in the current document to answer that."
    return {
        "messages": [AIMessage(content=msg)],
        "sources": [],
    }
