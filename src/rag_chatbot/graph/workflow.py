"""LangGraph construction and compilation."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from rag_chatbot.config import settings
from rag_chatbot.graph.nodes import (
    abstain,
    classify_intent,
    contextualize,
    generate_answer,
    grade_context,
    rerank,
    respond_casually,
    respond_with_scope,
    retrieve,
    rewrite_query,
)
from rag_chatbot.graph.state import GraphState


def route_intent(state: GraphState) -> str:
    """Choose the branch selected by the intent classifier."""
    intent = state.get("intent", "document_query")
    if intent == "casual_chat":
        return "casual"
    if intent == "unsupported_action":
        return "unsupported"
    return "document"


def route_grading(state: GraphState) -> str:
    """Generate, retry, or abstain based on evidence and retry count."""
    if state.get("context_sufficient"):
        return "generate"
    attempts = state.get("retrieval_attempts", 0)
    if attempts > settings.MAX_REWRITE_ATTEMPTS:
        return "abstain"
    return "rewrite"


def build_workflow(vector_store, reranker_instance) -> StateGraph:
    """Build the bounded conversational RAG workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("classifier", classify_intent)
    workflow.add_node("casual_responder", respond_casually)
    workflow.add_node("scope_responder", respond_with_scope)
    workflow.add_node("contextualizer", contextualize)
    workflow.add_node("retriever", partial(retrieve, vector_store=vector_store))
    workflow.add_node("reranker", partial(rerank, reranker=reranker_instance))
    workflow.add_node("grader", grade_context)
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("generator", generate_answer)
    workflow.add_node("abstainer", abstain)

    workflow.add_edge(START, "classifier")

    workflow.add_conditional_edges(
        "classifier",
        route_intent,
        {
            "casual": "casual_responder",
            "unsupported": "scope_responder",
            "document": "contextualizer",
        },
    )

    workflow.add_edge("casual_responder", END)
    workflow.add_edge("scope_responder", END)

    workflow.add_edge("contextualizer", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "grader")

    workflow.add_conditional_edges(
        "grader",
        route_grading,
        {
            "generate": "generator",
            "rewrite": "rewriter",
            "abstain": "abstainer",
        },
    )

    workflow.add_edge("rewriter", "retriever")
    workflow.add_edge("generator", END)
    workflow.add_edge("abstainer", END)

    return workflow
