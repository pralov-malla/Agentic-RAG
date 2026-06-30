"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from rag_chatbot.api.routes import health, knowledge_base
from rag_chatbot.config import settings
from rag_chatbot.core.logging import configure_logging
from rag_chatbot.rag.embeddings import build_embeddings
from rag_chatbot.rag.vector_store import VectorStoreManager
from rag_chatbot.services.knowledge_base import KnowledgeBaseService

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings.ensure_directories()

        logger.info("Initializing Gemini embeddings (%s)...", settings.GEMINI_EMBEDDING_MODEL)
        embeddings = build_embeddings(settings)

        logger.info(
            "Connecting to ChromaDB at %s:%s...", settings.CHROMA_HOST, settings.CHROMA_PORT
        )
        vector_store = VectorStoreManager(
            chroma_host=settings.CHROMA_HOST,
            chroma_port=settings.CHROMA_PORT,
            chroma_ssl=settings.CHROMA_SSL,
            manifest_path=settings.INDEX_MANIFEST_PATH,
            embeddings=embeddings,
        )

        app.state.vector_store = vector_store
        app.state.knowledge_base_service = KnowledgeBaseService(vector_store, settings)

        if vector_store.is_ready:
            logger.info(
                "Active index loaded: %s (%d chunks)",
                vector_store.manifest.collection_name,
                vector_store.manifest.chunk_count,
            )
        else:
            logger.warning(
                "No active index found. "
                "Ingest a document via /api/v1/knowledge-base/upload."
            )
    except Exception as exc:
        logger.exception("Lifespan startup failed")
        app.state.startup_error = str(exc)
        yield
        return

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Agentic RAG Chatbot",
    description="Context-aware RAG chatbot using LangGraph, Gemini, and ChromaDB",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(knowledge_base.router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Agentic RAG Chatbot is running</h1>"
