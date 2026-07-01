"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rag_chatbot.api.routes import chat, health, knowledge_base, web
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
                "No active index found. Ingest a document via /api/v1/knowledge-base/upload."
            )

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from rag_chatbot.graph.workflow import build_workflow
        from rag_chatbot.rag.reranker import build_reranker
        from rag_chatbot.services.chat import ChatService

        logger.info("Initializing checkpointer at %s...", settings.CHECKPOINT_DB_PATH)
        async with AsyncSqliteSaver.from_conn_string(
            str(settings.CHECKPOINT_DB_PATH)
        ) as checkpointer:
            await checkpointer.setup()

            logger.info("Compiling LangGraph workflow...")
            reranker = build_reranker()
            workflow = build_workflow(vector_store, reranker)
            agent = workflow.compile(checkpointer=checkpointer)

            app.state.chat_service = ChatService(agent, settings)

            yield

            logger.info("Shutting down...")
            return

    except Exception as exc:
        logger.exception("Lifespan startup failed")
        app.state.startup_error = str(exc)
        yield
        return


app = FastAPI(
    title="Agentic RAG Chatbot",
    description="Context-aware RAG chatbot using LangGraph, Gemini, and ChromaDB",
    version="0.1.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

app.include_router(health.router, prefix="/api/v1")
app.include_router(knowledge_base.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(web.router)
