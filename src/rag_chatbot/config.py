"""Environment-based application configuration."""

from pathlib import Path

from pydantic import ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        frozen=True,
    )

    # Gemini
    GEMINI_API_KEY: SecretStr = Field(description="Google Gemini API key")
    GEMINI_CHAT_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Gemini chat model name",
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model name",
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=768,
        gt=0,
        description="Embedding vector dimensionality",
    )

    # Cohere reranking (optional)
    COHERE_API_KEY: SecretStr | None = Field(
        default=None,
        description="Cohere API key (optional, falls back to vector ranking)",
    )
    COHERE_RERANK_MODEL: str = Field(
        default="rerank-v4.0-fast",
        description="Cohere rerank model name",
    )
    # ChromaDB (separate container)
    CHROMA_HOST: str = Field(
        default="localhost",
        min_length=1,
        description="ChromaDB server host",
    )
    CHROMA_PORT: int = Field(
        default=8100,
        ge=1,
        le=65535,
        description="ChromaDB server port",
    )
    CHROMA_SSL: bool = Field(default=False, description="Use SSL for ChromaDB connection")

    # Data paths
    INDEX_MANIFEST_PATH: Path = Field(
        default=Path("data/checkpoints/active_index.json"),
        description="Active index manifest file path",
    )
    CHECKPOINT_DB_PATH: Path = Field(
        default=Path("data/checkpoints/checkpoints.sqlite"),
        description="SQLite checkpoint database path",
    )
    DEFAULT_DOCUMENT_PATH: Path = Field(
        default=Path("data/documents/pralov_malla.pdf"),
        description="Bundled default document path",
    )
    # Chunking and retrieval
    CHUNK_SIZE: int = Field(default=1200, description="Maximum chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=180, description="Chunk overlap in characters")
    RETRIEVAL_K: int = Field(default=12, description="Number of retrieval candidates")
    RERANK_TOP_N: int = Field(default=4, description="Number of chunks after reranking")
    MAX_REWRITE_ATTEMPTS: int = Field(default=1, description="Maximum query rewrite retries")
    MAX_HISTORY_MESSAGES: int = Field(default=8, description="Conversation history window")
    MAX_UPLOAD_MB: int = Field(default=10, description="Maximum upload file size in MB")

    # Intent routing
    INTENT_CONFIDENCE_THRESHOLD: float = Field(
        default=0.70,
        description="Minimum confidence to trust intent classification",
    )

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 1200)
        if v >= chunk_size:
            msg = f"CHUNK_OVERLAP ({v}) must be smaller than CHUNK_SIZE ({chunk_size})"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def rerank_within_retrieval(self) -> "Settings":
        if self.RERANK_TOP_N > self.RETRIEVAL_K:
            msg = (
                f"RERANK_TOP_N ({self.RERANK_TOP_N}) "
                f"cannot exceed RETRIEVAL_K ({self.RETRIEVAL_K})"
            )
            raise ValueError(msg)
        return self

    @field_validator("INTENT_CONFIDENCE_THRESHOLD")
    @classmethod
    def threshold_between_zero_and_one(cls, v: float) -> float:
        if not 0 <= v <= 1:
            msg = f"INTENT_CONFIDENCE_THRESHOLD must be between 0 and 1, got {v}"
            raise ValueError(msg)
        return v

    @property
    def has_cohere(self) -> bool:
        return bool(
            self.COHERE_API_KEY
            and self.COHERE_API_KEY.get_secret_value().strip()
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create data directories if they do not exist."""
        self.INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.DEFAULT_DOCUMENT_PATH.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
