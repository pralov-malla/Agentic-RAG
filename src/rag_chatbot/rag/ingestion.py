"""Document loading, splitting, and indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
) -> list[Document]:
    """Extract pages as Markdown using PyMuPDF4LLM."""
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,
        show_progress=False,
    )

    if start_page or end_page:
        pages = pages[(start_page or 1) - 1 : end_page or len(pages)]

    documents = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        metadata = page.get("metadata", {})
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page_number": metadata.get("page", 0),
                    "page_label": str(metadata.get("page", 0)),
                    "source": path.name,
                },
            )
        )

    logger.info("Loaded %d pages from PDF: %s", len(documents), path.name)
    return documents


def load_txt(path: Path) -> list[Document]:
    """Load a UTF-8 text file as a single document."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Text file is empty: {path.name}")

    logger.info("Loaded TXT document: %s (%d chars)", path.name, len(text))
    return [
        Document(
            page_content=text,
            metadata={"page_number": 1, "page_label": "1", "source": path.name},
        )
    ]


def load_document(
    path: Path,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
) -> list[Document]:
    """Load a PDF or TXT document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, start_page=start_page, end_page=end_page)
    if suffix == ".txt":
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    document_id: str,
    title: str,
    source_url: str,
    document_type: str,
) -> list[Document]:
    """Split documents into indexed chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        keep_separator=True,
    )

    chunks: list[Document] = []
    chunk_index = 0

    for doc in documents:
        page_number = doc.metadata.get("page_number", 0)
        page_label = doc.metadata.get("page_label", str(page_number))
        source = doc.metadata.get("source", "")

        for split_text in splitter.split_text(doc.page_content):
            content_hash = hashlib.sha256(split_text.encode()).hexdigest()[:16]
            chunks.append(
                Document(
                    page_content=split_text,
                    metadata={
                        "document_id": document_id,
                        "title": title,
                        "source": source,
                        "source_url": source_url,
                        "page_number": page_number,
                        "page_label": page_label,
                        "section": _extract_section(split_text),
                        "chunk_index": chunk_index,
                        "content_hash": content_hash,
                        "document_type": document_type,
                    },
                )
            )
            chunk_index += 1

    logger.info(
        "Chunked %d pages into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def generate_document_id(path: Path) -> str:
    """Deterministic document ID from file content hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _extract_section(text: str) -> str:
    """Extract the nearest Markdown heading from text."""
    for line in text.splitlines():
        if line.strip().startswith("#"):
            return line.strip().lstrip("# ").strip()
    return ""
