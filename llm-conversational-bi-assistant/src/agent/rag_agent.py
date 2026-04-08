"""
RAG Agent — Retrieval-Augmented Generation over internal documents.

Flow:
  1. On initialization: ingest .md from data/documents/ into ChromaDB
     (idempotent — skip if already ingested)
  2. On each query: retrieve the top-K most relevant chunks via
     cosine similarity (embeddings sentence-transformers)
  3. Generate response via LLM — sources are returned separately and
     displayed by the CLI/API layer, NOT embedded in the answer text.

Design decisions:
  - Chunk size: 500 chars with 50 chars of overlap — balances context and precision
  - Top-K: 3 chunks — sufficient for policy documents without exceeding context window
  - Embedding model: all-MiniLM-L6-v2 — same as cache, loaded once
  - Document access is unrestricted (company-wide policies)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from src.config import settings
from src.utils import get_logger

logger = get_logger("agent.rag")

_COLLECTION_NAME = "internal_documents"
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50
_TOP_K = 3

_SYSTEM_PROMPT = """You are a helpful assistant for a staffing operations team.
Answer questions based ONLY on the provided document excerpts.
Always read and map all relevant content from the provided documents to answer the question.
Summarize the answer in a concise and clear manner.
Do not repeat the content of the documents in your answer.
Create your own answer based on the provided documents with a professional and friendly tone.
If you use more than one document, organize the information in a logical and coherent manner.
Do NOT include source citations or references inside your answer text, sources will be displayed separately by the system.
If the answer is not found in the provided documents, say so clearly.
Always respond in the same language as the user's question."""


class RAGAgent:
    """Answers questions about internal documents, returning sources separately."""

    def __init__(self) -> None:
        default_headers = None
        if "openrouter.ai" in settings.openrouter_base_url:
            default_headers = {
                "HTTP-Referer": settings.openrouter_app_url,
                "X-Title": settings.openrouter_app_name,
            }
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=default_headers,
        )

        self._chroma = chromadb.PersistentClient(
            path=str(settings.chroma_persist_path)
        )
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._chroma.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=cast(Any, self._embed_fn),
            metadata={"hnsw:space": "cosine"},
        )

        self._ensure_ingested()

    def _ensure_ingested(self) -> None:
        """Ingest documents only if the collection is empty (idempotent)."""
        if self._collection.count() > 0:
            logger.debug(
                f"ChromaDB: {self._collection.count()} chunks already present — "
                "ingestion ignored."
            )
            return

        docs_path = settings.documents_path
        md_files = list(docs_path.glob("*.md"))

        if not md_files:
            logger.warning(f"No .md file found in {docs_path}")
            return

        logger.info(f"Starting ingestion of {len(md_files)} documents...")
        total_chunks = 0

        for md_file in md_files:
            chunks = self._chunk_document(md_file)
            if not chunks:
                continue

            self._collection.add(
                ids=[f"{md_file.name}::chunk{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[
                    {"source": md_file.name, "chunk_index": i}
                    for i in range(len(chunks))
                ],
            )
            total_chunks += len(chunks)
            logger.debug(f"  {md_file.name}: {len(chunks)} chunks")

        logger.info(f"Ingestion completed: {total_chunks} chunks of {len(md_files)} documents")

    def _chunk_document(self, path: Path) -> list[str]:
        """
        Split a Markdown document into chunks with overlap.

        Prefers paragraph breaks (\\n\\n) to preserve semantic cohesion.
        Falls back to character-level splitting for oversized paragraphs.
        """
        text = path.read_text(encoding="utf-8")

        text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= _CHUNK_SIZE:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) > _CHUNK_SIZE:
                    for i in range(0, len(para), _CHUNK_SIZE - _CHUNK_OVERLAP):
                        chunks.append(para[i: i + _CHUNK_SIZE])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    def reingest(self) -> int:
        """
        Force re-ingestion of all documents.
        Useful when source files are updated.
        Returns the number of chunks ingested.
        """
        logger.info("Forced re-ingestion: removing existing collection...")
        self._chroma.delete_collection(_COLLECTION_NAME)
        self._collection = self._chroma.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=cast(Any, self._embed_fn),
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_ingested()
        return self._collection.count()

    def query(self, question: str) -> dict:
        """
        Answer a question about internal documents.

        Args:
            question: Natural language question.

        Returns:
            Dict with keys:
              answer      (str)       — LLM-generated answer, no inline citations
              sources     (list[str]) — deduplicated source filenames
              chunks_used (int)       — number of retrieved chunks

            Never raises — errors are returned as friendly answer strings.
        """
        logger.info(f"RAG query: {question!r}")

        try:
            if self._collection.count() == 0:
                return {
                    "answer": "No internal documents are available at the moment.",
                    "sources": [],
                    "chunks_used": 0,
                }

            results = self._collection.query(
                query_texts=[question],
                n_results=min(_TOP_K, self._collection.count()),
            )

            documents = cast(list[list[str]], results["documents"])[0]
            metadatas = cast(list[list[dict[str, Any]]], results["metadatas"])[0]

        except Exception as exc:
            logger.error(f"RAG retrieval failed: {exc}")
            return {
                "answer": (
                    "We had a problem searching the internal documents. "
                    "Please try again in a moment."
                ),
                "sources": [],
                "chunks_used": 0,
            }

        if not documents:
            return {
                "answer": "No relevant information was found in the internal documents.",
                "sources": [],
                "chunks_used": 0,
            }

        seen: set[str] = set()
        sources: list[str] = []
        for meta in metadatas:
            src = meta.get("source", "unknown")
            if src not in seen:
                sources.append(src)
                seen.add(src)

        context_blocks = [
            f"[{meta['source']}]\n{doc}"
            for doc, meta in zip(documents, metadatas, strict=False)
        ]
        context = "\n\n---\n\n".join(context_blocks)

        try:
            answer = self._generate_answer(question, context)
        except Exception as exc:
            logger.error(f"RAG answer generation failed: {exc}")
            return {
                "answer": (
                    "We're having trouble reaching the AI service right now. "
                    "Please try again in a few seconds."
                ),
                "sources": sources,
                "chunks_used": len(documents),
            }

        logger.debug(f"RAG answer generated. Sources: {sources}")

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(documents),
        }

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer via LLM using the retrieved context."""
        user_message = (
            f"Document excerpts:\n\n{context}\n\n"
            f"Question: {question}"
        )

        response = self._client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=768,
        )

        return (response.choices[0].message.content or "").strip()

    def document_count(self) -> int:
        """Return the number of ingested chunks (used by /health)."""
        return self._collection.count()
