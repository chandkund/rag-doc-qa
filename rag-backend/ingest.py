"""PDF loading, chunking, embedding, and in-memory Chroma indexing helpers."""

import re
import os
import json
from io import BytesIO
from pathlib import Path
from typing import Union

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from llm import embed_texts

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))


class IngestionProviderError(Exception):
    """Embedding provider error with safe diagnostic details for callers."""

    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code

def load_pdf(path_or_bytes: Union[str, Path, bytes], filename: str) -> list[dict]:
    """Extract non-empty pages from a PDF path or its byte contents."""
    pdf_source = BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    reader = PdfReader(pdf_source)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": filename, "page": page_number})

    return pages


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at whitespace, never inside a word."""
    words = text.split()
    pieces = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_size:
            pieces.append(current)
            current = word
        else:
            current = candidate

    if current:
        pieces.append(current)
    return pieces


def _paragraphs(text: str, max_size: int) -> list[str]:
    """Split on paragraph and sentence boundaries, with word-safe fallback."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    units = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) <= max_size:
            if paragraph:
                units.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            if len(sentence) <= max_size:
                units.append(sentence)
            else:
                units.extend(_split_long_text(sentence, max_size))
    return units


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Create size-limited chunks while retaining each page's citation metadata."""
    chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = DEFAULT_CHUNK_OVERLAP if overlap is None else overlap
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        units = _paragraphs(page["text"], chunk_size)
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append({"text": current, "source": page["source"], "page": page["page"]})
                # Start the next chunk with a word-safe tail to preserve context.
                tail = current[-overlap:] if overlap else ""
                if tail and len(current) > overlap and not current[-overlap].isspace():
                    tail = tail.split(" ", 1)[-1]
                current = f"{tail} {unit}".strip()
                # A single unit can become larger when combined with the overlap.
                if len(current) > chunk_size:
                    pieces = _split_long_text(current, chunk_size)
                    current = pieces.pop(0)
                    for piece in pieces:
                        chunks.append({"text": current, "source": page["source"], "page": page["page"]})
                        current = piece
            else:
                current = candidate

        if current:
            chunks.append({"text": current, "source": page["source"], "page": page["page"]})

    return chunks


def build_index(chunks: list[dict], collection_name: str):
    """Create an in-memory cosine-distance Chroma collection and fill it."""
    client = chromadb.Client()
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        texts = [chunk["text"] for chunk in chunks]
        try:
            vectors = embed_texts(texts)
        except Exception as error:
            status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
            message = str(error) or type(error).__name__
            for sensitive in (os.getenv("GEMINI_API_KEY", ""), *texts):
                if sensitive:
                    message = message.replace(sensitive, "[redacted]")
            raise IngestionProviderError(status_code, message[:1000]) from error
        collection.add(
            ids=[f"chunk-{index}" for index in range(len(chunks))],
            documents=texts,
            embeddings=vectors,
            metadatas=[{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks],
        )

    return collection


def load_precomputed_index(index_path: str | Path, collection_name: str, client=None):
    """Load precomputed chunk vectors into Chroma without calling an embedding API."""
    path = Path(index_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"{path} does not contain a chunks list")

    client = client or chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if chunks:
        ids, documents, embeddings, metadatas = [], [], [], []
        for index, chunk in enumerate(chunks):
            text = chunk.get("text")
            vector = chunk.get("vector")
            metadata = chunk.get("metadata")
            if not isinstance(text, str) or not isinstance(vector, list) or not vector or not isinstance(metadata, dict):
                raise ValueError(f"Invalid chunk entry at index {index} in {path}")
            if not isinstance(metadata.get("source"), str) or not isinstance(metadata.get("page"), int):
                raise ValueError(f"Invalid citation metadata at chunk {index} in {path}")
            ids.append(f"sample-{index}")
            documents.append(text)
            embeddings.append(vector)
            metadatas.append(metadata)
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return collection, data


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent / "docs"
    all_pages = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        all_pages.extend(load_pdf(pdf_path, pdf_path.name))

    all_chunks = chunk_pages(all_pages)
    build_index(all_chunks, "documents")
    print(f"Pages: {len(all_pages)}")
    print(f"Chunks: {len(all_chunks)}")
