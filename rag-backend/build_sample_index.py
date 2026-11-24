"""Build the sample document index once; the API loads these vectors at startup."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

from ingest import chunk_pages, load_pdf
from llm import embed_texts
from pypdf import PdfReader


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pdf_paths = sorted((BASE_DIR / "docs").glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit("No sample PDFs found in rag-backend/docs.")

    pages = []
    file_details = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf(pdf_path, pdf_path.name))
        file_details.append({"name": pdf_path.name, "pages": len(PdfReader(str(pdf_path)).pages)})
    chunks = chunk_pages(pages)
    if not chunks:
        raise SystemExit("No extractable text found in the sample PDFs.")

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks")
    data = {
        "version": 1,
        "files": [path.name for path in pdf_paths],
        "file_details": file_details,
        "pages": sum(item["pages"] for item in file_details),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "").strip(),
        "chunks": [
            {
                "text": chunk["text"],
                "metadata": {"source": chunk["source"], "page": chunk["page"]},
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    }
    destination = BASE_DIR / "sample_index.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)
    print(f"Saved {len(chunks)} chunks from {len(pdf_paths)} PDF(s) to {destination.name}.")


if __name__ == "__main__":
    main()
