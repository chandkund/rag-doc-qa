"""Retrieve relevant document chunks and answer with grounded citations."""

import json
import re
from pathlib import Path

import chromadb
from llm import embed_query, generate

SIMILARITY_THRESHOLD = 0.35
RETRIEVAL_CANDIDATES = 8
RETRIEVAL_RESULTS = 4
NOT_FOUND = "I could not find this in the uploaded documents."
SYSTEM_INSTRUCTION = (
    "Use ONLY the numbered context passages. Every sentence must be supported by a passage and end with its citation like [2]. "
    "Cite only passages that directly support that sentence. If part of the question is not covered by the passages, "
    "say 'The uploaded documents do not cover this part' for that part. Do not add examples, tables or facts from your own knowledge. "
    "Use the plain format [1], never 【1】."
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_CITATION_ALIAS_RE = re.compile(r"【(\d+)】")
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[a-z][.)])\s+(.+)$", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^(?:who|what|when|where|why|how|which|whom|whose|can|could|do|does|is|are|was|were|"
    r"will|would|should|has|have|define|describe|explain|state|list|name|give|compare|"
    r"illustrate|discuss|identify|write)\b",
    re.IGNORECASE,
)
_EXERCISE_REQUEST_RE = re.compile(r"\b(?:exercises?|practice questions?|quiz|worksheet|problem set)\b", re.IGNORECASE)
_chroma = chromadb.Client()


def asks_for_exercises(question: str) -> bool:
    """Return whether the user explicitly asks for exercises or practice material."""
    return bool(_EXERCISE_REQUEST_RE.search(question))


def looks_like_exercise_or_question_list(text: str) -> bool:
    """Identify chunks where more than 40% of non-empty lines are questions/prompts."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    question_lines = 0
    for line in lines:
        numbered_match = _NUMBERED_LINE_RE.match(line)
        numbered_question = bool(
            numbered_match
            and (_QUESTION_START_RE.match(numbered_match.group(1)) or line.endswith("?"))
        )
        if numbered_question or line.endswith("?"):
            question_lines += 1
    return question_lines / len(lines) > 0.4


def filter_retrieved_candidates(
    question: str,
    candidates: list[dict],
    k: int = RETRIEVAL_RESULTS,
    *,
    drop_exercises: bool = True,
) -> list[dict]:
    """Drop exercise-list chunks unless requested, then keep the highest-scoring k."""
    if k < 1:
        return []
    if drop_exercises and not asks_for_exercises(question):
        candidates = [
            candidate
            for candidate in candidates
            if not looks_like_exercise_or_question_list(candidate["text"])
        ]
    return candidates[:k]


def retrieve(
    question: str,
    collection_name: str,
    k: int = RETRIEVAL_RESULTS,
    *,
    candidate_k: int = RETRIEVAL_CANDIDATES,
    drop_exercises: bool = True,
) -> list[dict]:
    """Retrieve candidate chunks, filter exercise lists, and return up to k results."""
    if k < 1 or candidate_k < 1:
        return []

    collection = _chroma.get_collection(collection_name)
    result_count = min(candidate_k, collection.count())
    if result_count < 1:
        return []
    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        matches.append(
            {
                "id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                # Chroma returns cosine distance; similarity is one minus distance.
                "score": 1.0 - distance,
            }
        )
    return filter_retrieved_candidates(question, matches, k, drop_exercises=drop_exercises)


def answer(question: str, collection_name: str, matches: list[dict] | None = None) -> dict:
    """Answer from retrieved context and return source details for citations."""
    if matches is None:
        matches = retrieve(question, collection_name)
    if not matches or matches[0]["score"] < SIMILARITY_THRESHOLD:
        return {"answer": NOT_FOUND, "sources": [], "model_used": "none"}

    context_parts = []
    sources_by_citation = {}
    for citation_number, match in enumerate(matches, start=1):
        context_parts.append(
            f"[{citation_number}] Source: {match['source']}, page {match['page']}\n{match['text']}"
        )
        sources_by_citation[citation_number] = {
                "citation": citation_number,
                "id": match["id"],
                "source": match["source"],
                "page": match["page"],
                "snippet": match["text"][:240],
                "score": match["score"],
            }

    context = "\n\n".join(context_parts)
    prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
    result = generate(SYSTEM_INSTRUCTION, prompt)
    answer_text = _CITATION_ALIAS_RE.sub(r"[\1]", result["text"])
    cited_numbers = set()

    def keep_retrieved_citation(match: re.Match) -> str:
        citation_number = int(match.group(1))
        if citation_number not in sources_by_citation:
            return ""
        cited_numbers.add(citation_number)
        return f"[{citation_number}]"

    answer_text = _CITATION_RE.sub(keep_retrieved_citation, answer_text)
    if not cited_numbers:
        answer_text = NOT_FOUND

    cited_sources = [sources_by_citation[number] for number in sorted(cited_numbers)]
    return {"answer": answer_text, "sources": cited_sources, "model_used": result["model_used"]}


if __name__ == "__main__":
    from ingest import build_index, chunk_pages, load_pdf

    docs_dir = Path(__file__).resolve().parent / "docs"
    pages = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        pages.extend(load_pdf(pdf_path, pdf_path.name))
    chunks = chunk_pages(pages)

    # Build the in-memory index in this process before running sample questions.
    build_index(chunks, "documents")
    questions = [
        "What topics are covered in the uploaded documents?",
        "What key facts or recommendations do the documents provide?",
        "What dates, names, or figures are mentioned in the documents?",
    ]
    for question in questions:
        print(f"Question: {question}")
        print(json.dumps(answer(question, "documents"), indent=2, ensure_ascii=False))
