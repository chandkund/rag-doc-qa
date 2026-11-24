"""Run and resume the configured-provider evaluation suite."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
QUESTIONS_PATH = BASE_DIR / "test_questions.json"
RESULTS_PATH = BASE_DIR / "evaluation_results.json"
PROGRESS_PATH = BASE_DIR / "evaluation_progress.json"
BACKOFF_SECONDS = (5, 10, 20, 40, 60)
QUESTION_PAUSE_SECONDS = 4
RETRYABLE_STATUSES = {429, 503, 504}
QUESTION_COUNT = 19
ANSWERABLE_COUNT = 16
RETRIEVAL_CHUNK_CONFIGS = ((500, 100), (500, 150), (800, 100), (800, 150))


def _load_questions() -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != QUESTION_COUNT:
        raise ValueError(
            f"test_questions.json must contain {QUESTION_COUNT} questions: "
            f"{ANSWERABLE_COUNT} answerable and {QUESTION_COUNT - ANSWERABLE_COUNT} unanswerable."
        )
    answerable_count = sum(item.get("answerable", True) for item in questions)
    if answerable_count != ANSWERABLE_COUNT:
        raise ValueError(f"The evaluation set must contain exactly {ANSWERABLE_COUNT} answerable questions.")
    for item in questions:
        if not item.get("question"):
            raise ValueError("Every question needs a non-empty question field.")
        if item.get("answerable", True):
            for field in ("expected_source", "expected_page", "expected_keywords"):
                if field not in item:
                    raise ValueError(f"Answerable question is missing {field}: {item['question']}")
    return questions


def _write_json(path: Path, value: dict) -> None:
    """Replace JSON files atomically so a crash does not truncate progress."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _dataset_hash(pdf_paths: list[Path], questions: list[dict]) -> str:
    digest = hashlib.sha256(json.dumps(questions, sort_keys=True).encode("utf-8"))
    for pdf_path in pdf_paths:
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(pdf_path.read_bytes())
    return digest.hexdigest()


def _load_progress(metadata: dict) -> list[dict]:
    if not PROGRESS_PATH.exists():
        return []
    try:
        saved = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if any(saved.get(key) != value for key, value in metadata.items()):
        return []
    return saved.get("results", [])


def _provider_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _error_label(error: Exception) -> str:
    status = _provider_status(error)
    if status is not None:
        return f"HTTP {status}"
    reason = getattr(error, "reason", "")
    if reason:
        return str(reason)
    return type(error).__name__


def _error_details(error: Exception) -> dict:
    """Persist useful diagnostics without copying secrets or submitted text."""
    from llm import safe_provider_error_message

    message = safe_provider_error_message(error)
    return {
        "error_type": type(error).__name__,
        "status_code": _provider_status(error),
        "message": message[:1000],
    }


def _preflight(provider: str, model: str) -> bool:
    """Check one Gemini embedding and one selected-provider generation call."""
    from llm import EMBEDDING_MODEL, embed_query, generate
    from rag import SYSTEM_INSTRUCTION

    checks = (
        ("embedding", EMBEDDING_MODEL, lambda: embed_query("Preflight connection check.")),
        ("generation", model, lambda: generate(SYSTEM_INSTRUCTION, "Reply with the single word: ready")),
    )
    for label, model_name, call in checks:
        try:
            call()
        except Exception as error:
            details = _error_details(error)
            print(f"Preflight failed ({provider} {label}, model {model_name}).")
            print(f"Error type: {details['error_type']}")
            print(f"Status code: {details['status_code']}")
            print(f"Message: {details['message']}")
            return False
    print(f"Preflight passed: Gemini embeddings ({EMBEDDING_MODEL}) and {provider} generation ({model}).")
    return True


def _retrieval_only() -> None:
    """Compare baseline retrieval with requested chunk settings without answer generation."""
    from ingest import build_index, chunk_pages, load_pdf
    from llm import EMBEDDING_MODEL
    from rag import retrieve

    if not os.getenv("GEMINI_API_KEY") or not EMBEDDING_MODEL:
        raise ValueError("Retrieval-only evaluation requires GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL.")

    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Add the source PDFs to {DOCS_DIR} before the evaluation.")
    questions = _load_questions()
    pages = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf(pdf_path, pdf_path.name))

    configurations = [("before: raw top 4", 800, 150, 4, False)]
    configurations.extend(
        ("after: filter from top 8", chunk_size, overlap, 8, True)
        for chunk_size, overlap in RETRIEVAL_CHUNK_CONFIGS
    )
    rows = []
    for label, chunk_size, overlap, candidate_k, drop_exercises in configurations:
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        collection_name = "retrieval_eval_" + uuid.uuid4().hex
        build_index(chunks, collection_name)
        hits = 0
        total = 0
        for item in questions:
            if not item.get("answerable", True):
                continue
            matches = retrieve(
                item["question"],
                collection_name,
                k=4,
                candidate_k=candidate_k,
                drop_exercises=drop_exercises,
            )
            total += 1
            if any(
                Path(match["source"]).name == Path(item["expected_source"]).name
                and abs(int(match["page"]) - int(item["expected_page"])) <= 1
                for match in matches
            ):
                hits += 1
        rows.append(
            {
                "label": label,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunks": len(chunks),
                "hits": hits,
                "total": total,
                "hit_rate": hits / total if total else 0.0,
                "candidate_k": candidate_k,
                "drop_exercises": drop_exercises,
            }
        )

    best = max(rows[1:], key=lambda row: (row["hit_rate"], -row["chunks"]))
    headers = ("Mode", "Chunk", "Overlap", "Index chunks", "Hit@4")
    output_rows = [
        (
            row["label"],
            str(row["chunk_size"]),
            str(row["overlap"]),
            str(row["chunks"]),
            f"{row['hits']}/{row['total']} ({row['hit_rate']:.1%})",
        )
        for row in rows
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in output_rows)) for index in range(len(headers))]
    print("| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |")
    print("| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |")
    for row in output_rows:
        print("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |")

    print("\nBefore / after comparison:")
    print(f"Before: 800/150 overlap, raw top 4 = {rows[0]['hits']}/{rows[0]['total']} ({rows[0]['hit_rate']:.1%})")
    print(
        f"After: {best['chunk_size']}/{best['overlap']} overlap, top 8 filtered to 4 = "
        f"{best['hits']}/{best['total']} ({best['hit_rate']:.1%})"
    )
    print(f"Recommended chunk settings: chunk_size={best['chunk_size']}, overlap={best['overlap']}.")


def _question_result(item: dict, number: int, response: dict, retrieved: list[dict], model: str) -> dict:
    answerable = item.get("answerable", True)
    result = {
        "number": number,
        "question": item["question"],
        "answerable": answerable,
        "status": "completed",
        "answer": response["answer"],
        "sources": response["sources"],
        "model": model,
        # Store the exact configured model identifier, without a provider label.
        "model_used": model,
    }

    if answerable:
        expected_source = item["expected_source"]
        expected_page = int(item["expected_page"])
        result["retrieval_hit"] = any(
            Path(match["source"]).name == Path(expected_source).name
            and abs(int(match["page"]) - expected_page) <= 1
            for match in retrieved
        )
        answer_lower = response["answer"].casefold()
        result["answer_correct"] = all(
            str(keyword).casefold() in answer_lower
            for keyword in item["expected_keywords"]
        )
        result["expected_source"] = expected_source
        result["expected_page"] = expected_page
        result["expected_keywords"] = item["expected_keywords"]
    else:
        result["refusal_correct"] = response["answer"].strip() == "I could not find this in the uploaded documents."
    return result


def _print_table(results: list[dict]) -> None:
    headers = ("#", "Question", "Status", "Model", "Hit@4", "Answer / refusal")
    rows = []
    for result in results:
        if result["status"] == "error":
            outcome = result["error"]
        else:
            outcome = "pass" if result.get("answer_correct", result.get("refusal_correct", False)) else "fail"
        hit_display = (
            "error" if result["status"] == "error"
            else "hit" if result.get("retrieval_hit")
            else "miss" if result["answerable"]
            else "n/a"
        )
        rows.append(
            (
                str(result["number"]),
                result["question"].replace("|", "\\|")[:52],
                result["status"],
                result["model"],
                hit_display,
                outcome,
            )
        )
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print("| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |")
    print("| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |")
    for row in rows:
        print("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Document Q&A with citations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help=f"Run or resume all {QUESTION_COUNT} questions.")
    group.add_argument("--preflight", action="store_true", help="Check the embedding and answer providers only.")
    group.add_argument("--retrieval-only", action="store_true", help="Compare retrieval settings without answer generation.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env", override=True)
    except ImportError:
        pass

    if args.retrieval_only:
        _retrieval_only()
        return

    provider = os.getenv("EVAL_PROVIDER", "").strip().lower()
    model_variables = {
        "gemini": "GEMINI_ANSWER_MODEL",
        "groq": "GROQ_MODEL",
        "openrouter": "OPENROUTER_MODEL",
    }
    if provider not in model_variables:
        raise ValueError("Set EVAL_PROVIDER to gemini, groq, or openrouter.")
    model = os.getenv(model_variables[provider], "").strip()
    if not model:
        raise ValueError(f"Set {model_variables[provider]} to evaluate with {provider}.")
    os.environ["EVAL_PROVIDER"] = provider
    os.environ["LLM_ORDER"] = provider
    os.environ["LLM_EVAL_MODE"] = "1"
    os.environ["EVAL_TIMEOUT_SECONDS"] = "60"

    if not _preflight(provider, model):
        raise SystemExit(1)
    if args.preflight:
        return

    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Add the source PDFs to {DOCS_DIR} before the evaluation.")
    questions = _load_questions()

    from ingest import build_index, chunk_pages, load_pdf
    from rag import NOT_FOUND, answer, retrieve

    metadata = {
        "provider": provider,
        "model": model,
        "dataset_hash": _dataset_hash(pdf_paths, questions),
    }
    results = _load_progress(metadata)
    completed_numbers = {
        result["number"] for result in results if result.get("status") == "completed"
    }

    pages = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf(pdf_path, pdf_path.name))
    chunks = chunk_pages(pages)
    collection_name = "evaluation_" + uuid.uuid4().hex
    build_index(chunks, collection_name)

    def save_progress() -> None:
        _write_json(
            PROGRESS_PATH,
            {
                **metadata,
                "question_count": len(questions),
                "results": sorted(results, key=lambda result: result["number"]),
            },
        )

    for number, item in enumerate(questions, start=1):
        if number in completed_numbers:
            continue

        retrieved: list[dict] = []
        response = None
        final_error = None
        for attempt in range(len(BACKOFF_SECONDS) + 1):
            try:
                retrieved = retrieve(item["question"], collection_name, k=4)
                response = answer(item["question"], collection_name, matches=retrieved)
                final_error = None
                break
            except Exception as error:
                final_error = error
                status = _provider_status(error)
                if status not in RETRYABLE_STATUSES or attempt == len(BACKOFF_SECONDS):
                    break
                delay = BACKOFF_SECONDS[attempt]
                print(f"Question {number}: provider HTTP {status}; retry {attempt + 1}/5 in {delay}s.")
                time.sleep(delay)

        if final_error is not None or response is None:
            details = _error_details(final_error) if final_error else {
                "error_type": "NoResponseError", "status_code": None, "message": "No response"
            }
            result = {
                "number": number,
                "question": item["question"],
                "answerable": item.get("answerable", True),
                "status": "error",
                "error": _error_label(final_error) if final_error else "No response",
                **details,
                "model": model,
                "model_used": model,
                "sources": [],
            }
        else:
            result = _question_result(item, number, response, retrieved, model)

        results = [existing for existing in results if existing["number"] != number]
        results.append(result)
        save_progress()
        completed_numbers.add(number)
        if number < len(questions):
            time.sleep(QUESTION_PAUSE_SECONDS)

    results = sorted(results, key=lambda result: result["number"])
    errors = [result for result in results if result["status"] == "error"]
    answerable = [
        result for result in results
        if result["answerable"] and result["status"] == "completed"
    ]
    answerable_total = sum(item.get("answerable", True) for item in questions)
    hit_rate = (
        sum(result.get("retrieval_hit", False) for result in answerable) / len(answerable)
        if answerable else None
    )
    answer_accuracy = (
        sum(result["answer_correct"] for result in answerable) / len(answerable)
        if answerable else None
    )
    correct_refusals = sum(
        result.get("refusal_correct", False)
        for result in results
        if not result["answerable"] and result["status"] == "completed"
    )

    report = {
        **metadata,
        "status": "completed" if not errors else "completed_with_errors",
        "pdf_files": [path.name for path in pdf_paths],
        "pages_indexed": len(pages),
        "chunks_indexed": len(chunks),
        "question_count": len(questions),
        "completed_questions": len(results) - len(errors),
        "failed_questions": len(errors),
        "metrics": {
            "hit_rate_at_4": hit_rate,
            "answer_accuracy": answer_accuracy,
            "answer_accuracy_questions": len(answerable),
            "correct_refusals": correct_refusals,
            "unanswerable_count": sum(not item.get("answerable", True) for item in questions),
        },
        "results": results,
    }
    _write_json(RESULTS_PATH, report)
    save_progress()

    _print_table(results)
    print(f"Hit rate@4: {hit_rate:.1%}" if hit_rate is not None else "Hit rate@4: N/A (no answerable questions completed)")
    print(f"Answer accuracy: {answer_accuracy:.1%}" if answer_accuracy is not None else "Answer accuracy: N/A (no answerable questions completed)")
    print(f"Correct refusals: {correct_refusals} of 3")
    print(f"Model used: {model}")
    print(f"{len(errors)} questions failed due to errors")
    if errors:
        first_error = errors[0]
        print(
            "First error: "
            f"question {first_error['number']}, {first_error['error_type']}, "
            f"status {first_error['status_code']}: {first_error['message']}"
        )
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved resumable progress to {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
