"""FastAPI endpoints for temporary, session-based document Q&A."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

import chromadb
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.formparsers import MultiPartParser

from ingest import build_index, chunk_pages, load_pdf, load_precomputed_index
from llm import LLMRequestError, LLMUnavailableError, BUSY_MESSAGE, startup_config_line
from rag import answer as answer_question

DOCS_DIR = BASE_DIR / "docs"
MAX_FILES = 5
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_PAGES = 100
SESSION_TTL_SECONDS = 60 * 60
NOT_FOUND_ANSWER = "I could not find this in the uploaded documents."

# Keep multipart uploads in memory; Starlette otherwise spills larger uploads
# to a temporary file. The endpoint enforces the 5 MB limit for each file.
MultiPartParser.max_file_size = 0
if hasattr(MultiPartParser, "spool_max_size"):
    MultiPartParser.spool_max_size = 0

limiter = Limiter(key_func=get_remote_address, application_limits=["10/minute"])
chroma = chromadb.Client()
# Keep only timestamps and collection names in application memory; PDF bytes are never saved.
sessions: dict[str, dict] = {}
sample_session_id: str | None = None
logger = logging.getLogger(__name__)


def _delete_expired_sessions() -> None:
    """Remove session collections that have been idle for more than an hour."""
    expired = [
        session_id
        for session_id, session in sessions.items()
        if session_id != sample_session_id
        and time.time() - session["created_at"] > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        try:
            chroma.delete_collection(sessions[session_id]["collection_name"])
        except Exception:
            # The collection may already have been removed by a concurrent cleanup.
            pass
        sessions.pop(session_id, None)


def _index_sample_documents() -> str | None:
    """Load the checked-in sample vectors without calling an embedding provider."""
    index_path = BASE_DIR / "sample_index.json"
    if not index_path.is_file():
        return None
    session_id = "sample_documents"
    collection, data = load_precomputed_index(index_path, session_id, chroma)
    files = data.get("files", [])
    file_details = data.get("file_details", [])
    sessions[session_id] = {
        "created_at": time.time(),
        "collection_name": session_id,
        "files": files,
        "file_details": file_details,
        "pages": data.get("pages", sum(item.get("pages", 0) for item in file_details)),
        "chunks": collection.count(),
    }
    return session_id


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(60)
        _delete_expired_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sample_session_id
    logger.info("%s", startup_config_line())
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        # The service can start without a sample index; uploads remain available.
        try:
            sample_session_id = await run_in_threadpool(_index_sample_documents)
        except Exception:
            sample_session_id = None
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        for session in list(sessions.values()):
            try:
                chroma.delete_collection(session["collection_name"])
            except Exception:
                pass
        sessions.clear()


production = os.getenv("ENV", "").strip().lower() == "production"
app = FastAPI(
    title="Document Q&A with Citations",
    lifespan=lifespan,
    docs_url=None if production else "/docs",
    redoc_url=None if production else "/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=500)


@app.get("/health")
@limiter.limit("10/minute")
def health(request: Request):
    return {"status": "ok"}


@app.post("/upload")
@limiter.limit("10/minute")
async def upload(request: Request, files: list[UploadFile] = File(...)):
    """Validate PDFs in memory, then create a separate Chroma collection."""
    _delete_expired_sessions()
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(status_code=400, detail="Upload between 1 and 5 PDF files.")

    pages = []
    file_names = []
    file_details = []
    total_pages = 0

    for file in files:
        filename = Path(file.filename or "").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

        data = await file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds the 5 MB file limit.")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail=f"{filename} is not a valid PDF file.")

        try:
            page_count = len(PdfReader(BytesIO(data)).pages)
            file_pages = load_pdf(data, filename)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Could not read {filename} as a PDF.") from error

        total_pages += page_count
        if total_pages > MAX_TOTAL_PAGES:
            raise HTTPException(status_code=413, detail="The combined page limit is 100 pages.")
        pages.extend(file_pages)
        file_names.append(filename)
        file_details.append({"name": filename, "pages": page_count})

    chunks = chunk_pages(pages)
    session_id = f"session_{uuid.uuid4().hex}"
    try:
        await run_in_threadpool(build_index, chunks, session_id)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not build the document index: {error}") from error

    sessions[session_id] = {
        "created_at": time.time(),
        "collection_name": session_id,
        "files": file_names,
        "file_details": file_details,
        "pages": total_pages,
        "chunks": len(chunks),
    }
    return {
        "session_id": session_id,
        "files": file_names,
        "file_details": file_details,
        "pages": total_pages,
        "chunks": len(chunks),
    }


@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest):
    """Answer a question using only the collection for its upload session."""
    _delete_expired_sessions()
    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session_expired")
    if session["chunks"] == 0:
        return {"answer": NOT_FOUND_ANSWER, "sources": [], "model_used": "none"}
    try:
        return answer_question(body.question, session["collection_name"])
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
    except LLMRequestError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Could not retrieve from this document session.") from error
    except Exception as error:
        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        if status_code in (400, 401):
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        raise


@app.get("/sample-session")
@limiter.limit("10/minute")
def get_sample_session(request: Request):
    global sample_session_id
    _delete_expired_sessions()
    if sample_session_id is None or sample_session_id not in sessions:
        try:
            sample_session_id = _index_sample_documents()
        except Exception:
            logger.exception("Could not load the precomputed sample index.")
            sample_session_id = None
    if sample_session_id is None or sample_session_id not in sessions:
        raise HTTPException(status_code=404, detail="No sample index is available. Run build_sample_index.py first.")
    session = sessions[sample_session_id]
    return {
        "session_id": sample_session_id,
        "files": session["files"],
        "file_details": session["file_details"],
        "pages": session["pages"],
    }
