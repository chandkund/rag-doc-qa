"""Provider configuration, Gemini embeddings, and ordered answer generation."""

import logging
import os
import socket
import time
from pathlib import Path
from urllib.error import URLError

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

LLM_ORDER = tuple(
    provider.strip().lower()
    for provider in os.getenv("LLM_ORDER", "").split(",")
    if provider.strip()
)
MODEL_NAMES = {
    "gemini": os.getenv("GEMINI_ANSWER_MODEL", "").strip(),
    "groq": os.getenv("GROQ_MODEL", "").strip(),
    "openrouter": os.getenv("OPENROUTER_MODEL", "").strip(),
}
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()
BATCH_SIZE = 50
BUSY_MESSAGE = "The answer service is busy. Please try again in a minute."
logger = logging.getLogger(__name__)
_GEMINI_CLIENTS: dict[tuple[str, float | None], genai.Client] = {}


class LLMUnavailableError(Exception):
    """Raised when none of the configured answer providers can respond."""

    def __init__(self, message: str = BUSY_MESSAGE, reason: str = "", status_code: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


class LLMRequestError(Exception):
    """A provider request error retained for callers that need its status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class LLMProviderError(Exception):
    """A provider failure with a safe status and response message."""

    def __init__(self, status_code: int | None, reason: str):
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


class LLMConfigurationError(LLMProviderError):
    """A configured provider is missing a key, order entry, or model name."""


def startup_config_line() -> str:
    """Return the exact provider order and model identifiers loaded at startup."""
    order = " -> ".join(LLM_ORDER) or "<empty>"
    models = ", ".join(
        f"{provider}={MODEL_NAMES.get(provider) or '<unset>'}"
        for provider in LLM_ORDER
    ) or "<none>"
    embedding = EMBEDDING_MODEL or "<unset>"
    return f"LLM config: order={order}; models={models}; embedding=gemini:{embedding}"


def _status_code(error: Exception) -> int | None:
    """Read HTTP status from common SDK and HTTP client exceptions."""
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    response = getattr(error, "response", None)
    try:
        return int(response.status_code) if response is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _message_from_payload(payload) -> str | None:
    if isinstance(payload, dict):
        error_payload = payload.get("error")
        if isinstance(error_payload, dict) and error_payload.get("message"):
            return str(error_payload["message"])
        if payload.get("message"):
            return str(payload["message"])
    if isinstance(payload, list):
        for item in payload:
            message = _message_from_payload(item)
            if message:
                return message
    return None


def _response_error_message(error: Exception) -> str | None:
    """Extract only the provider's error.message field, never a full body."""
    for payload in (getattr(error, "details", None), getattr(error, "response_json", None)):
        message = _message_from_payload(payload)
        if message:
            return message

    response = getattr(error, "response", None)
    if response is not None:
        try:
            message = _message_from_payload(response.json())
            if message:
                return message
        except Exception:
            pass

    # Google GenAI's APIError.message is parsed directly from response.error.message.
    message = getattr(error, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    reason = getattr(error, "reason", None)
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return None


def _sanitize_error_message(message: str, sensitive_values: list[str] | None = None) -> str:
    for secret in (
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("OPENROUTER_API_KEY", ""),
        *(sensitive_values or []),
    ):
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:1000]


def safe_provider_error_message(error: Exception, sensitive_values: list[str] | None = None) -> str:
    """Return only a sanitized provider error.message, without request content."""
    message = _response_error_message(error) or f"No error.message in response body ({type(error).__name__})"
    return _sanitize_error_message(message, sensitive_values)


def _provider_failure(error: Exception, sensitive_values: list[str] | None = None) -> LLMProviderError:
    status = _status_code(error)
    if isinstance(error, (TimeoutError, socket.timeout, httpx.TimeoutException)):
        status = 504
    return LLMProviderError(status, safe_provider_error_message(error, sensitive_values))


def _log_provider_error(provider: str, error: Exception, sensitive_values: list[str] | None = None) -> LLMProviderError:
    failure = error if isinstance(error, LLMProviderError) else _provider_failure(error, sensitive_values)
    message = _sanitize_error_message(failure.reason, sensitive_values)
    logger.error(
        "Provider error provider=%s status_code=%s error.message=%s",
        provider,
        failure.status_code,
        message,
    )
    return LLMProviderError(failure.status_code, message)


def _required_model(provider: str) -> str:
    model = MODEL_NAMES.get(provider, "")
    if not model:
        raise LLMConfigurationError(None, f"{provider} model is not configured")
    return model


def _gemini_client(timeout_seconds: float | None = None) -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError(None, "GEMINI_API_KEY is not configured")
    cache_key = (api_key, timeout_seconds)
    if cache_key not in _GEMINI_CLIENTS:
        options = types.HttpOptions(timeout=int(timeout_seconds * 1000)) if timeout_seconds is not None else None
        _GEMINI_CLIENTS[cache_key] = genai.Client(api_key=api_key, http_options=options) if options else genai.Client(api_key=api_key)
    return _GEMINI_CLIENTS[cache_key]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks in batches using the configured Gemini embedding model."""
    try:
        if not EMBEDDING_MODEL:
            raise LLMConfigurationError(None, "GEMINI_EMBEDDING_MODEL is not configured")
        client = _gemini_client()
    except Exception as error:
        raise _log_provider_error("gemini-embedding", error, texts) from error

    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        attempts = 1 if os.getenv("LLM_EVAL_MODE") == "1" else 3
        for attempt in range(attempts):
            try:
                response = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
                vectors.extend([item.values for item in response.embeddings])
                break
            except Exception as error:
                failure = _log_provider_error("gemini-embedding", error, batch)
                if failure.status_code != 429 or attempt == attempts - 1:
                    raise failure from error
                time.sleep(attempt + 1)
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a retrieval query using Gemini's configured embedding model."""
    try:
        if not EMBEDDING_MODEL:
            raise LLMConfigurationError(None, "GEMINI_EMBEDDING_MODEL is not configured")
        client = _gemini_client(timeout_seconds=60 if os.getenv("LLM_EVAL_MODE") == "1" else None)
    except Exception as error:
        raise _log_provider_error("gemini-embedding", error, [question]) from error

    attempts = 1 if os.getenv("LLM_EVAL_MODE") == "1" else 6
    for attempt in range(attempts):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=question,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return response.embeddings[0].values
        except Exception as error:
            failure = _log_provider_error("gemini-embedding", error, [question])
            if failure.status_code != 429 or attempt == attempts - 1:
                raise failure from error
            delay = min(5 * (2**attempt), 30)
            logger.warning("Gemini embedding rate limit; retrying in %s seconds.", delay)
            time.sleep(delay)
    raise LLMProviderError(None, "Gemini embedding returned no result")


def _is_timeout(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout, httpx.TimeoutException)):
        return True
    if isinstance(error, URLError):
        return isinstance(error.reason, (TimeoutError, socket.timeout))
    return "timeout" in type(error).__name__.lower()


def _fallback_reason(error: Exception) -> str | None:
    status = _status_code(error)
    if status in (429, 503, 504) or _is_timeout(error):
        return str(status or "timeout")
    return None


def _openai_compatible_generate(provider: str, system_prompt: str, user_prompt: str, timeout_seconds: float) -> dict:
    model = _required_model(provider)
    key_name = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
    api_key = os.getenv(key_name, "").strip()
    if not api_key:
        raise LLMConfigurationError(None, f"{key_name} is not configured")
    url = (
        "https://api.groq.com/openai/v1/chat/completions"
        if provider == "groq"
        else "https://openrouter.ai/api/v1/chat/completions"
    )
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
    except Exception as error:
        raise _provider_failure(error, [system_prompt, user_prompt]) from error

    if not isinstance(text, str) or not text.strip():
        raise LLMProviderError(None, "No error.message in response body (empty completion)")
    provider_label = "Groq" if provider == "groq" else "OpenRouter"
    logger.info("Answer generated by %s model %s.", provider_label, model)
    return {"text": text, "model_used": f"{provider_label} ({model})"}


def _groq_generate(system_prompt: str, user_prompt: str, *, timeout_seconds: float = 30.0) -> dict:
    return _openai_compatible_generate("groq", system_prompt, user_prompt, timeout_seconds)


def _openrouter_generate(system_prompt: str, user_prompt: str, *, timeout_seconds: float = 30.0) -> dict:
    return _openai_compatible_generate("openrouter", system_prompt, user_prompt, timeout_seconds)


def _generate_with_provider(provider: str, system_prompt: str, user_prompt: str, timeout_seconds: float) -> dict:
    model = _required_model(provider)
    if provider == "gemini":
        response = _gemini_client(timeout_seconds=timeout_seconds).models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        if not isinstance(response.text, str) or not response.text.strip():
            raise LLMProviderError(None, "No error.message in response body (empty completion)")
        logger.info("Answer generated by Gemini model %s.", model)
        return {"text": response.text, "model_used": f"Gemini ({model})"}
    if provider in ("groq", "openrouter"):
        if provider == "groq":
            return _groq_generate(system_prompt, user_prompt, timeout_seconds=timeout_seconds)
        return _openrouter_generate(system_prompt, user_prompt, timeout_seconds=timeout_seconds)
    raise LLMConfigurationError(None, f"Unknown provider in LLM_ORDER: {provider}")


def generate(system_prompt: str, user_prompt: str) -> dict:
    """Try answer providers in the configured LLM_ORDER, moving on after errors."""
    if not LLM_ORDER:
        failure = _log_provider_error("configuration", LLMConfigurationError(None, "LLM_ORDER is not configured"))
        raise LLMUnavailableError(BUSY_MESSAGE, failure.reason, failure.status_code)

    failures = []
    timeout_seconds = 60.0 if os.getenv("LLM_EVAL_MODE") == "1" else 30.0
    for provider in LLM_ORDER:
        attempts = 2 if provider == "gemini" else 1
        for attempt in range(attempts):
            try:
                return _generate_with_provider(provider, system_prompt, user_prompt, timeout_seconds)
            except Exception as error:
                failure = _log_provider_error(provider, error, [system_prompt, user_prompt])
                failures.append((provider, failure))
                if attempt + 1 < attempts and _fallback_reason(failure):
                    time.sleep(1)
                    continue
                break

    summary = "; ".join(
        f"{provider} HTTP {failure.status_code}: {failure.reason}"
        if failure.status_code is not None
        else f"{provider}: {failure.reason}"
        for provider, failure in failures
    )
    last_status = failures[-1][1].status_code if failures else None
    raise LLMUnavailableError(BUSY_MESSAGE, summary or "No provider could generate an answer", last_status)
