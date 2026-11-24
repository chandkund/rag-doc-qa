"""Mocked tests for ordered provider generation and API failure handling."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import importlib
import importlib.util
import sys
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient


class _SdkConfig:
    def __init__(self, **kwargs):
        self.options = kwargs


class _FakeChromaClient:
    def delete_collection(self, _name):
        pass


# The runtime environment may not have optional application packages installed.
# Stub only their imports; every provider call is mocked separately below.
if importlib.util.find_spec("google.genai") is None:
    google_package = importlib.import_module("google")
    genai_module = ModuleType("google.genai")
    genai_types = ModuleType("google.genai.types")
    genai_types.EmbedContentConfig = _SdkConfig
    genai_types.GenerateContentConfig = _SdkConfig
    genai_types.HttpOptions = _SdkConfig
    genai_module.types = genai_types
    genai_module.Client = Mock()
    google_package.genai = genai_module
    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.types"] = genai_types

if importlib.util.find_spec("chromadb") is None:
    chroma_module = ModuleType("chromadb")
    chroma_module.Client = _FakeChromaClient
    sys.modules["chromadb"] = chroma_module

import llm


class FakeGeminiModels:
    def __init__(self, *results):
        self.generate_content = Mock(side_effect=results)


class FakeGeminiClient:
    def __init__(self, *results):
        self.models = FakeGeminiModels(*results)


def _http_error(status_code: int):
    error = RuntimeError(f"Gemini HTTP {status_code}")
    error.code = status_code
    return error


def _generate_api_response(monkeypatch, answer_call, *, session_id=None):
    """Call /ask without starting sample indexing or making external requests."""
    import main

    monkeypatch.setattr(main, "_index_sample_documents", lambda: None)
    monkeypatch.setattr(main, "answer_question", answer_call)
    test_session = session_id or f"test-{uuid.uuid4().hex}"
    monkeypatch.setitem(
        main.sessions,
        test_session,
        {"chunks": 1, "collection_name": test_session, "created_at": time.time()},
    )

    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/ask",
            json={"session_id": test_session, "question": "What is in the document?"},
        )
    return response


def _answer_from_llm(*_):
    result = llm.generate("system", "question")
    return {"answer": result["text"], "sources": [], "model_used": result["model_used"]}


def test_gemini_success_uses_configured_order(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini",))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    client = FakeGeminiClient(SimpleNamespace(text="Gemini answer"))
    monkeypatch.setattr(llm, "_gemini_client", lambda: client)
    openrouter = Mock()
    monkeypatch.setattr(llm, "_openrouter_generate", openrouter)

    result = llm.generate("system", "question")

    assert result == {"text": "Gemini answer", "model_used": "Gemini (test-gemini-model)"}
    client.models.generate_content.assert_called_once()
    openrouter.assert_not_called()


def test_gemini_429_twice_uses_openrouter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/provider-model")
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini", "openrouter"))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    monkeypatch.setitem(llm.MODEL_NAMES, "openrouter", "test/provider-model")
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    client = FakeGeminiClient(_http_error(429), _http_error(429))
    monkeypatch.setattr(llm, "_gemini_client", lambda: client)
    openrouter = Mock(return_value={"text": "Fallback answer", "model_used": "OpenRouter (test/provider-model)"})
    monkeypatch.setattr(llm, "_openrouter_generate", openrouter)

    result = llm.generate("system prompt", "user prompt")

    assert result["model_used"] == "OpenRouter (test/provider-model)"
    assert client.models.generate_content.call_count == 2
    openrouter.assert_called_once_with("system prompt", "user prompt", timeout_seconds=30.0)


def test_gemini_timeout_uses_openrouter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini", "openrouter"))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    monkeypatch.setitem(llm.MODEL_NAMES, "openrouter", "test-model")
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    client = FakeGeminiClient(TimeoutError(), TimeoutError())
    monkeypatch.setattr(llm, "_gemini_client", lambda: client)
    monkeypatch.setattr(
        llm,
        "_openrouter_generate",
        lambda *_, **__: {"text": "Fallback answer", "model_used": "OpenRouter (test-model)"},
    )

    result = llm.generate("system", "question")

    assert result["model_used"] == "OpenRouter (test-model)"
    assert client.models.generate_content.call_count == 2


def test_gemini_400_moves_to_next_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini", "openrouter"))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    monkeypatch.setitem(llm.MODEL_NAMES, "openrouter", "test/provider-model")
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGeminiClient(_http_error(400)))
    openrouter = Mock(return_value={"text": "Fallback answer", "model_used": "OpenRouter (test/provider-model)"})
    monkeypatch.setattr(llm, "_openrouter_generate", openrouter)

    result = llm.generate("system", "question")

    assert result["text"] == "Fallback answer"
    openrouter.assert_called_once()


def test_both_providers_fail_return_api_503(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/provider-model")
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini", "openrouter"))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    monkeypatch.setitem(llm.MODEL_NAMES, "openrouter", "test/provider-model")
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGeminiClient(_http_error(503), _http_error(503)))
    monkeypatch.setattr(
        llm.httpx,
        "post",
        Mock(side_effect=httpx.ReadTimeout("test timeout")),
    )

    response = _generate_api_response(monkeypatch, _answer_from_llm)

    assert response.status_code == 503
    assert response.json()["detail"] == llm.BUSY_MESSAGE


def test_missing_openrouter_key_and_gemini_failure_return_api_503(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini",))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGeminiClient(_http_error(503), _http_error(503)))
    post = Mock()
    monkeypatch.setattr(llm.httpx, "post", post)

    response = _generate_api_response(monkeypatch, _answer_from_llm)

    assert response.status_code == 503
    assert response.json()["detail"] == llm.BUSY_MESSAGE
    post.assert_not_called()


def test_api_keys_never_appear_in_logs_or_responses(monkeypatch, caplog):
    gemini_key = "test-gemini-secret-value"
    openrouter_key = "test-openrouter-secret-value"
    monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
    monkeypatch.setenv("OPENROUTER_API_KEY", openrouter_key)
    monkeypatch.setenv("OPENROUTER_MODEL", "test/provider-model")
    monkeypatch.setattr(llm, "LLM_ORDER", ("gemini", "openrouter"))
    monkeypatch.setitem(llm.MODEL_NAMES, "gemini", "test-gemini-model")
    monkeypatch.setitem(llm.MODEL_NAMES, "openrouter", "test/provider-model")
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGeminiClient(_http_error(429), _http_error(429)))

    def fake_post(url, *, headers, json, timeout):
        assert headers["Authorization"] == f"Bearer {openrouter_key}"
        assert timeout == 30.0
        assert json["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
        assert json["temperature"] == 0.2
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "Safe fallback answer"}}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    response = _generate_api_response(monkeypatch, _answer_from_llm)

    assert response.status_code == 200
    assert response.json()["answer"] == "Safe fallback answer"
    assert response.json()["model_used"] == "OpenRouter (test/provider-model)"
    assert gemini_key not in caplog.text + response.text
    assert openrouter_key not in caplog.text + response.text
