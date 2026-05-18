import pytest

from ai_review.services.llm.azure_openai.client import AzureOpenAILLMClient
from ai_review.services.llm.bedrock.client import BedrockLLMClient
from ai_review.services.llm.claude.client import ClaudeLLMClient
from ai_review.services.llm.factory import get_llm_client
from ai_review.services.llm.gemini.client import GeminiLLMClient
from ai_review.services.llm.ollama.client import OllamaLLMClient
from ai_review.services.llm.openai.client import OpenAILLMClient
from ai_review.services.llm.openrouter.client import OpenRouterLLMClient
from ai_review.services.llm.types import LoggingLLMClient


@pytest.mark.usefixtures("openai_v1_http_client_config")
def test_get_llm_client_returns_openai(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, OpenAILLMClient)


@pytest.mark.usefixtures("gemini_http_client_config")
def test_get_llm_client_returns_gemini(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, GeminiLLMClient)


@pytest.mark.usefixtures("claude_http_client_config")
def test_get_llm_client_returns_claude(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, ClaudeLLMClient)


@pytest.mark.usefixtures("ollama_http_client_config")
def test_get_llm_client_returns_ollama(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, OllamaLLMClient)


@pytest.mark.usefixtures("bedrock_http_client_config")
def test_get_llm_client_returns_bedrock(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, BedrockLLMClient)


@pytest.mark.usefixtures("openrouter_http_client_config")
def test_get_llm_client_returns_openrouter(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, OpenRouterLLMClient)


@pytest.mark.usefixtures("azure_openai_http_client_config")
def test_get_llm_client_returns_azure_openai(monkeypatch: pytest.MonkeyPatch):
    client = get_llm_client()
    assert isinstance(client, LoggingLLMClient)
    assert isinstance(client._client, AzureOpenAILLMClient)


def test_get_llm_client_unsupported_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ai_review.services.llm.factory.settings.llm.provider", "UNSUPPORTED")
    with pytest.raises(ValueError):
        get_llm_client()
