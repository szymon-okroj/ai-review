import pytest

from ai_review.services.llm.types import ChatResultSchema, LoggingLLMClient
from ai_review.tests.fixtures.services.llm import FakeLLMClient


@pytest.mark.asyncio
async def test_logging_llm_client_logs_request_and_response_on_success(
        capsys: pytest.CaptureFixture,
        fake_llm_client: FakeLLMClient,
):
    fake_llm_client.responses["chat"] = ChatResultSchema(
        text="MODEL_OUTPUT",
        prompt_tokens=10,
        completion_tokens=20,
    )
    client = LoggingLLMClient(fake_llm_client)

    result = await client.chat("user prompt", "system prompt")

    assert result.text == "MODEL_OUTPUT"
    output = capsys.readouterr().out
    assert "LLM request starting" in output
    assert "LLM system prompt:" in output
    assert "system prompt" in output
    assert "LLM user prompt:" in output
    assert "user prompt" in output
    assert "LLM request succeeded" in output
    assert "prompt_tokens=10" in output
    assert "completion_tokens=20" in output
    assert "LLM response:" in output
    assert "MODEL_OUTPUT" in output
    assert "LLM request failed" not in output


@pytest.mark.asyncio
async def test_logging_llm_client_logs_failure(
        capsys: pytest.CaptureFixture,
        fake_llm_client: FakeLLMClient,
):
    async def failing_chat(prompt: str, prompt_system: str) -> ChatResultSchema:
        raise RuntimeError("connection refused")

    fake_llm_client.chat = failing_chat
    client = LoggingLLMClient(fake_llm_client)

    with pytest.raises(RuntimeError, match="connection refused"):
        await client.chat("user prompt", "system prompt")

    output = capsys.readouterr().out
    assert "LLM request starting" in output
    assert "LLM system prompt:" in output
    assert "LLM user prompt:" in output
    assert "LLM request failed: connection refused" in output
    assert "LLM request succeeded" not in output
    assert "LLM response:" not in output
