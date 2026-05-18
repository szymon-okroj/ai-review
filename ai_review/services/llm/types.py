from typing import Protocol

from pydantic import BaseModel

from ai_review.config import settings
from ai_review.libs.logger import get_logger


class ChatResultSchema(BaseModel):
    text: str
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClientProtocol(Protocol):
    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        ...


class LoggingLLMClient:
    """Transparent wrapper that logs every LLM request and response at INFO level."""

    def __init__(self, client: LLMClientProtocol, logger_name: str = "LLM_CLIENT"):
        self._client = client
        self._logger = get_logger(logger_name)

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        model = settings.llm.meta.model
        self._logger.info(
            f"LLM request starting (model={model}, "
            f"system_chars={len(prompt_system)}, prompt_chars={len(prompt)})"
        )
        self._logger.info(f"LLM system prompt:\n{prompt_system}")
        self._logger.info(f"LLM user prompt:\n{prompt}")

        try:
            result = await self._client.chat(prompt=prompt, prompt_system=prompt_system)
        except Exception as error:
            self._logger.info(f"LLM request failed: {error}")
            raise

        self._logger.info(
            f"LLM request succeeded (prompt_tokens={result.prompt_tokens}, "
            f"completion_tokens={result.completion_tokens}, response_chars={len(result.text)})"
        )
        self._logger.info(f"LLM response:\n{result.text}")

        return result
