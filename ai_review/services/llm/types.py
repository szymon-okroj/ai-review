from typing import Protocol

from pydantic import BaseModel

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
    """Transparent wrapper that logs every prompt and response at DEBUG level."""

    def __init__(self, client: LLMClientProtocol, logger_name: str = "LLM_CLIENT"):
        self._client = client
        self._logger = get_logger(logger_name)

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        self._logger.debug(
            f"LLM system prompt (chars={len(prompt_system)}):\n{prompt_system}"
        )
        self._logger.debug(
            f"LLM prompt (chars={len(prompt)}):\n{prompt}"
        )

        result = await self._client.chat(prompt=prompt, prompt_system=prompt_system)

        self._logger.debug(
            f"LLM response (chars={len(result.text)}, "
            f"prompt_tokens={result.prompt_tokens}, "
            f"completion_tokens={result.completion_tokens}):\n{result.text}"
        )

        return result
