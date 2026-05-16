from typing import Literal

from pydantic import BaseModel, ConfigDict


class BedrockContentSchema(BaseModel):
    type: Literal["text"] = "text"
    text: str


class BedrockMessageSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: list[BedrockContentSchema]


class BedrockChatRequestSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    anthropic_version: str = "bedrock-2023-05-31"
    messages: list[BedrockMessageSchema]
    system: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class BedrockUsageSchema(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BedrockChatResponseSchema(BaseModel):
    id: str
    type: str
    role: str
    usage: BedrockUsageSchema
    content: list[BedrockContentSchema]

    @property
    def first_text(self) -> str:
        if not self.content:
            return ""

        return (self.content[0].text or "").strip()
