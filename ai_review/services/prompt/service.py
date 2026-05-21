from ai_review.config import settings
from ai_review.services.agent.loop.schema import AgentTraceSchema
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.prompt.schema import PromptContextSchema
from ai_review.services.prompt.tools import normalize_prompt
from ai_review.services.prompt.types import PromptServiceProtocol
from ai_review.services.prompt.xml import (
    build_agent_request_xml,
    build_review_request_xml,
    format_traces_xml,
    split_instruction_and_context_prompts,
)
from ai_review.services.vcs.types import ReviewThreadSchema


class PromptService(PromptServiceProtocol):
    @classmethod
    def prepare_prompt(cls, prompts: list[str], context: PromptContextSchema) -> str:
        prompt = "\n\n".join(prompts)
        return cls.prepare_prompt_text(prompt, context)

    @classmethod
    def prepare_prompt_text(cls, prompt: str, context: PromptContextSchema) -> str:
        prompt = context.apply_format(prompt)

        if settings.prompt.normalize_prompts:
            prompt = normalize_prompt(prompt)

        return prompt

    @classmethod
    def _prepare_review_parts(
            cls,
            raw_prompts: list[str],
            context: PromptContextSchema,
            *,
            split_supplemental_context: bool,
    ) -> tuple[str, str]:
        instructions_raw, context_raw = split_instruction_and_context_prompts(
            raw_prompts,
            split_supplemental_context=split_supplemental_context,
        )
        instructions = cls.prepare_prompt_text(instructions_raw, context) if instructions_raw else ""
        supplemental_context = cls.prepare_prompt_text(context_raw, context) if context_raw else ""
        return instructions, supplemental_context

    @classmethod
    def build_agent_request(
            cls,
            traces: list[AgentTraceSchema],
            force_final: bool,
            original_prompt: str,
            original_prompt_system: str,
    ) -> str:
        mode = "Return FINAL only." if force_final else "You can either call a tool or return FINAL."
        instructions, supplemental_context = cls._prepare_review_parts(
            settings.prompt.load_agent(),
            PromptContextSchema(),
            split_supplemental_context=settings.prompt.include_agent_prompts,
        )

        return build_agent_request_xml(
            instructions=instructions,
            supplemental_context=supplemental_context,
            agent_mode=mode,
            task_output_format=original_prompt_system,
            task=original_prompt,
            agent_history=format_traces_xml(traces),
        )

    @classmethod
    def build_inline_request(cls, diff: DiffFileSchema, context: PromptContextSchema) -> str:
        instructions, supplemental_context = cls._prepare_review_parts(
            settings.prompt.load_inline(),
            context,
            split_supplemental_context=settings.prompt.include_inline_prompts,
        )
        return build_review_request_xml(
            instructions=instructions,
            supplemental_context=supplemental_context,
            review_context=context,
            diffs=[diff],
        )

    @classmethod
    def build_summary_request(cls, diffs: list[DiffFileSchema], context: PromptContextSchema) -> str:
        instructions, supplemental_context = cls._prepare_review_parts(
            settings.prompt.load_summary(),
            context,
            split_supplemental_context=settings.prompt.include_summary_prompts,
        )
        return build_review_request_xml(
            instructions=instructions,
            supplemental_context=supplemental_context,
            review_context=context,
            diffs=diffs,
        )

    @classmethod
    def build_context_request(cls, diffs: list[DiffFileSchema], context: PromptContextSchema) -> str:
        instructions, supplemental_context = cls._prepare_review_parts(
            settings.prompt.load_context(),
            context,
            split_supplemental_context=settings.prompt.include_context_prompts,
        )
        return build_review_request_xml(
            instructions=instructions,
            supplemental_context=supplemental_context,
            review_context=context,
            diffs=diffs,
        )

    @classmethod
    def build_inline_reply_request(
            cls,
            diff: DiffFileSchema,
            thread: ReviewThreadSchema,
            context: PromptContextSchema,
    ) -> str:
        instructions, supplemental_context = cls._prepare_review_parts(
            settings.prompt.load_inline_reply(),
            context,
            split_supplemental_context=settings.prompt.include_inline_reply_prompts,
        )
        return build_review_request_xml(
            instructions=instructions,
            supplemental_context=supplemental_context,
            review_context=context,
            diffs=[diff],
            conversation=thread,
        )

    @classmethod
    def build_summary_reply_request(
            cls,
            diffs: list[DiffFileSchema],
            thread: ReviewThreadSchema,
            context: PromptContextSchema,
    ) -> str:
        instructions, supplemental_context = cls._prepare_review_parts(
            settings.prompt.load_summary_reply(),
            context,
            split_supplemental_context=settings.prompt.include_summary_reply_prompts,
        )
        return build_review_request_xml(
            instructions=instructions,
            supplemental_context=supplemental_context,
            review_context=context,
            diffs=diffs,
            conversation=thread,
        )

    @classmethod
    def build_system_agent_request(cls) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_agent(), PromptContextSchema())

    @classmethod
    def build_system_inline_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_inline(), context)

    @classmethod
    def build_system_context_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_context(), context)

    @classmethod
    def build_system_summary_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_summary(), context)

    @classmethod
    def build_system_inline_reply_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_inline_reply(), context)

    @classmethod
    def build_system_summary_reply_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_summary_reply(), context)
