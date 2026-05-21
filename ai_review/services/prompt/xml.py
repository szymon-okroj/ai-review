import html
import re

from ai_review.config import settings
from ai_review.services.agent.loop.schema import AgentTraceSchema
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.prompt.schema import PromptContextSchema
from ai_review.services.vcs.types import ReviewThreadSchema

_XML_TAG_RE = re.compile(r"^[\w.-]+$")


def escape_xml_text(text: str) -> str:
    return html.escape(text, quote=False)


def escape_xml_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _safe_xml_tag(name: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", name.strip())
    return tag if tag and _XML_TAG_RE.match(tag) else "field"


def wrap_cdata_element(tag: str, content: str) -> str:
    if not content:
        return f"<{tag}></{tag}>"
    safe = content.replace("]]>", "]]]]><![CDATA[>")
    return f"<{tag}><![CDATA[\n{safe}\n]]></{tag}>"


def split_instruction_and_context_prompts(
        prompts: list[str],
        *,
        split_supplemental_context: bool,
) -> tuple[str, str]:
    if not prompts:
        return "", ""
    if not split_supplemental_context or len(prompts) == 1:
        return "\n\n".join(prompts), ""
    return prompts[0], "\n\n".join(prompts[1:])


def format_review_metadata_xml(context: PromptContextSchema) -> str:
    values = {**context.model_dump(), **settings.prompt.context}
    lines = ["<review_metadata>"]
    has_fields = False

    for key, value in values.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        tag = _safe_xml_tag(key)
        lines.append(f"  <{tag}>{escape_xml_text(str(value))}</{tag}>")
        has_fields = True

    lines.append("</review_metadata>")
    return "\n".join(lines) if has_fields else ""


def format_file_xml(diff: DiffFileSchema) -> str:
    path = escape_xml_attr(diff.file.strip())
    if not diff.diff:
        return f'  <file path="{path}"></file>'

    safe = diff.diff.replace("]]>", "]]]]><![CDATA[>")
    return (
        f'  <file path="{path}"><![CDATA[\n'
        f"{safe}\n"
        f"  ]]></file>"
    )


def format_files_xml(diffs: list[DiffFileSchema]) -> str:
    if not diffs:
        return "<diff></diff>"
    files = "\n".join(format_file_xml(diff) for diff in diffs)
    return f"<diff>\n{files}\n</diff>"


def format_conversation_xml(thread: ReviewThreadSchema) -> str:
    if not thread.comments:
        return "<conversation></conversation>"

    comments: list[str] = []
    for comment in thread.comments:
        body = (comment.body or "").strip()
        if not body:
            continue
        author = (comment.author.name or comment.author.username or "User").strip()
        comments.append(
            f'  <comment author="{escape_xml_attr(author)}">'
            f"{escape_xml_text(body)}</comment>"
        )

    if not comments:
        return "<conversation></conversation>"

    return "<conversation>\n" + "\n".join(comments) + "\n</conversation>"


def format_trace_xml(trace: AgentTraceSchema) -> str:
    lines = [f'  <step iteration="{trace.iteration}">']

    if trace.step.command:
        lines.append(f"    <command>{escape_xml_text(trace.step.command)}</command>")

    if trace.tool_output:
        for line in wrap_cdata_element("tool_output", trace.tool_output).splitlines():
            lines.append(f"    {line}")

    if trace.step.content:
        lines.append(f"    <content>{escape_xml_text(trace.step.content)}</content>")

    if trace.warning:
        lines.append(f"    <warning>{escape_xml_text(trace.warning)}</warning>")

    lines.append("  </step>")
    return "\n".join(lines)


def format_traces_xml(traces: list[AgentTraceSchema]) -> str:
    if not traces:
        return "<agent_history></agent_history>"
    steps = "\n".join(format_trace_xml(trace) for trace in traces)
    return f"<agent_history>\n{steps}\n</agent_history>"


def build_review_request_xml(
        *,
        instructions: str,
        supplemental_context: str,
        review_context: PromptContextSchema,
        diffs: list[DiffFileSchema] | None = None,
        conversation: ReviewThreadSchema | None = None,
) -> str:
    parts = ["<ai_review_request>"]

    if instructions:
        parts.append(wrap_cdata_element("instructions", instructions))

    if supplemental_context:
        parts.append(wrap_cdata_element("context", supplemental_context))

    metadata = format_review_metadata_xml(review_context)
    if metadata:
        parts.append(metadata)

    if conversation is not None:
        parts.append(format_conversation_xml(conversation))

    if diffs is not None:
        parts.append(format_files_xml(diffs))

    parts.append("</ai_review_request>")
    return "\n".join(parts)


def build_agent_request_xml(
        *,
        instructions: str,
        supplemental_context: str,
        agent_mode: str,
        task_output_format: str,
        task: str,
        agent_history: str,
) -> str:
    parts = ["<ai_review_agent_request>"]

    if instructions:
        parts.append(wrap_cdata_element("instructions", instructions))

    if supplemental_context:
        parts.append(wrap_cdata_element("context", supplemental_context))

    parts.append(f"<agent_mode>{escape_xml_text(agent_mode)}</agent_mode>")
    parts.append(wrap_cdata_element("task_output_format", task_output_format))
    parts.append(wrap_cdata_element("task", task))
    parts.append(agent_history)

    parts.append("</ai_review_agent_request>")
    return "\n".join(parts)
