from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.prompt.schema import PromptContextSchema
from ai_review.services.prompt.xml import (
    build_review_request_xml,
    format_file_xml,
    format_files_xml,
    split_instruction_and_context_prompts,
    wrap_cdata_element,
)
from ai_review.services.vcs.types import ReviewCommentSchema, ReviewThreadSchema, ThreadKind, UserSchema


def test_split_instruction_and_context_prompts_when_enabled() -> None:
    instructions, context = split_instruction_and_context_prompts(
        ["DEFAULT", "SNIPPET"],
        split_supplemental_context=True,
    )
    assert instructions == "DEFAULT"
    assert context == "SNIPPET"


def test_split_instruction_and_context_prompts_when_disabled() -> None:
    instructions, context = split_instruction_and_context_prompts(
        ["ONLY"],
        split_supplemental_context=False,
    )
    assert instructions == "ONLY"
    assert context == ""


def test_wrap_cdata_element_escapes_invalid_sequences() -> None:
    result = wrap_cdata_element("content", "a]]>b")
    assert "]]]]><![CDATA[>" in result


def test_format_file_xml_includes_path_and_diff() -> None:
    diff = DiffFileSchema(file="src/foo.py", diff="+ added\n- removed")
    result = format_file_xml(diff)
    assert 'path="src/foo.py"' in result
    assert "+ added" in result
    assert "- removed" in result


def test_format_files_xml_wraps_multiple_files() -> None:
    diffs = [
        DiffFileSchema(file="a.py", diff="+ foo"),
        DiffFileSchema(file="b.py", diff="- bar"),
    ]
    result = format_files_xml(diffs)
    assert result.startswith("<diff>")
    assert 'path="a.py"' in result
    assert 'path="b.py"' in result


def test_build_review_request_xml_sections() -> None:
    context = PromptContextSchema(review_title="Fix bug", changed_files=["a.py"])
    result = build_review_request_xml(
        instructions="Review carefully.",
        supplemental_context="Import context here.",
        review_context=context,
        diffs=[DiffFileSchema(file="a.py", diff="+ x")],
    )

    assert result.startswith("<ai_review_request>")
    assert result.endswith("</ai_review_request>")
    assert "<instructions>" in result
    assert "Review carefully." in result
    assert "<context>" in result
    assert "Import context here." in result
    assert "<review_metadata>" in result
    assert "<review_title>Fix bug</review_title>" in result
    assert "<diff>" in result
    assert 'path="a.py"' in result


def test_build_review_request_xml_includes_conversation() -> None:
    thread = ReviewThreadSchema(
        id="t1",
        kind=ThreadKind.INLINE,
        comments=[ReviewCommentSchema(id=1, body="Please clarify", author=UserSchema(name="Alice"))],
    )
    result = build_review_request_xml(
        instructions="Reply",
        supplemental_context="",
        review_context=PromptContextSchema(),
        diffs=[],
        conversation=thread,
    )

    assert "<conversation>" in result
    assert 'author="Alice"' in result
    assert "Please clarify" in result
