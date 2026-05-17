import json
import re
from typing import TypeVar, Generic, Type

from pydantic import BaseModel, ValidationError

from ai_review.libs.json import sanitize_json_string
from ai_review.libs.logger import get_logger

logger = get_logger("LLM_JSON_PARSER")

T = TypeVar("T", bound=BaseModel)

CLEAN_JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMOutputJSONParser(Generic[T]):
    """Reusable JSON parser for LLM responses."""

    def __init__(self, model: Type[T]):
        self.model = model
        self.model_name = self.model.__name__

    def try_parse(self, raw: str) -> T | None:
        logger.debug(f"[{self.model_name}] Attempting JSON parse (len={len(raw)})")

        try:
            return self.model.model_validate_json(raw)
        except ValidationError as error:
            logger.warning(f"[{self.model_name}] Raw JSON parse failed: {error}")
            cleaned = sanitize_json_string(raw)

            if cleaned != raw:
                logger.debug(f"[{self.model_name}] Sanitized JSON differs, retrying parse...")
                try:
                    return self.model.model_validate_json(cleaned)
                except ValidationError as error:
                    logger.warning(f"[{self.model_name}] Sanitized JSON still invalid: {error}")
                    return None
            else:
                logger.debug(f"[{self.model_name}] Sanitized JSON identical — skipping retry")
                return None

    def parse_all_output(self, output: str) -> list[T]:
        """Extract and parse every JSON object present in the output.

        Handles models that emit multiple JSON objects in a single response
        despite being instructed to return only one.
        """
        output = (output or "").strip()
        if not output:
            return []

        if match := CLEAN_JSON_BLOCK_RE.search(output):
            output = match.group(1).strip()

        results: list[T] = []
        remaining = output.lstrip()
        decoder = json.JSONDecoder()

        while remaining:
            try:
                obj, idx = decoder.raw_decode(remaining)
            except json.JSONDecodeError:
                break

            try:
                results.append(self.model.model_validate_json(json.dumps(obj)))
            except ValidationError as error:
                logger.warning(f"[{self.model_name}] Skipping invalid object during multi-parse: {error}")

            remaining = remaining[idx:].lstrip()

        if results:
            logger.debug(f"[{self.model_name}] Parsed {len(results)} object(s) from output")

        return results

    def parse_output(self, output: str) -> T | None:
        output = (output or "").strip()
        if not output:
            logger.warning(f"[{self.model_name}] Empty LLM output")
            return None

        logger.debug(f"[{self.model_name}] Parsing output (len={len(output)})")

        if match := CLEAN_JSON_BLOCK_RE.search(output):
            logger.debug(f"[{self.model_name}] Found fenced JSON block, extracting...")
            output = match.group(1).strip()

        if parsed := self.try_parse(output):
            logger.info(f"[{self.model_name}] Successfully parsed")
            return parsed

        # Fallback: take just the first object (handles trailing multi-object responses)
        if results := self.parse_all_output(output):
            logger.info(f"[{self.model_name}] Successfully parsed first JSON object from multi-object response")
            return results[0]

        logger.error(f"[{self.model_name}] No valid JSON found in output")
        return None
