"""Thin wrapper around the OpenAI Responses API."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from novel_factory.config import AppConfig

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LlmRequestError(RuntimeError):
    """Raised when an LLM request fails after retries."""


class OpenAIResponsesClient:
    """Encapsulates the official OpenAI SDK for text and structured calls."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, timeout=config.request_timeout_seconds)

    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int = 5_000,
        model_override: str = "",
    ) -> str:
        """Generates free-form text with the Responses API."""

        model = model_override or self.config.model

        def _request() -> str:
            request_kwargs = {
                "model": model,
                "instructions": system_prompt,
                "input": user_prompt,
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": max_output_tokens,
                "metadata": {"task_name": task_name},
                "store": False,
                "truncation": "auto",
            }
            if self._supports_temperature(model):
                request_kwargs["temperature"] = temperature

            response = self.client.responses.create(
                **request_kwargs,
            )
            output_text = (response.output_text or "").strip()
            if not output_text:
                raise LlmRequestError(f"Empty response body for task '{task_name}'.")
            return output_text

        return self._with_retries(task_name=task_name, callback=_request)

    def structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int = 4_000,
        verbosity: str = "medium",
        model_override: str = "",
    ) -> SchemaT:
        """Generates structured output and parses it into a Pydantic model."""

        model = model_override or self.config.model

        def _request() -> SchemaT:
            request_kwargs = {
                "model": model,
                "instructions": system_prompt,
                "input": user_prompt,
                "text_format": schema,
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": max_output_tokens,
                "metadata": {"task_name": task_name},
                "text": {"verbosity": verbosity},
                "store": False,
                "truncation": "auto",
            }
            if self._supports_temperature(model):
                request_kwargs["temperature"] = temperature

            try:
                response = self.client.responses.parse(**request_kwargs)
                parsed = response.output_parsed
                if parsed is None:
                    raise LlmRequestError(f"Empty parsed body for task '{task_name}'.")
                return parsed
            except Exception as parse_error:  # noqa: BLE001
                logger.warning(
                    "LLM task %s parse path returned malformed structured output; using JSON fallback: %s",
                    task_name,
                    parse_error,
                )
                return self._structured_json_fallback(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    task_name=task_name,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    max_output_tokens=max(max_output_tokens, 5_000),
                    verbosity=verbosity,
                    model=model,
                )

        return self._with_retries(task_name=task_name, callback=_request)

    def _structured_json_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int,
        verbosity: str,
        model: str,
    ) -> SchemaT:
        """Retries a structured task with JSON-only output and local validation."""

        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True, separators=(",", ":"))
        fallback_input = (
            f"{user_prompt}\n\n"
            "Return only a valid JSON object. Do not wrap it in markdown fences. "
            "Every required field must be present.\n"
            f"JSON schema:\n{schema_json}"
        )
        request_kwargs = {
            "model": model,
            "instructions": f"{system_prompt}\n\nReturn only a valid JSON object with no surrounding prose.",
            "input": fallback_input,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "metadata": {"task_name": f"{task_name}_json_fallback"},
            "text": {"verbosity": verbosity},
            "store": False,
            "truncation": "auto",
        }
        if self._supports_temperature(model):
            request_kwargs["temperature"] = temperature

        response = self.client.responses.create(**request_kwargs)
        output_text = (response.output_text or "").strip()
        if not output_text:
            raise LlmRequestError(f"Empty JSON fallback body for task '{task_name}'.")
        return schema.model_validate_json(self._extract_json_object(output_text))

    def _with_retries(self, *, task_name: str, callback: Callable[[], SchemaT | str]) -> SchemaT | str:
        """Retries transient request failures with exponential backoff."""

        attempts = self.config.retry_attempts
        base_delay = self.config.retry_base_delay_seconds
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                logger.info("LLM task %s attempt %s/%s", task_name, attempt, attempts)
                return callback()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts:
                    break
                sleep_seconds = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "LLM task %s failed on attempt %s/%s: %s",
                    task_name,
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(sleep_seconds)

        raise LlmRequestError(f"LLM task '{task_name}' failed after {attempts} attempts.") from last_error

    def _supports_temperature(self, model: str) -> bool:
        """Returns whether the configured model accepts sampling temperature."""

        return not model.lower().startswith("gpt-5")

    def _extract_json_object(self, text: str) -> str:
        """Extracts a JSON object from a response that may include wrapper text."""

        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise LlmRequestError("JSON fallback did not return a parseable JSON object.")
        return stripped[start : end + 1]
