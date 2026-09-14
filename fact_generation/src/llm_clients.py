"""
LLM Clients for Atomic Fact Generation Pipeline.

Thin wrappers around OpenAI (GPT-5) and Google Gemini APIs with:
- Exponential backoff retry
- JSON response parsing with one-retry hardening
- Structured logging
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, OpenAIError, RateLimitError
from google import genai
from google.genai import types as genai_types

load_dotenv()

logger = logging.getLogger("atomic_facts")


def _clean_json_response(text: str) -> str:
    """Strip markdown fences and leading labels from a JSON response."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


class OpenAIClient:
    """GPT-5 wrapper using the OpenAI Responses API."""

    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 1.0,
        reasoning_effort: str = "high",
        max_output_tokens: int = 16000,
        max_retries: int = 3,
        backoff_base: int = 2,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Call GPT-5 and parse the response as JSON.
        On JSON parse failure, retries once with a stricter prompt suffix.
        """
        raw = self._call(system_prompt, user_prompt)
        if not raw:
            logger.warning(f"[OpenAI] generate_json got empty response from _call")
        else:
            logger.debug(f"[OpenAI] generate_json raw response ({len(raw)} chars): {raw[:200]}...")
        cleaned = _clean_json_response(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_err:
            logger.warning(f"JSON parse failed on first attempt: {first_err}. Raw ({len(raw)} chars): {raw[:300]}")
            # Retry with stricter suffix
            strict_suffix = (
                "\n\nYour previous response was not valid JSON. "
                "Respond with ONLY a valid JSON object. No markdown, no commentary."
            )
            raw2 = self._call(system_prompt, user_prompt + strict_suffix)
            cleaned2 = _clean_json_response(raw2)
            try:
                return json.loads(cleaned2)
            except json.JSONDecodeError as second_err:
                logger.error(f"JSON parse failed on retry: {second_err}")
                raise ValueError(
                    f"Model response was not valid JSON after retry. "
                    f"First error: {first_err}. Second error: {second_err}. "
                    f"Raw response (truncated): {raw2[:500]}"
                ) from second_err

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Raw API call with retry logic."""
        prompt_chars = len(system_prompt) + len(user_prompt)
        logger.debug(f"[OpenAI] Calling {self.model}, prompt_chars={prompt_chars}, temp={self.temperature}")
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()},
                    ],
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                # Log response metadata
                logger.debug(f"[OpenAI] Response id={getattr(response, 'id', 'N/A')}, status={getattr(response, 'status', 'N/A')}")
                if response.usage:
                    logger.debug(
                        f"[OpenAI] Tokens — input: {response.usage.input_tokens}, "
                        f"output: {response.usage.output_tokens}"
                    )
                text = self._extract_text(response)
                if not text:
                    logger.warning(
                        f"[OpenAI] Empty text extracted. "
                        f"output type={type(getattr(response, 'output', None))}, "
                        f"output len={len(getattr(response, 'output', None) or [])}, "
                        f"output_text={repr(getattr(response, 'output_text', 'N/A'))[:200]}, "
                        f"raw output items: {self._dump_output_structure(response)}"
                    )
                else:
                    logger.debug(f"[OpenAI] Extracted {len(text)} chars")
                return text
            except (RateLimitError, APIConnectionError) as exc:
                last_error = exc
                wait = self.backoff_base ** (attempt + 1)
                logger.warning(f"[OpenAI] {type(exc).__name__}, retry in {wait}s (attempt {attempt+1}/{self.max_retries})")
                time.sleep(wait)
            except OpenAIError as exc:
                logger.error(f"[OpenAI] API error: {type(exc).__name__}: {exc}")
                raise RuntimeError(f"OpenAI API error: {exc}") from exc
        raise RuntimeError(f"OpenAI failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _dump_output_structure(response: Any) -> str:
        """Dump the response output structure for debugging."""
        try:
            output = getattr(response, "output", None)
            if output is None:
                return "output=None"
            items = []
            for i, item in enumerate(output):
                item_type = getattr(item, "type", "?")
                content = getattr(item, "content", None)
                if content is None:
                    items.append(f"[{i}] type={item_type}, content=None")
                else:
                    parts = []
                    for j, c in enumerate(content):
                        c_type = getattr(c, "type", "?")
                        c_text = getattr(c, "text", None)
                        parts.append(f"type={c_type}, text_len={len(c_text) if c_text else 0}")
                    items.append(f"[{i}] type={item_type}, content=[{', '.join(parts)}]")
            return "; ".join(items)
        except Exception as e:
            return f"dump_error: {e}"

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from OpenAI Responses API output."""
        parts = []
        for item in (getattr(response, "output", None) or []):
            for content in (getattr(item, "content", None) or []):
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        if not parts and hasattr(response, "output_text"):
            parts.append(response.output_text)
        return "".join(parts).strip()


class GeminiClient:
    """Gemini wrapper using the google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0,
        top_p: float = 1,
        max_output_tokens: int = 4096,
        max_retries: int = 3,
        backoff_base: int = 2,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Call Gemini and parse the response as JSON.
        On JSON parse failure, retries once with a stricter prompt suffix.
        """
        raw = self._call(system_prompt, user_prompt)
        cleaned = _clean_json_response(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_err:
            logger.warning(f"[Gemini] JSON parse failed on first attempt: {first_err}")
            strict_suffix = (
                "\n\nYour previous response was not valid JSON. "
                "Respond with ONLY a valid JSON object. No markdown, no commentary."
            )
            raw2 = self._call(system_prompt, user_prompt + strict_suffix)
            cleaned2 = _clean_json_response(raw2)
            try:
                return json.loads(cleaned2)
            except json.JSONDecodeError as second_err:
                logger.error(f"[Gemini] JSON parse failed on retry: {second_err}")
                raise ValueError(
                    f"Gemini response was not valid JSON after retry. "
                    f"First error: {first_err}. Second error: {second_err}. "
                    f"Raw response (truncated): {raw2[:500]}"
                ) from second_err

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Raw API call with retry logic."""
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt.strip(),
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_prompt.strip())],
            ),
        ]
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                if response.usage_metadata:
                    logger.debug(
                        f"[Gemini] Tokens — input: {response.usage_metadata.prompt_token_count}, "
                        f"output: {response.usage_metadata.candidates_token_count}"
                    )
                return (response.text or "").strip()
            except Exception as exc:
                last_error = exc
                wait = self.backoff_base ** (attempt + 1)
                logger.warning(
                    f"[Gemini] {type(exc).__name__}, retry in {wait}s "
                    f"(attempt {attempt+1}/{self.max_retries}): {str(exc)[:120]}"
                )
                time.sleep(wait)
        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last_error}")
