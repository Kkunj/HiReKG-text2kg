""" LLM Client for OpenAI API and self-hosted models, plus local embeddings """

import json
import logging
import os
import time
import requests
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, OpenAIError, RateLimitError
from google import genai
from google.genai import types as genai_types


load_dotenv()


def strip_thinking(response: str) -> str:
    """
    Remove <think>...</think> blocks from LLM response.
    Some local models (e.g., Nemotron) include reasoning in <think> tags.
    This function returns only the text after </think> tag.
    """
    if '</think>' in response:
        return response.split('</think>', 1)[1].strip()
    return response.strip()


# A/B-test hooks: module-level metrics ledger + thinking-toggle env var.
# KG_DISABLE_THINKING=1 prepends `/no_think` to the system prompt (Nemotron convention).
CALL_METRICS: List[Dict[str, Any]] = []


class LLMClient:
    """
    Unified LLM client supporting OpenAI, Gemini, and self-hosted models.

    Model routing:
    - model_type="openai"  → OpenAI API (responses endpoint)
    - model_type="gemini"  → Google Gemini API (google-genai SDK)
    - model_type="local"   → Self-hosted endpoint (chat completions)
    - model_type="nim"     → NVIDIA NIM endpoint (chat completions via OpenAI SDK)
    """

    # Self-hosted endpoint configuration
    SELF_HOSTED_URL = os.getenv("LOCAL_LLM_URL", "http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions")

    def __init__(
        self,
        model: Optional[str] = "gpt-4o-2024-08-06",
        model_type: Optional[str] = "openai",  # "local" or "openai" - explicit backend selection
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 16384,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger("KGPipeline")

        if self.model_type not in ("local", "openai", "gemini", "nim"):
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'local', 'openai', 'gemini', or 'nim'.")

        if self.model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Missing OPENAI_API_KEY. Provide it via environment variable or .env file."
                )
            self.client = OpenAI(api_key=api_key)

        if self.model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Missing GEMINI_API_KEY. Provide it via environment variable or .env file."
                )
            self.gemini_client = genai.Client(api_key=api_key)

        if self.model_type == "nim":
            nim_url = os.getenv("NVIDIA_ENDPOINT_URL")
            nim_key = os.getenv("NVIDIA_ENDPOINT_API")
            if not nim_url or not nim_key:
                raise ValueError(
                    "Missing NVIDIA_ENDPOINT_URL or NVIDIA_ENDPOINT_API. "
                    "Provide them via environment variables or .env file."
                )
            self.nim_client = OpenAI(base_url=nim_url, api_key=nim_key)

        backend_display = {"local": "Local LLM", "openai": "OpenAI", "gemini": "Google Gemini", "nim": "NVIDIA NIM"}[self.model_type]
        self.logger.debug(f"[LLMClient] Initialized with model: {self.model} (backend: {backend_display})")



    def _run_request(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Route request on the basis of model type.
        """
        if self.model_type == "openai":
            return self._run_openai_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
            )
        elif self.model_type == "gemini":
            return self._run_gemini_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
            )
        elif self.model_type == "nim":
            return self._run_nim_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
            )
        else:
            return self._run_self_hosted_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
            )

    def _run_openai_request(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Execute request using OpenAI API (responses endpoint).
        """
        desired_format = response_format
        include_response_format = desired_format is not None
        response_format = desired_format or {"type": "text"}
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < self.max_retries:
            try:
                request_payload: Dict[str, Any] = {
                    "model": self.model,
                    "input": [
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()},
                    ],
                    "temperature": self.temperature,
                    "max_output_tokens": max_output_tokens or self.max_output_tokens,
                }
                if include_response_format:
                    request_payload["response_format"] = response_format

                response = self.client.responses.create(**request_payload)
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self.logger.debug(f"[OpenAI] Tokens - Input: {input_tokens}, Output: {output_tokens}")
                return self._extract_text_openai(response)
            except TypeError as exc:
                if include_response_format and "response_format" in str(exc):
                    # Older client versions do not support response_format.
                    include_response_format = False
                    continue
                raise
            except (RateLimitError, APIConnectionError) as exc:
                last_error = exc
                attempt += 1
                backoff = 2**attempt
                self.logger.debug(f"[OpenAI] Rate limit/connection error, retrying in {backoff}s...")
                time.sleep(backoff)
            except OpenAIError as exc:
                raise RuntimeError(f"OpenAI API error: {exc}") from exc

        raise RuntimeError(f"Failed to get response from OpenAI after {self.max_retries} attempts: {last_error}")

    def _run_gemini_request(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Execute request using Google Gemini API (google-genai SDK).
        """
        attempt = 0
        last_error: Optional[Exception] = None
        tokens = max_output_tokens or self.max_output_tokens

        # Build config
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt.strip(),
            "temperature": self.temperature,
            "max_output_tokens": tokens,
            "thinking_config": genai_types.ThinkingConfig(thinking_level="HIGH"),
        }
        # Request JSON output when asked
        if response_format and response_format.get("type") == "json_object":
            config_kwargs["response_mime_type"] = "application/json"

        generate_config = genai_types.GenerateContentConfig(**config_kwargs)

        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_prompt.strip())],
            ),
        ]

        while attempt < self.max_retries:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generate_config,
                )

                # Log token usage if available
                if response.usage_metadata:
                    self.logger.debug(
                        f"[Gemini] Tokens - Input: {response.usage_metadata.prompt_token_count}, "
                        f"Output: {response.usage_metadata.candidates_token_count}"
                    )

                text = response.text or ""
                return text.strip()

            except Exception as exc:
                last_error = exc
                attempt += 1
                backoff = 2 ** attempt
                self.logger.debug(
                    f"[Gemini] Error (attempt {attempt}/{self.max_retries}), "
                    f"retrying in {backoff}s... {type(exc).__name__}: {str(exc)[:100]}"
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"Failed to get response from Gemini after {self.max_retries} attempts: {last_error}"
        )

    def _run_nim_request(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Execute request using NVIDIA NIM endpoint via OpenAI-compatible SDK.
        Returns only the actual output content (reasoning content is discarded).
        """
        attempt = 0
        last_error: Optional[Exception] = None
        tokens = max_output_tokens or self.max_output_tokens

        while attempt < self.max_retries:
            try:
                completion = self.nim_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()},
                    ],
                    temperature=self.temperature,
                    max_tokens=tokens,
                    stream=False,
                )

                message = completion.choices[0].message

                # Log token usage if available
                if completion.usage:
                    self.logger.debug(
                        f"[NIM] Tokens - Input: {completion.usage.prompt_tokens}, "
                        f"Output: {completion.usage.completion_tokens}"
                    )

                # NIM models may return reasoning separately via reasoning_content.
                # We only return the actual output content.
                content = message.content or ""
                return strip_thinking(content)

            except (RateLimitError, APIConnectionError) as exc:
                last_error = exc
                attempt += 1
                backoff = 2 ** attempt
                self.logger.debug(f"[NIM] Rate limit/connection error (attempt {attempt}/{self.max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
            except OpenAIError as exc:
                raise RuntimeError(f"NVIDIA NIM API error: {exc}") from exc

        raise RuntimeError(f"Failed to get response from NVIDIA NIM after {self.max_retries} attempts: {last_error}")

    def _run_self_hosted_request(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Execute request using self-hosted LLM endpoint (Qwen/Nemo).
        Uses standard chat completions format.
        """
        attempt = 0
        last_error: Optional[Exception] = None
        tokens = max_output_tokens or self.max_output_tokens

        disable_thinking = os.getenv("KG_DISABLE_THINKING") == "1"
        effective_system = system_prompt.strip()
        if disable_thinking:
            effective_system = "/no_think\n\n" + effective_system

        system_len = len(effective_system)
        user_len = len(user_prompt.strip())
        total_prompt_chars = system_len + user_len

        while attempt < self.max_retries:
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": effective_system},
                        {"role": "user", "content": user_prompt.strip()}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": tokens,
                }
                
                # Add response format hint for JSON if requested
                # Note: Self-hosted models may not support response_format natively
                # We rely on the prompt to instruct JSON output
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                self.logger.debug(
                    f"[SelfHosted] Request: URL={self.SELF_HOSTED_URL}, "
                    f"prompt_chars={total_prompt_chars}, max_tokens={tokens}, "
                    f"disable_thinking={disable_thinking}"
                )

                _t0 = time.time()
                response = requests.post(
                    self.SELF_HOSTED_URL,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=600
                )
                _latency = time.time() - _t0
                response.raise_for_status()

                result = response.json()
                choice = result['choices'][0]
                content = choice['message']['content']
                finish_reason = choice.get('finish_reason')

                usage = result.get('usage') or {}
                prompt_tokens = usage.get('prompt_tokens')
                completion_tokens = usage.get('completion_tokens')
                self.logger.debug(
                    f"[SelfHosted] Tokens - Input: {prompt_tokens}, "
                    f"Output: {completion_tokens}, finish_reason: {finish_reason}, "
                    f"latency_s: {_latency:.1f}"
                )

                if finish_reason == "length":
                    self.logger.warning(
                        f"[SelfHosted] Response TRUNCATED at max_tokens={tokens} "
                        f"(finish_reason='length'). JSON parsing will likely fail."
                    )

                has_think_open = "<think>" in content
                has_think_close = "</think>" in content
                self.logger.debug(
                    f"[SelfHosted] Raw content: {len(content)} chars, "
                    f"has_<think>={has_think_open}, has_</think>={has_think_close}"
                )

                stripped = strip_thinking(content)

                CALL_METRICS.append({
                    "disable_thinking": disable_thinking,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "has_think_open": has_think_open,
                    "has_think_close": has_think_close,
                    "raw_len": len(content),
                    "stripped_len": len(stripped),
                    "latency_s": round(_latency, 2),
                })
                self.logger.debug(
                    f"[SelfHosted] After strip_thinking: {len(stripped)} chars "
                    f"(removed {len(content) - len(stripped)} chars of reasoning)"
                )
                return stripped
                
            except requests.exceptions.Timeout as exc:
                last_error = exc
                attempt += 1
                backoff = 2**attempt
                self.logger.debug(f"[SelfHosted] Timeout (attempt {attempt}/{self.max_retries}), retrying in {backoff}s...")
                self.logger.debug(f"[SelfHosted] Timeout details: {type(exc).__name__}: {exc}")
                time.sleep(backoff)
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                attempt += 1
                backoff = 2**attempt
                self.logger.debug(f"[SelfHosted] Connection error (attempt {attempt}/{self.max_retries}), retrying in {backoff}s...")
                self.logger.debug(f"[SelfHosted] Connection error details: {type(exc).__name__}: {exc}")
                time.sleep(backoff)
            except requests.exceptions.RequestException as exc:
                self.logger.debug(f"[SelfHosted] Request error details: {type(exc).__name__}: {exc}")
                raise RuntimeError(f"Self-hosted LLM request error: {exc}") from exc
            except (KeyError, IndexError) as exc:
                self.logger.debug(f"[SelfHosted] Response parsing error: {type(exc).__name__}: {exc}")
                self.logger.debug(f"[SelfHosted] Raw response text: {response.text[:500] if hasattr(response, 'text') else 'N/A'}")
                raise RuntimeError(f"Error parsing self-hosted response: {exc}") from exc

        # Log final error details before raising
        self.logger.debug(f"[SelfHosted] FAILED after {self.max_retries} attempts")
        self.logger.debug(f"[SelfHosted] Last error type: {type(last_error).__name__}")
        self.logger.debug(f"[SelfHosted] Last error details: {last_error}")
        raise RuntimeError(f"Failed to get response from self-hosted LLM after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _extract_text_openai(response: Any) -> str:
        """
        Consolidate OpenAI Response API chunks into a single string.
        """
        parts = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        # Fallback for older shapes
        if not parts and hasattr(response, "output_text"):
            parts.append(response.output_text)
        if not parts and hasattr(response, "output_str"):
            parts.append(response.output_str)
        return "".join(parts).strip()

    def _fix_malformed_json(
        self,
        malformed_response: str,
        error_message: str,
        original_system_prompt: str,
        original_user_prompt: str,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Ask the LLM to fix malformed JSON by providing the error message and previous response.
        
        Args:
            malformed_response: The malformed JSON response from the LLM
            error_message: The JSON parsing error message
            original_system_prompt: The original system prompt
            original_user_prompt: The original user prompt
            max_output_tokens: Maximum tokens for output
        
        Returns:
            Fixed JSON string
        """
        fix_system_prompt = """You are a JSON repair expert. You will receive a malformed JSON response and an error message.
        Your task is to fix the JSON and return ONLY the corrected, valid JSON.

        Rules:
        - Return ONLY valid JSON, no explanations
        - Fix common issues: unterminated strings, missing quotes, trailing commas, etc.
        - Preserve the original content and structure as much as possible
        - Use double quotes for all strings
        - Ensure all strings are properly terminated
        - Remove any comments
        - No trailing commas in arrays or objects"""

        fix_user_prompt = f"""Original task context:
        SYSTEM: {original_system_prompt[:500]}...
        USER: {original_user_prompt[:500]}...

        The LLM generated this malformed JSON:
        ```
        {malformed_response[:2000]}
        ```

        Error message:
        {error_message}

        Please fix this JSON and return ONLY the corrected, valid JSON (no explanations, no markdown code blocks)."""

        try:
            fixed_response = self._run_request(
                system_prompt=fix_system_prompt,
                user_prompt=fix_user_prompt,
                response_format={"type": "json_object"},
                max_output_tokens=max_output_tokens,
            )
            return fixed_response
        except Exception as e:
            self.logger.debug(f"⚠️  JSON fix attempt failed: {str(e)}")
            raise

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a request where the model is instructed to return valid JSON.
        Retries up to 2 times if the response is not valid JSON.
        If JSON parsing fails, attempts to fix the malformed JSON by calling the LLM again.
        """
        max_json_retries = 2
        last_error = None
        last_raw_response = None
        
        # For self-hosted models, enhance the system prompt to enforce JSON output
        effective_system_prompt = system_prompt
        if self.model_type in ("local", "nim", "gemini"):
            json_instruction = "\n\nIMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations, just the JSON object."
            if "json" not in system_prompt.lower():
                effective_system_prompt = system_prompt + json_instruction
        
        for attempt in range(max_json_retries):
            raw = self._run_request(
                system_prompt=effective_system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
                max_output_tokens=max_output_tokens,
            )
            last_raw_response = raw
            cleaned = self._clean_json_response(raw)
            
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                last_error = exc
                error_msg = str(exc)
                self.logger.debug(f"⚠️  JSON parse error (attempt {attempt + 1}/{max_json_retries}): {error_msg}")
                
                if attempt < max_json_retries - 1:
                    # Try to fix the malformed JSON by asking the LLM
                    self.logger.debug(f"   → Attempting to fix malformed JSON with LLM...")
                    try:
                        fixed_raw = self._fix_malformed_json(
                            malformed_response=cleaned,
                            error_message=error_msg,
                            original_system_prompt=system_prompt,
                            original_user_prompt=user_prompt,
                            max_output_tokens=max_output_tokens,
                        )
                        fixed_cleaned = self._clean_json_response(fixed_raw)
                        result = json.loads(fixed_cleaned)
                        self.logger.debug(f"   ✓ Successfully fixed malformed JSON")
                        return result
                    except Exception as fix_error:
                        self.logger.debug(f"   ✗ JSON fix failed: {str(fix_error)[:100]}")
                        time.sleep(1)
                        continue
        
        # If all retries failed, raise the error
        raise ValueError(f"Model response was not valid JSON after {max_json_retries} attempts. Last error: {last_error}") from last_error


    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Remove Markdown code fences or leading labels before attempting to parse JSON.
        """
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:].lstrip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return cleaned.strip()


# ═════════════════════════════════════════════════════════════════════════════
#  Embedding Client — local sentence-transformers models (e.g. BGE-M3)
# ═════════════════════════════════════════════════════════════════════════════

class EmbeddingClient:
    """
    Unified embedding client.  Default backend is a local sentence-transformers
    model (BAAI/bge-m3) so no external API is required.

    Backends:
        - "local"  → sentence-transformers (runs on CPU/GPU, no API key needed)
        - "openai" → OpenAI Embeddings API (requires OPENAI_API_KEY)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        backend: str = "local",       # "local" or "openai"
        batch_size: int = 64,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model
        self.backend = backend
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger("KGPipeline")

        if backend == "local":
            from sentence_transformers import SentenceTransformer

            self.logger.debug(f"[EmbeddingClient] Loading local model: {model}")
            self._st_model = SentenceTransformer(model)
            self.logger.debug(
                f"[EmbeddingClient] Model loaded "
                f"(dim={self._st_model.get_sentence_embedding_dimension()})"
            )
        elif backend == "openai":
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY for OpenAI embeddings.")
            self._openai_client = OpenAI(api_key=api_key)
            self.logger.debug(f"[EmbeddingClient] Using OpenAI embeddings: {model}")
        else:
            raise ValueError(
                f"Invalid embedding backend '{backend}'. Must be 'local' or 'openai'."
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        if not texts:
            return []

        if self.backend == "local":
            embeddings = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self.logger.debug(f"[EmbeddingClient] Embedded {len(texts)} texts locally")
            return embeddings.tolist()

        # OpenAI backend
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._openai_client.embeddings.create(
                model=self.model_name, input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        self.logger.debug(f"[EmbeddingClient] Embedded {len(texts)} texts via OpenAI")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]
