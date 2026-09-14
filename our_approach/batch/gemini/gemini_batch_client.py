"""
Gemini Batch API client.

Mirrors batch/batch_client.py (OpenAI) but targets the Google Gemini Batch API.
Same lifecycle: write_input_file -> submit -> wait -> download -> parse_results,
with a convenience run_job() that chains them all.

Two request types are supported:
    - "generate"  (text generation via client.batches.create)
    - "embed"     (embeddings via client.batches.create_embeddings)

State (uploaded file names, batch names, status) is persisted to disk so
long-running jobs can resume after a process restart.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# Reuse the universal result type from the OpenAI batch client so that
# downstream pipeline code (which indexes by custom_id) works unchanged.
from batch.batch_client import BatchResultEntry


load_dotenv()

DEFAULT_POLL_INTERVAL_SECONDS = 60

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GeminiBatchRequest:
    """A single line in a Gemini batch .jsonl file."""

    key: str                    # maps to OpenAI's custom_id
    request: Dict[str, Any]     # full request body (contents, system_instruction, …)
    request_type: str = "generate"  # "generate" or "embed" (not serialised to JSONL)

    def to_jsonl_line(self) -> str:
        return json.dumps(
            {"key": self.key, "request": self.request},
            ensure_ascii=False,
        )


@dataclass
class GeminiBatchJobMeta:
    """Metadata about a submitted Gemini batch, persisted for resumability."""

    name: str
    model: str
    request_type: str                          # "generate" or "embed"
    input_file_path: str
    uploaded_file_name: Optional[str] = None   # Gemini File API name (files/…)
    batch_name: Optional[str] = None           # Gemini batch name (batches/…)
    state: Optional[str] = None                # JOB_STATE_*
    dest_file_name: Optional[str] = None       # output file in Gemini File API
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GeminiBatchJobMeta":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------
def build_generate_request(
    key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 10_000,
    json_mode: bool = True,
) -> GeminiBatchRequest:
    """
    Build a Gemini batch request for text generation.

    Note: the model is NOT included per-request — it is specified at batch
    creation time via client.batches.create(model=…).
    """
    gen_config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if json_mode:
        gen_config["response_mime_type"] = "application/json"

    request_body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt.strip()}],
            }
        ],
        "system_instruction": {
            "parts": [{"text": system_prompt.strip()}],
        },
        "generation_config": gen_config,
    }
    return GeminiBatchRequest(key=key, request=request_body, request_type="generate")


def build_embed_request(
    key: str,
    text: str,
) -> GeminiBatchRequest:
    """
    Build a Gemini batch request for embeddings.

    Model is specified at batch creation, not per-request.
    """
    request_body: Dict[str, Any] = {
        "content": {
            "parts": [{"text": text}],
        },
    }
    return GeminiBatchRequest(key=key, request=request_body, request_type="embed")


# ---------------------------------------------------------------------------
# Content extractors
# ---------------------------------------------------------------------------
def _extract_gemini_text(response: Dict[str, Any]) -> str:
    """
    Extract text from a Gemini generateContent response body.
    Path: response.candidates[0].content.parts[*].text
    """
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: List[str] = []
    for part in parts:
        t = part.get("text")
        if t:
            texts.append(t)
    return "".join(texts).strip()


def _extract_gemini_embedding(response: Dict[str, Any]) -> Optional[List[float]]:
    """
    Extract embedding vector from a Gemini embedContent response body.
    Path: response.embedding.values
    """
    embedding = response.get("embedding") or {}
    return embedding.get("values")


# ---------------------------------------------------------------------------
# Batch client
# ---------------------------------------------------------------------------
class GeminiBatchClient:
    """
    Manages the Gemini Batch API lifecycle for this pipeline.

    Interface mirrors batch.batch_client.BatchClient so that pipeline code
    can swap one for the other with minimal changes.
    """

    def __init__(
        self,
        workdir: Path,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        logger: Optional[logging.Logger] = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. Provide it via environment variable or .env file."
            )
        self.client = genai.Client(api_key=api_key)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval_seconds
        self.logger = logger or logging.getLogger("KGPipeline")

    # ---- paths ----
    def _job_dir(self, job_name: str) -> Path:
        d = self.workdir / job_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _input_path(self, job_name: str) -> Path:
        return self._job_dir(job_name) / "input.jsonl"

    def _meta_path(self, job_name: str) -> Path:
        return self._job_dir(job_name) / "meta.json"

    def _output_path(self, job_name: str) -> Path:
        return self._job_dir(job_name) / "output.jsonl"

    # ---- step 1: write input file ----
    def write_input_file(
        self,
        job_name: str,
        requests: Iterable[GeminiBatchRequest],
    ) -> Path:
        """Serialise requests to a .jsonl file. Validates unique keys."""
        path = self._input_path(job_name)
        seen_keys: set = set()
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for req in requests:
                if req.key in seen_keys:
                    raise ValueError(
                        f"Duplicate key in batch '{job_name}': {req.key}"
                    )
                seen_keys.add(req.key)
                f.write(req.to_jsonl_line() + "\n")
                count += 1

        size = path.stat().st_size
        if count == 0:
            raise ValueError(
                f"Batch '{job_name}' has 0 requests — refusing to submit empty batch."
            )
        self.logger.info(
            f"[GeminiBatch:{job_name}] Wrote {count} requests ({size:,} bytes) to {path}"
        )
        return path

    # ---- step 2: submit ----
    def submit(
        self,
        job_name: str,
        model: str,
        request_type: str = "generate",
    ) -> GeminiBatchJobMeta:
        """
        Upload the input file and create the batch.  Resumable: if a batch has
        already been submitted under this name, returns the existing metadata.
        """
        meta_path = self._meta_path(job_name)
        if meta_path.exists():
            existing = GeminiBatchJobMeta.load(meta_path)
            if existing.batch_name:
                self.logger.info(
                    f"[GeminiBatch:{job_name}] Already submitted "
                    f"(batch={existing.batch_name}); reusing."
                )
                return existing

        input_path = self._input_path(job_name)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found for job '{job_name}': {input_path}. "
                f"Call write_input_file() first."
            )

        # Upload file to Gemini File API
        self.logger.info(f"[GeminiBatch:{job_name}] Uploading input file...")
        uploaded = self.client.files.upload(
            file=str(input_path),
            config=genai_types.UploadFileConfig(
                display_name=f"{job_name}_input",
                mime_type="jsonl",
            ),
        )

        # Create batch — branch on request_type
        self.logger.info(
            f"[GeminiBatch:{job_name}] Creating batch "
            f"(model={model}, type={request_type})..."
        )
        batch_config = {"display_name": job_name}

        if request_type == "embed":
            batch_job = self.client.batches.create_embeddings(
                model=model,
                src=uploaded.name,
                config=batch_config,
            )
        else:
            batch_job = self.client.batches.create(
                model=model,
                src=uploaded.name,
                config=batch_config,
            )

        state_name = (
            batch_job.state.name
            if hasattr(batch_job.state, "name")
            else str(batch_job.state)
        )

        meta = GeminiBatchJobMeta(
            name=job_name,
            model=model,
            request_type=request_type,
            input_file_path=str(input_path),
            uploaded_file_name=uploaded.name,
            batch_name=batch_job.name,
            state=state_name,
            submitted_at=time.time(),
        )
        meta.save(meta_path)
        self.logger.info(
            f"[GeminiBatch:{job_name}] Submitted: batch={batch_job.name}, "
            f"state={state_name}"
        )
        return meta

    # ---- step 3: poll ----
    def wait(self, job_name: str) -> GeminiBatchJobMeta:
        """Poll until the batch reaches a terminal state. Returns updated metadata."""
        meta_path = self._meta_path(job_name)
        meta = GeminiBatchJobMeta.load(meta_path)
        if not meta.batch_name:
            raise RuntimeError(
                f"Job '{job_name}' has no batch_name — submit first."
            )

        self.logger.info(
            f"[GeminiBatch:{job_name}] Polling every {self.poll_interval}s "
            f"(batch={meta.batch_name})..."
        )
        while True:
            batch_job = self.client.batches.get(name=meta.batch_name)
            state_name = (
                batch_job.state.name
                if hasattr(batch_job.state, "name")
                else str(batch_job.state)
            )
            meta.state = state_name

            if state_name in TERMINAL_STATES:
                dest = getattr(batch_job, "dest", None)
                if dest:
                    meta.dest_file_name = getattr(dest, "file_name", None)
                meta.completed_at = time.time()
                meta.save(meta_path)

                stats = getattr(batch_job, "batch_stats", None)
                success = getattr(stats, "success_count", "?") if stats else "?"
                failed = getattr(stats, "failed_request_count", "?") if stats else "?"
                self.logger.info(
                    f"[GeminiBatch:{job_name}] Terminal state: {state_name} "
                    f"(success={success}, failed={failed})"
                )
                return meta

            elapsed = ""
            if meta.submitted_at:
                mins = (time.time() - meta.submitted_at) / 60
                elapsed = f" elapsed={mins:.0f}m"
            meta.save(meta_path)
            self.logger.info(
                f"[GeminiBatch:{job_name}] state={state_name}{elapsed}"
            )
            time.sleep(self.poll_interval)

    # ---- step 4: download ----
    def download(self, job_name: str) -> Path:
        """
        Download the output JSONL file.  Returns the local output path.

        Unlike OpenAI, Gemini has no separate error file — errors appear
        inline as {"key": "…", "error": {…}} in the same output.
        """
        meta = GeminiBatchJobMeta.load(self._meta_path(job_name))
        if not meta.dest_file_name:
            raise RuntimeError(
                f"Job '{job_name}' has no dest_file_name. State: {meta.state}"
            )

        output_path = self._output_path(job_name)
        self.logger.info(f"[GeminiBatch:{job_name}] Downloading output file...")
        content = self.client.files.download(file=meta.dest_file_name)

        # SDK may return bytes or str
        if isinstance(content, bytes):
            output_path.write_bytes(content)
        else:
            output_path.write_text(str(content), encoding="utf-8")

        return output_path

    # ---- step 5: parse ----
    def parse_results(
        self,
        job_name: str,
        request_type: str = "generate",
    ) -> Dict[str, BatchResultEntry]:
        """
        Parse output JSONL into a dict keyed by request key.

        Returns BatchResultEntry (from batch_client) so downstream pipeline
        code that indexes by custom_id works unchanged — we map Gemini's
        ``key`` field into ``BatchResultEntry.custom_id``.
        """
        extractor = (
            _extract_gemini_embedding
            if request_type == "embed"
            else _extract_gemini_text
        )
        results: Dict[str, BatchResultEntry] = {}

        output_path = self._output_path(job_name)
        if not output_path.exists():
            self.logger.warning(
                f"[GeminiBatch:{job_name}] No output file found at {output_path}."
            )
            return results

        raw = output_path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    f"[GeminiBatch:{job_name}] Skipping malformed line: {exc}"
                )
                continue

            key = row.get("key", "")
            error = row.get("error")
            response = row.get("response")

            if error or response is None:
                results[key] = BatchResultEntry(
                    custom_id=key,
                    success=False,
                    error=error or {
                        "code": "no_response",
                        "message": "Missing response in output",
                    },
                    raw_body=response,
                )
                continue

            extracted = extractor(response)
            if isinstance(extracted, list):  # embedding vector
                results[key] = BatchResultEntry(
                    custom_id=key,
                    success=True,
                    embedding=extracted,
                    raw_body=response,
                )
            else:
                results[key] = BatchResultEntry(
                    custom_id=key,
                    success=True,
                    content=extracted,
                    raw_body=response,
                )

        success_count = sum(1 for e in results.values() if e.success)
        self.logger.info(
            f"[GeminiBatch:{job_name}] Parsed {len(results)} entries "
            f"(success={success_count}, failed={len(results) - success_count})"
        )
        return results

    # ---- convenience: one-shot ----
    def run_job(
        self,
        job_name: str,
        model: str,
        request_type: str,
        requests: Iterable[GeminiBatchRequest],
    ) -> Dict[str, BatchResultEntry]:
        """
        End-to-end: write -> submit -> wait -> download -> parse.
        Resumable: if meta.json already records a batch_name, skips re-submission.
        If output.jsonl already exists, skips polling/download entirely.
        """
        meta_path = self._meta_path(job_name)
        output_path = self._output_path(job_name)

        already_downloaded = output_path.exists() and meta_path.exists()
        if not already_downloaded:
            if not self._input_path(job_name).exists():
                self.write_input_file(job_name, requests)
            self.submit(job_name, model=model, request_type=request_type)
            meta = self.wait(job_name)
            if meta.state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(
                    f"Batch '{job_name}' ended in state '{meta.state}'."
                )
            self.download(job_name)
        else:
            self.logger.info(
                f"[GeminiBatch:{job_name}] Output already present at "
                f"{output_path}; skipping submit/wait/download."
            )

        return self.parse_results(job_name, request_type=request_type)
