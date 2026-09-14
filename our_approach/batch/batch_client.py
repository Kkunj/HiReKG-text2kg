"""
OpenAI Batch API client.

Thin wrapper around the Batch API lifecycle (upload -> submit -> poll -> download)
tailored for this pipeline. Supports the three endpoints we need:
    - /v1/responses          (chat-style LLM calls matching llm_client._run_openai_request)
    - /v1/chat/completions   (fallback / alternative)
    - /v1/embeddings         (for semantic linking phase)

State (file ids, batch ids, status) is persisted to disk so long-running jobs
can resume after a process restart.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


# ---------------------------------------------------------------------------
# Limits per OpenAI Batch API docs
# ---------------------------------------------------------------------------
MAX_REQUESTS_PER_BATCH = 50_000
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB

DEFAULT_POLL_INTERVAL_SECONDS = 60
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BatchRequest:
    """A single line in a batch .jsonl file."""
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]

    def to_jsonl_line(self) -> str:
        return json.dumps(
            {
                "custom_id": self.custom_id,
                "method": self.method,
                "url": self.url,
                "body": self.body,
            },
            ensure_ascii=False,
        )


@dataclass
class BatchJobMeta:
    """Metadata about a submitted batch, persisted to disk so jobs can resume."""
    name: str
    endpoint: str
    input_file_path: str
    input_file_id: Optional[str] = None
    batch_id: Optional[str] = None
    status: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None
    request_counts: Dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BatchJobMeta":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class BatchResultEntry:
    """A parsed line from a batch output file."""
    custom_id: str
    success: bool
    content: Optional[str] = None          # extracted text for LLM endpoints
    embedding: Optional[List[float]] = None  # for /v1/embeddings
    raw_body: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------
def build_responses_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 10_000,
    json_mode: bool = True,
) -> BatchRequest:
    """
    Build a batch request targeting /v1/responses (mirrors llm_client._run_openai_request).
    """
    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if json_mode:
        # /v1/responses uses text.format (the Chat Completions `response_format`
        # equivalent). The OpenAI SDK translates it on sync calls, but the Batch
        # API validates the raw body and rejects `response_format`.
        body["text"] = {"format": {"type": "json_object"}}
    return BatchRequest(custom_id=custom_id, method="POST", url="/v1/responses", body=body)


def build_chat_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 10_000,
    json_mode: bool = True,
) -> BatchRequest:
    """
    Build a batch request targeting /v1/chat/completions.
    """
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return BatchRequest(custom_id=custom_id, method="POST", url="/v1/chat/completions", body=body)


def build_embeddings_request(
    custom_id: str,
    model: str,
    text: str,
) -> BatchRequest:
    """
    Build a batch request targeting /v1/embeddings. One text per request.
    """
    body = {"model": model, "input": text}
    return BatchRequest(custom_id=custom_id, method="POST", url="/v1/embeddings", body=body)


# ---------------------------------------------------------------------------
# Content extractors (one per endpoint)
# ---------------------------------------------------------------------------
def _extract_responses_content(body: Dict[str, Any]) -> str:
    """
    Extract text from a /v1/responses body. Mirrors LLMClient._extract_text_openai
    but works against the raw dict shape returned in batch output.
    """
    parts: List[str] = []
    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if text:
                parts.append(text)
    if not parts and "output_text" in body:
        parts.append(body["output_text"])
    return "".join(parts).strip()


def _extract_chat_content(body: Dict[str, Any]) -> str:
    """Extract assistant content from a /v1/chat/completions body."""
    choices = body.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "") or ""


def _extract_embedding(body: Dict[str, Any]) -> Optional[List[float]]:
    """Extract the single embedding vector from a /v1/embeddings body."""
    data = body.get("data") or []
    if not data:
        return None
    return data[0].get("embedding")


# ---------------------------------------------------------------------------
# Batch client
# ---------------------------------------------------------------------------
class BatchClient:
    """
    Manages the OpenAI Batch API lifecycle for this pipeline.

    A "job" corresponds to one uploaded .jsonl file and one submitted batch.
    All artifacts (input file, metadata, output, errors) live under a single
    directory keyed by the job name so runs are resumable.
    """

    def __init__(
        self,
        workdir: Path,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        logger: Optional[logging.Logger] = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. Provide it via environment variable or .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval_seconds
        self.logger = logger or logging.getLogger("KGPipeline")

    # ----- paths -----
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

    def _error_path(self, job_name: str) -> Path:
        return self._job_dir(job_name) / "errors.jsonl"

    # ----- step 1: write input file -----
    def write_input_file(self, job_name: str, requests: Iterable[BatchRequest]) -> Path:
        """Serialize requests to a .jsonl file and validate size/count limits."""
        path = self._input_path(job_name)
        seen_ids: set = set()
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for req in requests:
                if req.custom_id in seen_ids:
                    raise ValueError(f"Duplicate custom_id in batch '{job_name}': {req.custom_id}")
                seen_ids.add(req.custom_id)
                f.write(req.to_jsonl_line() + "\n")
                count += 1

        size = path.stat().st_size
        if count == 0:
            raise ValueError(f"Batch '{job_name}' has 0 requests — refusing to submit empty batch.")
        if count > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"Batch '{job_name}' has {count} requests (max {MAX_REQUESTS_PER_BATCH})."
            )
        if size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"Batch '{job_name}' file is {size} bytes (max {MAX_FILE_SIZE_BYTES})."
            )
        self.logger.info(f"[Batch:{job_name}] Wrote {count} requests ({size:,} bytes) to {path}")
        return path

    # ----- step 2: submit -----
    def submit(self, job_name: str, endpoint: str, metadata: Optional[Dict[str, str]] = None) -> BatchJobMeta:
        """
        Upload the input file and create the batch. Returns job metadata (persisted to disk).
        If a batch has already been submitted under this name, returns the existing metadata.
        """
        meta_path = self._meta_path(job_name)
        if meta_path.exists():
            existing = BatchJobMeta.load(meta_path)
            if existing.batch_id:
                self.logger.info(
                    f"[Batch:{job_name}] Already submitted (batch_id={existing.batch_id}); reusing."
                )
                return existing

        input_path = self._input_path(job_name)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found for job '{job_name}': {input_path}. "
                f"Call write_input_file() first."
            )

        self.logger.info(f"[Batch:{job_name}] Uploading input file...")
        with open(input_path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")

        self.logger.info(f"[Batch:{job_name}] Creating batch (endpoint={endpoint})...")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata=metadata or {"job_name": job_name},
        )

        meta = BatchJobMeta(
            name=job_name,
            endpoint=endpoint,
            input_file_path=str(input_path),
            input_file_id=uploaded.id,
            batch_id=batch.id,
            status=batch.status,
            submitted_at=time.time(),
        )
        meta.save(meta_path)
        self.logger.info(
            f"[Batch:{job_name}] Submitted: batch_id={batch.id}, status={batch.status}"
        )
        return meta

    # ----- step 3: poll -----
    def wait(self, job_name: str) -> BatchJobMeta:
        """Poll until the batch reaches a terminal status. Returns updated metadata."""
        meta_path = self._meta_path(job_name)
        meta = BatchJobMeta.load(meta_path)
        if not meta.batch_id:
            raise RuntimeError(f"Job '{job_name}' has no batch_id — submit first.")

        self.logger.info(
            f"[Batch:{job_name}] Polling every {self.poll_interval}s (batch_id={meta.batch_id})..."
        )
        while True:
            batch = self.client.batches.retrieve(meta.batch_id)
            meta.status = batch.status
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                meta.request_counts = {
                    "total": getattr(counts, "total", 0),
                    "completed": getattr(counts, "completed", 0),
                    "failed": getattr(counts, "failed", 0),
                }
            if batch.status in TERMINAL_STATUSES:
                meta.output_file_id = getattr(batch, "output_file_id", None)
                meta.error_file_id = getattr(batch, "error_file_id", None)
                meta.completed_at = time.time()
                meta.save(meta_path)
                self.logger.info(
                    f"[Batch:{job_name}] Terminal status: {batch.status} "
                    f"(completed={meta.request_counts.get('completed', 0)}, "
                    f"failed={meta.request_counts.get('failed', 0)})"
                )
                return meta

            meta.save(meta_path)
            self.logger.debug(
                f"[Batch:{job_name}] status={batch.status} "
                f"completed={meta.request_counts.get('completed', 0)}/"
                f"{meta.request_counts.get('total', 0)}"
            )
            time.sleep(self.poll_interval)

    # ----- step 4: download -----
    def download(self, job_name: str) -> Tuple[Path, Optional[Path]]:
        """
        Download output + (optional) error files. Returns (output_path, error_path).
        error_path is None if the batch produced no error file.
        """
        meta = BatchJobMeta.load(self._meta_path(job_name))
        if not meta.output_file_id and not meta.error_file_id:
            raise RuntimeError(
                f"Job '{job_name}' has no output_file_id or error_file_id. "
                f"Status: {meta.status}"
            )

        output_path = self._output_path(job_name)
        error_path: Optional[Path] = None

        if meta.output_file_id:
            self.logger.info(f"[Batch:{job_name}] Downloading output file...")
            content = self.client.files.content(meta.output_file_id).text
            output_path.write_text(content, encoding="utf-8")
        else:
            # Write an empty output file for consistent downstream handling
            output_path.write_text("", encoding="utf-8")

        if meta.error_file_id:
            error_path = self._error_path(job_name)
            self.logger.info(f"[Batch:{job_name}] Downloading error file...")
            err_content = self.client.files.content(meta.error_file_id).text
            error_path.write_text(err_content, encoding="utf-8")

        return output_path, error_path

    # ----- step 5: parse -----
    def parse_results(
        self,
        job_name: str,
        endpoint: str,
    ) -> Dict[str, BatchResultEntry]:
        """
        Parse output + errors into a dict keyed by custom_id. One entry per
        request from the original input, whether successful or failed.
        """
        extractor = self._extractor_for(endpoint)
        results: Dict[str, BatchResultEntry] = {}

        output_path = self._output_path(job_name)
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning(f"[Batch:{job_name}] Skipping malformed output line: {exc}")
                    continue
                entry = self._row_to_entry(row, extractor)
                results[entry.custom_id] = entry

        error_path = self._error_path(job_name)
        if error_path.exists():
            for line in error_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning(f"[Batch:{job_name}] Skipping malformed error line: {exc}")
                    continue
                entry = self._row_to_entry(row, extractor)
                # Error file entries overwrite any partial success entry for the same id
                results[entry.custom_id] = entry

        self.logger.info(
            f"[Batch:{job_name}] Parsed {len(results)} result entries "
            f"(success={sum(1 for e in results.values() if e.success)}, "
            f"failed={sum(1 for e in results.values() if not e.success)})"
        )
        return results

    @staticmethod
    def _extractor_for(endpoint: str) -> Callable[[Dict[str, Any]], Any]:
        if endpoint == "/v1/responses":
            return _extract_responses_content
        if endpoint == "/v1/chat/completions":
            return _extract_chat_content
        if endpoint == "/v1/embeddings":
            return _extract_embedding
        raise ValueError(f"Unsupported endpoint for parsing: {endpoint}")

    @staticmethod
    def _row_to_entry(
        row: Dict[str, Any],
        extractor: Callable[[Dict[str, Any]], Any],
    ) -> BatchResultEntry:
        custom_id = row.get("custom_id", "")
        error = row.get("error")
        response = row.get("response") or {}
        body = response.get("body") if isinstance(response, dict) else None
        status_code = response.get("status_code") if isinstance(response, dict) else None

        if error or not body or (status_code is not None and status_code >= 400):
            return BatchResultEntry(
                custom_id=custom_id,
                success=False,
                error=error or {"code": "no_body", "message": f"status_code={status_code}"},
                raw_body=body,
            )

        extracted = extractor(body)
        if isinstance(extracted, list):  # embedding vector
            return BatchResultEntry(
                custom_id=custom_id,
                success=True,
                embedding=extracted,
                raw_body=body,
            )
        return BatchResultEntry(
            custom_id=custom_id,
            success=True,
            content=extracted,
            raw_body=body,
        )

    # ----- convenience: one-shot -----
    def run_job(
        self,
        job_name: str,
        endpoint: str,
        requests: Iterable[BatchRequest],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, BatchResultEntry]:
        """
        End-to-end: write -> submit -> wait -> download -> parse.
        Resumable: if the job's meta.json already has a batch_id, skips re-submission.
        If output.jsonl already exists (from a prior completed run), skips polling/download.
        """
        meta_path = self._meta_path(job_name)
        output_path = self._output_path(job_name)

        already_downloaded = output_path.exists() and meta_path.exists()
        if not already_downloaded:
            if not self._input_path(job_name).exists():
                self.write_input_file(job_name, requests)
            self.submit(job_name, endpoint=endpoint, metadata=metadata)
            meta = self.wait(job_name)
            if meta.status != "completed" and not meta.output_file_id and not meta.error_file_id:
                raise RuntimeError(
                    f"Batch '{job_name}' ended in status '{meta.status}' with no output."
                )
            self.download(job_name)
        else:
            self.logger.info(
                f"[Batch:{job_name}] Output already present at {output_path}; skipping submit/wait/download."
            )

        return self.parse_results(job_name, endpoint=endpoint)
