"""
Ollama HTTP client.

Calls the Ollama REST API at /api/generate and returns the model's response text.
Supports configurable model, timeout, and retry with exponential back-off.
"""
import json
import logging
import time
from typing import Optional

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2  # seconds


class OllamaError(RuntimeError):
    """Raised when Ollama returns an error or is unreachable."""


class OllamaClient:
    """Thin wrapper around the Ollama /api/generate endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self._base_url = base_url
        self._model = model
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    @property
    def base_url(self) -> str:
        url = self._base_url or getattr(settings, "OLLAMA_URL", None) or settings.OLLAMA_BASE_URL
        return url.rstrip("/")

    @property
    def model(self) -> str:
        return self._model or getattr(settings, "OLLAMA_MODEL", settings.OLLAMA_MODEL)

    def generate(self, prompt: str, system: Optional[str] = None, format: Optional[str] = None) -> str:
        """Send prompt to Ollama with streaming enabled to prevent read timeouts.

        Logs request start time, first token arrival time (TTFT), and total generation time.
        Retries up to _MAX_RETRIES ONLY for connection errors.
        Does NOT retry on read timeouts or once generation starts.
        """
        url = f"{self.base_url}/api/generate"
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": 16384,
                "num_predict": 4096
            }
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        timeout_val = self.timeout or getattr(settings, "OLLAMA_TIMEOUT", 600)
        # For stream=True, requests timeout=(connect_timeout, read_timeout)
        # We enforce a high read timeout (1800s) to prevent generation cutoffs for large contexts
        req_timeout = (30.0, max(1800.0, float(timeout_val)))
        last_error: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            start_time = time.time()
            first_token_time: Optional[float] = None
            response_chunks = []

            start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            print(f"Ollama request started at: {start_str} (attempt {attempt}/{_MAX_RETRIES}, model={self.model})")
            logger.info("Ollama request started at: %s (attempt %d/%d, model=%s)", start_str, attempt, _MAX_RETRIES, self.model)

            try:
                with requests.post(url, json=payload, stream=True, timeout=req_timeout) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text_part = chunk.get("response", "")
                        if text_part:
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft = first_token_time - start_time
                                print(f"First token received in {ttft:.2f}s")
                                logger.info("First token received in %.2fs", ttft)
                            response_chunks.append(text_part)

                        if chunk.get("done", False):
                            break

                full_text = "".join(response_chunks).strip()
                if not full_text:
                    raise OllamaError("Ollama returned an empty response")

                total_time = time.time() - start_time
                print(f"Total generation time: {total_time:.2f}s")
                logger.info("Total generation time: %.2fs", total_time)
                return full_text

            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                logger.warning("Ollama connection error (attempt %d): %s", attempt, exc)
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.info("Retrying connection in %ds ...", wait)
                    time.sleep(wait)
            except requests.exceptions.Timeout as exc:
                total_time = time.time() - start_time
                raise OllamaError(f"Ollama read timeout after {total_time:.2f}s (timeout={timeout_val}s)") from exc
            except requests.exceptions.HTTPError as exc:
                raise OllamaError(f"Ollama HTTP error: {exc}") from exc
            except OllamaError:
                raise
            except Exception as exc:
                raise OllamaError(f"Ollama error: {exc}") from exc

        raise OllamaError(
            f"Ollama connection failed after {_MAX_RETRIES} attempts: {last_error}"
        )


# Module-level singleton
ollama_client = OllamaClient()
