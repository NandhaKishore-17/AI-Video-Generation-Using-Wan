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
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Send *prompt* to Ollama and return the full response string.

        Retries up to _MAX_RETRIES times on connection / timeout errors.
        Raises OllamaError on permanent failure.
        """
        url = f"{self.base_url}/api/generate"
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "Ollama request attempt %d/%d (model=%s)",
                    attempt,
                    _MAX_RETRIES,
                    self.model,
                )
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                response_text: str = data.get("response", "")
                if not response_text:
                    raise OllamaError("Ollama returned an empty response")
                logger.info("Ollama responded successfully")
                return response_text
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                logger.warning("Ollama connection error (attempt %d): %s", attempt, exc)
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning("Ollama timeout (attempt %d)", attempt)
            except requests.exceptions.HTTPError as exc:
                # Non-retryable HTTP errors (e.g. 404 – model not found)
                raise OllamaError(f"Ollama HTTP error: {exc}") from exc
            except (KeyError, ValueError) as exc:
                raise OllamaError(f"Unexpected Ollama response format: {exc}") from exc

            # Exponential back-off before retry
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.info("Retrying in %ds …", wait)
                time.sleep(wait)

        raise OllamaError(
            f"Ollama unreachable after {_MAX_RETRIES} attempts: {last_error}"
        )


# Module-level singleton
ollama_client = OllamaClient()
