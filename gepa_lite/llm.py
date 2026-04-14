"""LLM utilities with retry logic, rate limiting, and error handling."""

import json
import logging
import re
import time
from threading import Lock
from typing import Any, Dict, List, Optional

from openai import OpenAI

from gepa_lite.config import Config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, min_interval: float = 0.0):
        self._min_interval = min_interval
        self._last_call_time = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self._min_interval <= 0:
            return

        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self._last_call_time = time.time()


class LLMClient:
    """Wrapper around OpenAI client with retry logic and rate limiting."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        config: Optional[Config] = None,
    ):
        self.client = client or OpenAI()
        self.config = config or Config()
        self._rate_limiter = RateLimiter(self.config.api_call_delay)
        self._call_count = 0
        self._cache_hits = 0
        self._lock = Lock()

    @property
    def stats(self) -> Dict[str, int]:
        """Return call statistics."""
        with self._lock:
            return {
                "api_calls": self._call_count,
                "cache_hits": self._cache_hits,
            }

    def _increment_calls(self) -> None:
        with self._lock:
            self._call_count += 1

    def _increment_cache_hits(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def call_json(
        self,
        prompt: str,
        *,
        model: str,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response with retries and exponential backoff."""
        retries = max_retries if max_retries is not None else self.config.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                self._rate_limiter.wait()
                self._increment_calls()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)

            except json.JSONDecodeError as exc:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {exc}")
                last_error = exc

            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{retries + 1}): {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(f"JSON LLM call failed after {retries + 1} attempts: {last_error}")

    def call_text(
        self,
        prompt: str,
        *,
        model: str,
        max_retries: Optional[int] = None,
    ) -> str:
        """Call LLM and return text response with retries and exponential backoff."""
        retries = max_retries if max_retries is not None else self.config.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                self._rate_limiter.wait()
                self._increment_calls()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or ""

            except Exception as exc:
                last_error = exc

                if attempt < retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{retries + 1}): {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Text LLM call failed after {retries + 1} attempts: {last_error}")


def extract_codeblock_or_text(text: str) -> str:
    """Extract content from markdown code block, or return text as-is."""
    match = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON, returning None on failure."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
