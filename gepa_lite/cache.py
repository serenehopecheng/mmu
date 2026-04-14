"""Disk caching for GEPA-lite evaluations."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from gepa_lite.config import Config
from gepa_lite.models import TopicEval

logger = logging.getLogger(__name__)


class EvalCache:
    """Disk-based cache for topic evaluations."""

    def __init__(self, config: Config):
        self.config = config
        self._hits = 0
        self._misses = 0

    @property
    def cache_dir(self) -> Path:
        return self.config.output_dir / self.config.cache_dirname

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def _make_cache_key(
        self,
        system_prompt: str,
        task_id: str,
        topic: str,
    ) -> str:
        """Generate a unique cache key for the evaluation parameters."""
        payload = {
            "schema_version": self.config.cache_schema_version,
            "task_id": task_id,
            "topic": topic,
            "system_prompt": system_prompt,
            "generation_model": self.config.generation_model,
            "evaluation_model": self.config.evaluation_model,
            "reference_example_limit": self.config.reference_example_limit,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        system_prompt: str,
        task_id: str,
        topic: str,
    ) -> Optional[TopicEval]:
        """Retrieve cached evaluation, or None if not found."""
        if not self.config.enable_disk_cache:
            return None

        cache_key = self._make_cache_key(system_prompt, task_id, topic)
        cache_path = self._cache_path(cache_key)

        if not cache_path.exists():
            self._misses += 1
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            result = TopicEval.from_dict(payload)
            if result is not None:
                self._hits += 1
                logger.debug(f"Cache hit for task_id={task_id}")
                return result
            else:
                logger.warning(f"Cache corruption detected for {cache_path}")
                self._misses += 1
                return None
        except json.JSONDecodeError as exc:
            logger.warning(f"Cache JSON decode error for {cache_path}: {exc}")
            self._misses += 1
            return None
        except Exception as exc:
            logger.warning(f"Cache read error for {cache_path}: {exc}")
            self._misses += 1
            return None

    def put(
        self,
        system_prompt: str,
        task_id: str,
        topic: str,
        result: TopicEval,
    ) -> None:
        """Store evaluation result in cache."""
        if not self.config.enable_disk_cache:
            return

        cache_key = self._make_cache_key(system_prompt, task_id, topic)
        cache_path = self._cache_path(cache_key)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"Cached evaluation for task_id={task_id}")
        except Exception as exc:
            logger.warning(f"Failed to write cache for {cache_path}: {exc}")

    def clear(self) -> int:
        """Clear all cached evaluations. Returns number of files removed."""
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as exc:
                logger.warning(f"Failed to delete cache file {cache_file}: {exc}")

        self._hits = 0
        self._misses = 0
        logger.info(f"Cleared {count} cached evaluations")
        return count

    def prune_invalid(self) -> int:
        """Remove corrupted cache entries. Returns number of files removed."""
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                if TopicEval.from_dict(payload) is None:
                    cache_file.unlink()
                    count += 1
                    logger.debug(f"Pruned invalid cache entry: {cache_file}")
            except Exception:
                cache_file.unlink()
                count += 1
                logger.debug(f"Pruned corrupted cache entry: {cache_file}")

        if count > 0:
            logger.info(f"Pruned {count} invalid cache entries")
        return count
