"""Tests for disk caching."""

import json
import tempfile
from pathlib import Path

import pytest

from gepa_lite.cache import EvalCache
from gepa_lite.config import Config
from gepa_lite.models import TopicEval


@pytest.fixture
def temp_config() -> Config:
    """Create config with temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Config(
            output_dir=Path(tmpdir),
            enable_disk_cache=True,
            cache_schema_version="test_v1",
        )


@pytest.fixture
def sample_eval() -> TopicEval:
    """Create a sample TopicEval for testing."""
    return TopicEval(
        task_id="test.mp4",
        topic="A test topic",
        raw_generation='{"script": "x", "narration": "y"}',
        parsed_generation={"script": "x", "narration": "y"},
        hard_checks={"valid_json": True},
        quality_scores={"visual_richness": 7.5},
        base_quality_score=7.5,
        final_score=6.5,
        feedback="Good work",
        strengths=["Clear"],
        fixes=["Add detail"],
    )


class TestEvalCache:
    """Tests for EvalCache."""

    def test_miss_when_not_cached(self, temp_config: Config) -> None:
        cache = EvalCache(temp_config)
        result = cache.get("prompt", "task", "topic")
        assert result is None
        assert cache.stats["misses"] == 1

    def test_put_and_get(self, temp_config: Config, sample_eval: TopicEval) -> None:
        cache = EvalCache(temp_config)

        cache.put("prompt", "task", "topic", sample_eval)
        result = cache.get("prompt", "task", "topic")

        assert result is not None
        assert result.task_id == sample_eval.task_id
        assert result.final_score == sample_eval.final_score
        assert cache.stats["hits"] == 1

    def test_different_prompts_different_keys(
        self, temp_config: Config, sample_eval: TopicEval
    ) -> None:
        cache = EvalCache(temp_config)

        cache.put("prompt1", "task", "topic", sample_eval)

        # Different prompt should miss
        result = cache.get("prompt2", "task", "topic")
        assert result is None

    def test_cache_disabled(self, sample_eval: TopicEval) -> None:
        config = Config(enable_disk_cache=False)
        cache = EvalCache(config)

        cache.put("prompt", "task", "topic", sample_eval)
        result = cache.get("prompt", "task", "topic")

        assert result is None

    def test_clear_cache(self, temp_config: Config, sample_eval: TopicEval) -> None:
        cache = EvalCache(temp_config)

        cache.put("prompt1", "task1", "topic1", sample_eval)
        cache.put("prompt2", "task2", "topic2", sample_eval)

        cleared = cache.clear()
        assert cleared == 2

        # Should miss now
        result = cache.get("prompt1", "task1", "topic1")
        assert result is None

    def test_prune_invalid_entries(self, temp_config: Config) -> None:
        cache = EvalCache(temp_config)
        cache_dir = cache.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write valid entry
        valid_data = {
            "task_id": "t1",
            "topic": "topic",
            "raw_generation": "{}",
            "parsed_generation": {},
            "hard_checks": {},
            "quality_scores": {},
            "base_quality_score": 5.0,
            "final_score": 5.0,
            "feedback": "",
        }
        (cache_dir / "valid.json").write_text(json.dumps(valid_data))

        # Write invalid entry (corrupted JSON)
        (cache_dir / "invalid.json").write_text("{not valid json")

        # Write entry missing required fields
        (cache_dir / "incomplete.json").write_text('{"foo": "bar"}')

        pruned = cache.prune_invalid()
        assert pruned == 2

        # Valid entry should still exist
        assert (cache_dir / "valid.json").exists()

    def test_stats_tracking(self, temp_config: Config, sample_eval: TopicEval) -> None:
        cache = EvalCache(temp_config)

        cache.get("p1", "t1", "topic")  # Miss
        cache.get("p2", "t2", "topic")  # Miss
        cache.put("p1", "t1", "topic", sample_eval)
        cache.get("p1", "t1", "topic")  # Hit

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)
