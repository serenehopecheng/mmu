"""Tests for configuration management."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from gepa_lite.config import Config, DEFAULT_TASKS


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self) -> None:
        config = Config()
        assert config.generation_model == "gpt-5-nano"
        assert config.max_iters == 12
        assert config.feedback_minibatch_size == 2
        assert config.accept_margin == 0.15
        assert config.random_seed == 42

    def test_default_tasks_loaded(self) -> None:
        config = Config()
        assert len(config.tasks) == len(DEFAULT_TASKS)
        assert "0.mp4" in config.tasks

    def test_custom_tasks(self) -> None:
        custom_tasks = {"a.mp4": "Task A", "b.mp4": "Task B"}
        config = Config(tasks=custom_tasks)
        assert config.tasks == custom_tasks

    def test_path_conversion(self) -> None:
        config = Config(
            output_dir="custom_output",
            reference_examples_path="custom_prompts.json",
        )
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.reference_examples_path, Path)

    def test_to_dict(self) -> None:
        config = Config()
        d = config.to_dict()

        assert d["generation_model"] == "gpt-5-nano"
        assert d["max_iters"] == 12
        assert isinstance(d["output_dir"], str)

    def test_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "generation_model": "custom-model",
                    "max_iters": 20,
                    "accept_margin": 0.25,
                },
                f,
            )
            f.flush()

            config = Config.from_file(Path(f.name))

            assert config.generation_model == "custom-model"
            assert config.max_iters == 20
            assert config.accept_margin == 0.25

            os.unlink(f.name)

    def test_from_env(self) -> None:
        env_vars = {
            "GEPA_GENERATION_MODEL": "env-model",
            "GEPA_MAX_ITERS": "30",
            "GEPA_ACCEPT_MARGIN": "0.5",
            "GEPA_LOG_LEVEL": "DEBUG",
        }

        with mock.patch.dict(os.environ, env_vars):
            config = Config.from_env()

            assert config.generation_model == "env-model"
            assert config.max_iters == 30
            assert config.accept_margin == 0.5
            assert config.log_level == "DEBUG"

    def test_from_env_partial_override(self) -> None:
        # Only override one variable
        with mock.patch.dict(os.environ, {"GEPA_MAX_ITERS": "50"}, clear=False):
            config = Config.from_env()

            assert config.max_iters == 50
            # Others should be defaults
            assert config.generation_model == "gpt-5-nano"


class TestDefaultTasks:
    """Tests for default task definitions."""

    def test_all_tasks_have_descriptions(self) -> None:
        for task_id, description in DEFAULT_TASKS.items():
            assert task_id.endswith(".mp4"), f"{task_id} should end with .mp4"
            assert len(description) > 10, f"{task_id} description too short"

    def test_task_count(self) -> None:
        # Should have a reasonable number of tasks
        assert len(DEFAULT_TASKS) >= 20
