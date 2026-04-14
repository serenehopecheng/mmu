"""Tests for evaluation logic."""

import pytest

from gepa_lite.evaluation import average_score
from gepa_lite.models import TopicEval


def make_topic_eval(task_id: str, score: float) -> TopicEval:
    """Helper to create a TopicEval with a specific score."""
    return TopicEval(
        task_id=task_id,
        topic=f"Topic for {task_id}",
        raw_generation="{}",
        parsed_generation={},
        hard_checks={},
        quality_scores={},
        base_quality_score=score,
        final_score=score,
        feedback="",
    )


class TestAverageScore:
    """Tests for average_score calculation."""

    def test_empty_results(self) -> None:
        assert average_score({}) == 0.0

    def test_single_result(self) -> None:
        results = {"t1": make_topic_eval("t1", 7.5)}
        assert average_score(results) == 7.5

    def test_multiple_results(self) -> None:
        results = {
            "t1": make_topic_eval("t1", 6.0),
            "t2": make_topic_eval("t2", 8.0),
            "t3": make_topic_eval("t3", 4.0),
        }
        # Average of 6.0, 8.0, 4.0 = 6.0
        assert average_score(results) == 6.0

    def test_rounding(self) -> None:
        results = {
            "t1": make_topic_eval("t1", 7.333),
            "t2": make_topic_eval("t2", 7.333),
            "t3": make_topic_eval("t3", 7.334),
        }
        # Should round to 2 decimal places
        avg = average_score(results)
        assert avg == pytest.approx(7.33, abs=0.01)
