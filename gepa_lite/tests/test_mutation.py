"""Tests for Pareto selection and mutation logic."""

import pytest

from gepa_lite.config import Config
from gepa_lite.models import Candidate, TopicEval
from gepa_lite.mutation import (
    compute_prompt_similarity,
    dominates,
    EarlyStopTracker,
    Mutator,
    pareto_frontier,
    sample_parent_from_frontier,
)


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


def make_candidate(cid: int, scores: dict) -> Candidate:
    """Helper to create a Candidate with specific task scores."""
    return Candidate(
        cid=cid,
        prompt=f"Prompt {cid}",
        parent_id=None,
        topic_results={
            task_id: make_topic_eval(task_id, score)
            for task_id, score in scores.items()
        },
    )


class TestDominates:
    """Tests for Pareto dominance checking."""

    def test_dominates_all_better(self) -> None:
        a = make_candidate(0, {"t1": 8.0, "t2": 9.0})
        b = make_candidate(1, {"t1": 5.0, "t2": 6.0})
        task_ids = ["t1", "t2"]

        assert dominates(a, b, task_ids) is True
        assert dominates(b, a, task_ids) is False

    def test_dominates_one_better_one_equal(self) -> None:
        a = make_candidate(0, {"t1": 8.0, "t2": 6.0})
        b = make_candidate(1, {"t1": 5.0, "t2": 6.0})
        task_ids = ["t1", "t2"]

        assert dominates(a, b, task_ids) is True

    def test_no_dominance_tradeoff(self) -> None:
        a = make_candidate(0, {"t1": 8.0, "t2": 4.0})
        b = make_candidate(1, {"t1": 4.0, "t2": 8.0})
        task_ids = ["t1", "t2"]

        assert dominates(a, b, task_ids) is False
        assert dominates(b, a, task_ids) is False

    def test_no_dominance_equal(self) -> None:
        a = make_candidate(0, {"t1": 5.0, "t2": 5.0})
        b = make_candidate(1, {"t1": 5.0, "t2": 5.0})
        task_ids = ["t1", "t2"]

        assert dominates(a, b, task_ids) is False
        assert dominates(b, a, task_ids) is False


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_single_candidate(self) -> None:
        c = make_candidate(0, {"t1": 5.0})
        frontier = pareto_frontier([c], ["t1"])
        assert len(frontier) == 1
        assert frontier[0].cid == 0

    def test_one_dominates_other(self) -> None:
        a = make_candidate(0, {"t1": 8.0, "t2": 8.0})
        b = make_candidate(1, {"t1": 5.0, "t2": 5.0})
        frontier = pareto_frontier([a, b], ["t1", "t2"])
        assert len(frontier) == 1
        assert frontier[0].cid == 0

    def test_pareto_tradeoff(self) -> None:
        a = make_candidate(0, {"t1": 10.0, "t2": 2.0})
        b = make_candidate(1, {"t1": 2.0, "t2": 10.0})
        c = make_candidate(2, {"t1": 5.0, "t2": 5.0})  # Dominated by neither
        frontier = pareto_frontier([a, b, c], ["t1", "t2"])

        # a and b should be on frontier (best at t1 and t2 respectively)
        frontier_ids = {c.cid for c in frontier}
        assert 0 in frontier_ids
        assert 1 in frontier_ids

    def test_empty_candidates(self) -> None:
        frontier = pareto_frontier([], ["t1"])
        assert frontier == []


class TestSampleParentFromFrontier:
    """Tests for weighted sampling from Pareto frontier."""

    def test_single_candidate(self) -> None:
        c = make_candidate(0, {"t1": 5.0})
        parent = sample_parent_from_frontier([c], ["t1"])
        assert parent.cid == 0

    def test_returns_from_frontier(self) -> None:
        a = make_candidate(0, {"t1": 10.0, "t2": 10.0})
        b = make_candidate(1, {"t1": 1.0, "t2": 1.0})  # Dominated

        # Should always return a (the only frontier member)
        for _ in range(10):
            parent = sample_parent_from_frontier([a, b], ["t1", "t2"])
            assert parent.cid == 0


class TestComputePromptSimilarity:
    """Tests for prompt similarity calculation."""

    def test_identical_prompts(self) -> None:
        similarity = compute_prompt_similarity("hello world", "hello world")
        assert similarity == 1.0

    def test_completely_different(self) -> None:
        similarity = compute_prompt_similarity("abc", "xyz")
        assert similarity < 0.5

    def test_partial_overlap(self) -> None:
        similarity = compute_prompt_similarity(
            "Generate a video about {topic}",
            "Generate a script about {topic}",
        )
        assert 0.5 < similarity < 1.0


class TestEarlyStopTracker:
    """Tests for early stopping logic."""

    def test_initial_state(self) -> None:
        config = Config(early_stop_enabled=True, early_stop_patience=3)
        tracker = EarlyStopTracker(config)
        assert tracker.should_stop() is False
        assert tracker.iterations_without_improvement == 0

    def test_improvement_resets_counter(self) -> None:
        config = Config(early_stop_enabled=True, early_stop_patience=3)
        tracker = EarlyStopTracker(config)

        tracker.update(5.0)
        tracker.update(4.0)  # No improvement
        tracker.update(4.0)  # No improvement
        assert tracker.iterations_without_improvement == 2

        tracker.update(6.0)  # Improvement!
        assert tracker.iterations_without_improvement == 0

    def test_stops_after_patience(self) -> None:
        config = Config(early_stop_enabled=True, early_stop_patience=2)
        tracker = EarlyStopTracker(config)

        tracker.update(5.0)
        assert tracker.should_stop() is False

        tracker.update(4.0)  # No improvement
        assert tracker.should_stop() is False

        tracker.update(4.0)  # No improvement (patience=2 reached)
        assert tracker.should_stop() is True

    def test_disabled_never_stops(self) -> None:
        config = Config(early_stop_enabled=False, early_stop_patience=1)
        tracker = EarlyStopTracker(config)

        tracker.update(5.0)
        tracker.update(4.0)
        tracker.update(3.0)
        tracker.update(2.0)

        assert tracker.should_stop() is False


class TestMutatorDiversity:
    """Tests for Mutator diversity checking."""

    def test_diverse_prompt_accepted(self) -> None:
        config = Config(diversity_enabled=True, diversity_threshold=0.3)
        mutator = Mutator(config, llm_client=None)

        existing = ["Generate a video about dogs"]
        new_prompt = "Create a script showing cats playing"

        assert mutator.is_sufficiently_diverse(new_prompt, existing) is True

    def test_similar_prompt_rejected(self) -> None:
        config = Config(diversity_enabled=True, diversity_threshold=0.3)
        mutator = Mutator(config, llm_client=None)

        existing = ["Generate a video about {topic}"]
        new_prompt = "Generate a video about {topic}"  # Identical

        assert mutator.is_sufficiently_diverse(new_prompt, existing) is False

    def test_diversity_disabled_accepts_all(self) -> None:
        config = Config(diversity_enabled=False)
        mutator = Mutator(config, llm_client=None)

        existing = ["same prompt"]
        new_prompt = "same prompt"

        assert mutator.is_sufficiently_diverse(new_prompt, existing) is True
