"""Tests for data models and Pydantic validation."""

import pytest

from gepa_lite.models import (
    Candidate,
    EvalResult,
    GeneratedOutput,
    HardChecks,
    QualityScores,
    TopicEval,
)


class TestQualityScores:
    """Tests for QualityScores validation."""

    def test_default_values(self) -> None:
        scores = QualityScores()
        assert scores.visual_richness == 0.0
        assert scores.temporal_depth == 0.0
        assert scores.mean() == 0.0

    def test_clamp_high_values(self) -> None:
        scores = QualityScores(visual_richness=15.0, temporal_depth=12.0)
        assert scores.visual_richness == 10.0
        assert scores.temporal_depth == 10.0

    def test_clamp_negative_values(self) -> None:
        scores = QualityScores(visual_richness=-5.0)
        assert scores.visual_richness == 0.0

    def test_invalid_type_defaults_to_zero(self) -> None:
        scores = QualityScores(visual_richness="invalid")
        assert scores.visual_richness == 0.0

    def test_mean_calculation(self) -> None:
        scores = QualityScores(
            visual_richness=10.0,
            temporal_depth=8.0,
            veo_prompt_craft=6.0,
            narration_insight=4.0,
            topic_precision=2.0,
        )
        assert scores.mean() == 6.0

    def test_to_dict(self) -> None:
        scores = QualityScores(visual_richness=7.5)
        d = scores.to_dict()
        assert d["visual_richness"] == 7.5
        assert "temporal_depth" in d


class TestHardChecks:
    """Tests for HardChecks model."""

    def test_default_all_false(self) -> None:
        checks = HardChecks()
        assert checks.topic_fidelity is False
        assert checks.narration_adds_context is False

    def test_to_dict(self) -> None:
        checks = HardChecks(topic_fidelity=True)
        d = checks.to_dict()
        assert d["topic_fidelity"] is True
        assert d["narration_adds_context"] is False


class TestEvalResult:
    """Tests for EvalResult parsing."""

    def test_from_llm_response_minimal(self) -> None:
        data = {}
        result = EvalResult.from_llm_response(data)
        assert result.feedback == ""
        assert result.strengths == []
        assert result.fixes == []

    def test_from_llm_response_full(self) -> None:
        data = {
            "hard_checks": {
                "topic_fidelity": True,
                "narration_adds_context": False,
            },
            "quality_scores": {
                "visual_richness": 8.5,
                "temporal_depth": 7.0,
            },
            "feedback": "Good but could improve narration",
            "strengths": ["Clear visuals", "On topic"],
            "fixes": ["Add context to narration"],
        }
        result = EvalResult.from_llm_response(data)
        assert result.hard_checks.topic_fidelity is True
        assert result.hard_checks.narration_adds_context is False
        assert result.quality_scores.visual_richness == 8.5
        assert result.feedback == "Good but could improve narration"
        assert len(result.strengths) == 2
        assert len(result.fixes) == 1

    def test_ensures_string_list(self) -> None:
        data = {"strengths": [1, 2, None, "valid"]}
        result = EvalResult.from_llm_response(data)
        assert "valid" in result.strengths


class TestGeneratedOutput:
    """Tests for GeneratedOutput parsing."""

    def test_from_raw_valid(self) -> None:
        raw = '{"script": "A dog running", "narration": "Watch the dog"}'
        output = GeneratedOutput.from_raw(raw)
        assert output is not None
        assert output.script == "A dog running"
        assert output.narration == "Watch the dog"
        assert output.is_valid()

    def test_from_raw_invalid_json(self) -> None:
        raw = "not json at all"
        output = GeneratedOutput.from_raw(raw)
        assert output is None

    def test_from_raw_missing_fields(self) -> None:
        raw = '{"script": "test"}'
        output = GeneratedOutput.from_raw(raw)
        assert output is not None
        assert output.script == "test"
        assert output.narration == ""
        assert not output.is_valid()


class TestTopicEval:
    """Tests for TopicEval serialization."""

    def test_to_dict_roundtrip(self) -> None:
        original = TopicEval(
            task_id="test.mp4",
            topic="A test topic",
            raw_generation='{"script": "x", "narration": "y"}',
            parsed_generation={"script": "x", "narration": "y"},
            hard_checks={"valid_json": True},
            quality_scores={"visual_richness": 7.5},
            base_quality_score=7.5,
            final_score=6.5,
            feedback="Good",
            strengths=["Clear"],
            fixes=["Add detail"],
        )

        d = original.to_dict()
        restored = TopicEval.from_dict(d)

        assert restored is not None
        assert restored.task_id == original.task_id
        assert restored.final_score == original.final_score
        assert restored.strengths == original.strengths

    def test_from_dict_invalid(self) -> None:
        result = TopicEval.from_dict({"invalid": "data"})
        assert result is None


class TestCandidate:
    """Tests for Candidate model."""

    def test_get_score_for_task(self) -> None:
        topic_eval = TopicEval(
            task_id="1.mp4",
            topic="Test",
            raw_generation="{}",
            parsed_generation={},
            hard_checks={},
            quality_scores={},
            base_quality_score=5.0,
            final_score=4.5,
            feedback="",
        )

        candidate = Candidate(
            cid=0,
            prompt="test prompt",
            parent_id=None,
            topic_results={"1.mp4": topic_eval},
        )

        assert candidate.get_score_for_task("1.mp4") == 4.5
        assert candidate.get_score_for_task("nonexistent") == 0.0

    def test_to_history_dict(self) -> None:
        candidate = Candidate(
            cid=1,
            prompt="test",
            parent_id=0,
            accepted_from_minibatch=5.0,
            minibatch_topics=["1.mp4"],
            pareto_avg=6.0,
        )

        d = candidate.to_history_dict()
        assert d["candidate_id"] == 1
        assert d["parent_id"] == 0
        assert d["pareto_avg"] == 6.0
