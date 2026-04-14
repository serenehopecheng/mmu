"""Data models with Pydantic validation for GEPA-lite."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HardChecks(BaseModel):
    """Hard check results from LLM evaluation."""

    topic_fidelity: bool = False
    narration_adds_context: bool = False
    single_subject_focus: bool = False
    example_alignment: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "topic_fidelity": self.topic_fidelity,
            "narration_adds_context": self.narration_adds_context,
            "single_subject_focus": self.single_subject_focus,
            "example_alignment": self.example_alignment,
        }


class QualityScores(BaseModel):
    """Quality scores from LLM evaluation (0-10 scale)."""

    visual_richness: float = Field(default=0.0, ge=0.0, le=10.0)
    temporal_depth: float = Field(default=0.0, ge=0.0, le=10.0)
    veo_prompt_craft: float = Field(default=0.0, ge=0.0, le=10.0)
    narration_insight: float = Field(default=0.0, ge=0.0, le=10.0)
    topic_precision: float = Field(default=0.0, ge=0.0, le=10.0)

    @field_validator("*", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        try:
            score = float(v)
            return max(0.0, min(10.0, score))
        except (TypeError, ValueError):
            return 0.0

    def mean(self) -> float:
        scores = [
            self.visual_richness,
            self.temporal_depth,
            self.veo_prompt_craft,
            self.narration_insight,
            self.topic_precision,
        ]
        return sum(scores) / len(scores)

    def to_dict(self) -> Dict[str, float]:
        return {
            "visual_richness": self.visual_richness,
            "temporal_depth": self.temporal_depth,
            "veo_prompt_craft": self.veo_prompt_craft,
            "narration_insight": self.narration_insight,
            "topic_precision": self.topic_precision,
        }


class EvalResult(BaseModel):
    """Validated LLM evaluation result."""

    hard_checks: HardChecks = Field(default_factory=HardChecks)
    quality_scores: QualityScores = Field(default_factory=QualityScores)
    feedback: str = ""
    strengths: List[str] = Field(default_factory=list)
    fixes: List[str] = Field(default_factory=list)

    @field_validator("strengths", "fixes", mode="before")
    @classmethod
    def ensure_string_list(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item]

    @field_validator("feedback", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        return str(v) if v else ""

    @classmethod
    def from_llm_response(cls, data: Dict[str, Any]) -> "EvalResult":
        """Parse LLM response into validated EvalResult."""
        hard_checks_data = data.get("hard_checks", {})
        quality_scores_data = data.get("quality_scores", {})

        return cls(
            hard_checks=HardChecks(**hard_checks_data) if hard_checks_data else HardChecks(),
            quality_scores=QualityScores(**quality_scores_data) if quality_scores_data else QualityScores(),
            feedback=data.get("feedback", ""),
            strengths=data.get("strengths", []),
            fixes=data.get("fixes", []),
        )


class GeneratedOutput(BaseModel):
    """Validated generated video script output."""

    script: str = ""
    narration: str = ""

    @classmethod
    def from_raw(cls, raw: str) -> Optional["GeneratedOutput"]:
        """Parse raw LLM output into GeneratedOutput, or return None if invalid."""
        import json

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return cls(
                    script=str(data.get("script", "")),
                    narration=str(data.get("narration", "")),
                )
        except Exception:
            pass
        return None

    def is_valid(self) -> bool:
        return bool(self.script and self.narration)


@dataclass
class TopicEval:
    """Evaluation result for a single topic."""

    task_id: str
    topic: str
    raw_generation: str
    parsed_generation: Dict[str, Any]
    hard_checks: Dict[str, bool]
    quality_scores: Dict[str, float]
    base_quality_score: float
    final_score: float
    feedback: str
    strengths: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "raw_generation": self.raw_generation,
            "parsed_generation": self.parsed_generation,
            "hard_checks": self.hard_checks,
            "quality_scores": self.quality_scores,
            "base_quality_score": self.base_quality_score,
            "final_score": self.final_score,
            "feedback": self.feedback,
            "strengths": self.strengths,
            "fixes": self.fixes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["TopicEval"]:
        try:
            return cls(
                task_id=str(data["task_id"]),
                topic=str(data["topic"]),
                raw_generation=str(data["raw_generation"]),
                parsed_generation=dict(data.get("parsed_generation") or {}),
                hard_checks={str(k): bool(v) for k, v in dict(data.get("hard_checks") or {}).items()},
                quality_scores={str(k): float(v) for k, v in dict(data.get("quality_scores") or {}).items()},
                base_quality_score=float(data.get("base_quality_score", 0.0)),
                final_score=float(data.get("final_score", 0.0)),
                feedback=str(data.get("feedback", "")),
                strengths=[str(x) for x in list(data.get("strengths") or [])],
                fixes=[str(x) for x in list(data.get("fixes") or [])],
            )
        except Exception:
            return None


@dataclass
class Candidate:
    """A prompt candidate in the evolutionary search."""

    cid: int
    prompt: str
    parent_id: Optional[int]
    accepted_from_minibatch: Optional[float] = None
    minibatch_topics: List[str] = field(default_factory=list)
    topic_results: Dict[str, TopicEval] = field(default_factory=dict)
    pareto_avg: float = 0.0

    def get_score_for_task(self, task_id: str) -> float:
        if task_id in self.topic_results:
            return self.topic_results[task_id].final_score
        return 0.0

    def to_history_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.cid,
            "parent_id": self.parent_id,
            "pareto_avg": self.pareto_avg,
            "accepted_from_minibatch": self.accepted_from_minibatch,
            "minibatch_topics": self.minibatch_topics,
            "task_scores": {
                task_id: result.final_score
                for task_id, result in self.topic_results.items()
            },
        }
