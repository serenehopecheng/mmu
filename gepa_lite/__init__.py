"""GEPA-lite: Genetic Evolution of Prompts with Acceptance."""

from gepa_lite.config import Config
from gepa_lite.models import TopicEval, Candidate, EvalResult
from gepa_lite.runner import run_gepa_lite

__version__ = "0.2.0"
__all__ = ["Config", "TopicEval", "Candidate", "EvalResult", "run_gepa_lite"]
