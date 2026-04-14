"""Evaluation logic for GEPA-lite with parallelization support."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from gepa_lite.cache import EvalCache
from gepa_lite.config import Config
from gepa_lite.llm import LLMClient, safe_json_loads
from gepa_lite.models import EvalResult, TopicEval

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles generation and evaluation of video prompts."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        cache: EvalCache,
        gold_standards: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.llm = llm_client
        self.cache = cache
        self.gold_standards = gold_standards or {}

    def get_gold_standard_examples(
        self,
        current_task_id: str,
        limit: Optional[int] = None,
    ) -> str:
        """Get formatted gold standard examples, excluding the current task."""
        limit = limit or self.config.reference_example_limit
        examples: List[Dict[str, str]] = []

        for task_id, gold_prompt in self.gold_standards.items():
            if task_id == current_task_id:
                continue
            topic = self.config.tasks.get(task_id, "Unknown topic")
            examples.append({
                "task_id": task_id,
                "topic": topic,
                "gold_standard_script": gold_prompt,
            })
            if len(examples) >= limit:
                break

        return json.dumps(examples, ensure_ascii=False, indent=2)

    def get_gold_standard_for_task(self, task_id: str) -> Optional[str]:
        """Get the gold standard prompt for a specific task."""
        return self.gold_standards.get(task_id)

    def local_hard_checks(
        self,
        raw_generation: str,
        parsed: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, bool], List[str]]:
        """Run local validation checks on generated output."""
        valid_json = (
            parsed is not None
            and isinstance(parsed.get("script"), str)
            and isinstance(parsed.get("narration"), str)
        )

        if not valid_json:
            return (
                {"valid_json": False},
                ["Output is not valid JSON with string fields script and narration."],
            )

        return {"valid_json": True}, []

    def generate_video_prompt(
        self,
        system_prompt: str,
        topic: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a video script for the given topic."""
        full_prompt = system_prompt.replace("{topic}", topic)
        raw = self.llm.call_text(
            full_prompt,
            model=self.config.generation_model,
        )
        parsed = safe_json_loads(raw) or {}
        return raw, parsed

    def llm_evaluate_output(
        self,
        task_id: str,
        topic: str,
        raw_generation: str,
        parsed_generation: Dict[str, Any],
        local_failures: List[str],
    ) -> EvalResult:
        """Use LLM to evaluate the generated output against the gold standard."""
        gold_standard = self.get_gold_standard_for_task(task_id)
        other_examples = self.get_gold_standard_examples(task_id)

        gold_section = ""
        if gold_standard:
            gold_section = f"""
GOLD STANDARD (the ideal output for this exact topic):
{gold_standard}

The generated output should match this gold standard in:
- Timestamp structure (e.g., [00:00-00:02], [00:02-00:04])
- Shot-by-shot visual descriptions with specific camera angles
- Concrete, cinematic, action-first language
- Level of detail and specificity
"""

        eval_prompt = f"""
You are evaluating a Veo 3 prompt-generation result by comparing it to a gold standard.

Topic:
{topic}

Generated output:
{raw_generation}
{gold_section}
Additional gold standard examples (for reference on style):
{other_examples}

Locally detected issues:
{json.dumps(local_failures, ensure_ascii=False)}

Evaluate the result in two stages.

Stage 1 hard checks:
- topic_fidelity: the script matches the requested topic and does not drift.
- narration_adds_context: narration adds useful context not already obvious from the visuals.
- single_subject_focus: the shot does not introduce unnecessary extra subjects or actions.
- example_alignment: does the output match the gold standard format? It MUST use timestamp segments like [00:00-00:02], include specific camera angles (macro, wide, tracking, close-up), and use concise, action-first cinematic language.

Stage 2 quality scores from 0 to 10 (compare against the gold standard):
- visual_richness: how well does it describe visual details (lighting, textures, colors, angles)?
- temporal_depth: does it break down the scene into clear timestamp segments with distinct actions?
- veo_prompt_craft: how well does it match the gold standard's cinematic prompt style?
- narration_insight: does narration add context beyond obvious visuals?
- topic_precision: does it accurately capture the specific subject/action?

Return strict JSON with this schema:
{{
  "hard_checks": {{
    "topic_fidelity": true,
    "narration_adds_context": true,
    "single_subject_focus": true,
    "example_alignment": true
  }},
  "quality_scores": {{
    "visual_richness": 0,
    "temporal_depth": 0,
    "veo_prompt_craft": 0,
    "narration_insight": 0,
    "topic_precision": 0
  }},
  "feedback": "2 to 4 sentences comparing to the gold standard and suggesting concrete fixes",
  "strengths": ["short bullet", "short bullet"],
  "fixes": ["short bullet", "short bullet"]
}}
""".strip()

        result = self.llm.call_json(
            eval_prompt,
            model=self.config.evaluation_model,
        )
        return EvalResult.from_llm_response(result)

    def evaluate_on_topic(
        self,
        task_id: str,
        system_prompt: str,
        topic: str,
    ) -> TopicEval:
        """Evaluate a prompt on a single topic."""
        cached = self.cache.get(system_prompt, task_id, topic)
        if cached is not None:
            return cached

        raw_generation, parsed_generation = self.generate_video_prompt(system_prompt, topic)
        local_checks, local_failures = self.local_hard_checks(
            raw_generation,
            parsed_generation if parsed_generation else None,
        )
        eval_result = self.llm_evaluate_output(
            task_id,
            topic,
            raw_generation,
            parsed_generation,
            local_failures,
        )

        combined_hard_checks = {
            **local_checks,
            "topic_fidelity": eval_result.hard_checks.topic_fidelity,
            "narration_adds_context": eval_result.hard_checks.narration_adds_context,
            "single_subject_focus": eval_result.hard_checks.single_subject_focus,
            "example_alignment": eval_result.hard_checks.example_alignment,
        }

        normalized_quality = eval_result.quality_scores.to_dict()
        base_quality = eval_result.quality_scores.mean()

        failed_hard_checks = [name for name, passed in combined_hard_checks.items() if not passed]
        penalty = self.config.hard_check_penalty * len(failed_hard_checks)
        final_score = round(max(0.0, min(10.0, base_quality - penalty)), 2)

        feedback_chunks: List[str] = []
        if local_failures:
            feedback_chunks.append("Local hard-check failures: " + " ".join(local_failures))
        if eval_result.feedback:
            feedback_chunks.append(eval_result.feedback.strip())
        if failed_hard_checks:
            feedback_chunks.append("Failed checks: " + ", ".join(failed_hard_checks))

        feedback = "\n".join(chunk for chunk in feedback_chunks if chunk).strip()

        topic_eval = TopicEval(
            task_id=task_id,
            topic=topic,
            raw_generation=raw_generation,
            parsed_generation=parsed_generation,
            hard_checks=combined_hard_checks,
            quality_scores=normalized_quality,
            base_quality_score=round(base_quality, 2),
            final_score=final_score,
            feedback=feedback,
            strengths=eval_result.strengths[:4],
            fixes=eval_result.fixes[:6],
        )

        self.cache.put(system_prompt, task_id, topic, topic_eval)
        return topic_eval

    def evaluate_on_topics(
        self,
        prompt: str,
        topics: List[Tuple[str, str]],
    ) -> Dict[str, TopicEval]:
        """Evaluate a prompt on multiple topics (parallel if enabled)."""
        if not self.config.parallel_evaluation or len(topics) <= 1:
            return {
                task_id: self.evaluate_on_topic(task_id, prompt, topic)
                for task_id, topic in topics
            }

        results: Dict[str, TopicEval] = {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_on_topic, task_id, prompt, topic): task_id
                for task_id, topic in topics
            }

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    results[task_id] = future.result()
                except Exception as exc:
                    logger.error(f"Evaluation failed for task {task_id}: {exc}")
                    raise

        return results

    def evaluate_on_topics_with_cache(
        self,
        prompt: str,
        topics: List[Tuple[str, str]],
        cached_results: Optional[Dict[str, TopicEval]] = None,
    ) -> Dict[str, TopicEval]:
        """Evaluate only missing topics and reuse precomputed per-topic results."""
        results: Dict[str, TopicEval] = dict(cached_results or {})
        missing_topics = [
            (task_id, topic)
            for task_id, topic in topics
            if task_id not in results
        ]

        if missing_topics:
            new_results = self.evaluate_on_topics(prompt, missing_topics)
            results.update(new_results)

        return results


def average_score(results: Dict[str, TopicEval]) -> float:
    """Calculate average final score across all topic results."""
    if not results:
        return 0.0
    return round(mean(item.final_score for item in results.values()), 2)
