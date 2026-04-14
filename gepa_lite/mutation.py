"""Pareto selection and reflective mutation for GEPA-lite."""

import json
import random
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from gepa_lite.config import Config
from gepa_lite.llm import LLMClient, extract_codeblock_or_text
from gepa_lite.models import Candidate, TopicEval


def dominates(a: Candidate, b: Candidate, task_ids: List[str]) -> bool:
    """Check if candidate a Pareto-dominates candidate b."""
    a_scores = [a.get_score_for_task(t) for t in task_ids]
    b_scores = [b.get_score_for_task(t) for t in task_ids]
    return (
        all(x >= y for x, y in zip(a_scores, b_scores))
        and any(x > y for x, y in zip(a_scores, b_scores))
    )


def pareto_frontier(
    candidates: List[Candidate],
    task_ids: List[str],
) -> List[Candidate]:
    """Compute the Pareto frontier from a list of candidates."""
    if not candidates:
        return []

    # Step 1: task-wise winners
    winners: List[Candidate] = []
    for task_id in task_ids:
        best_score = max(c.get_score_for_task(task_id) for c in candidates)
        winners.extend([
            c for c in candidates
            if c.get_score_for_task(task_id) == best_score
        ])

    unique_winners = list({c.cid: c for c in winners}.values())

    # Step 2: prune dominated winners
    frontier: List[Candidate] = []
    for cand in unique_winners:
        if not any(
            other.cid != cand.cid and dominates(other, cand, task_ids)
            for other in unique_winners
        ):
            frontier.append(cand)

    return frontier


def sample_parent_from_frontier(
    candidates: List[Candidate],
    task_ids: List[str],
) -> Candidate:
    """Sample a parent candidate from the Pareto frontier."""
    frontier = pareto_frontier(candidates, task_ids)
    if not frontier:
        return random.choice(candidates)

    weights = []
    for cand in frontier:
        wins = 0
        for task_id in task_ids:
            best_score = max(c.get_score_for_task(task_id) for c in frontier)
            if cand.get_score_for_task(task_id) == best_score:
                wins += 1
        weights.append(max(1, wins))

    return random.choices(frontier, weights=weights, k=1)[0]


def summarize_minibatch_feedback(results: Dict[str, TopicEval]) -> str:
    """Create a feedback summary from minibatch evaluation results."""
    blocks = []
    for item in results.values():
        block = {
            "task_id": item.task_id,
            "topic": item.topic,
            "score": item.final_score,
            "base_quality_score": item.base_quality_score,
            "hard_checks": item.hard_checks,
            "quality_scores": item.quality_scores,
            "generated_output": item.parsed_generation or item.raw_generation,
            "feedback": item.feedback,
            "strengths": item.strengths,
            "fixes": item.fixes,
        }
        blocks.append(json.dumps(block, ensure_ascii=False, indent=2))
    return "\n\n".join(blocks)


def compute_prompt_similarity(prompt_a: str, prompt_b: str) -> float:
    """Compute similarity ratio between two prompts (0.0 to 1.0)."""
    return SequenceMatcher(None, prompt_a, prompt_b).ratio()


class Mutator:
    """Handles reflective mutation of prompts."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        reference_examples: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.llm = llm_client
        self.reference_examples = reference_examples or config.tasks

    def is_sufficiently_diverse(
        self,
        new_prompt: str,
        existing_prompts: List[str],
    ) -> bool:
        """Check if new prompt is sufficiently different from existing ones."""
        if not self.config.diversity_enabled:
            return True

        for existing in existing_prompts:
            similarity = compute_prompt_similarity(new_prompt, existing)
            if similarity > (1.0 - self.config.diversity_threshold):
                return False
        return True

    def reflective_mutate(
        self,
        parent_prompt: str,
        minibatch_results: Dict[str, TopicEval],
    ) -> str:
        """Generate a mutated prompt based on evaluation feedback."""
        feedback_blob = summarize_minibatch_feedback(minibatch_results)
        minibatch_topics = [item.topic for item in minibatch_results.values()]
        reference_examples = json.dumps(
            [
                {
                    "task_id": task_id,
                    "topic": self.reference_examples.get(task_id, topic),
                }
                for task_id, topic in list(self.reference_examples.items())[:self.config.reference_example_limit]
            ],
            ensure_ascii=False,
            indent=2,
        )

        mutation_prompt = f"""
I provided an assistant with the following instructions to perform a task for me:
```
{parent_prompt}
```

These reference examples show the target style the prompt should generalize to:
```
{reference_examples}
```

The following are minibatch examples of task inputs, assistant responses, hard-check results, rubric scores,
and feedback about how the response could be better:
```
{feedback_blob}
```

When you rewrite the instruction, make the critique actually follow the reference examples above:
for each recurring failure, infer the missing rule from the examples and encode that rule directly into the new instruction.
Do not just restate evaluator feedback in generic terms.

Current minibatch topics:
{json.dumps(minibatch_topics, ensure_ascii=False)}

Write a new improved instruction for the assistant.

Requirements:
- Preserve the exact topic placeholder: {{topic}}
- The output must still require valid JSON with keys "script" and "narration"
- Keep the task focused on a single 8-second Veo 3 shot
- Make the instruction more robust against the failures shown above
- Explicitly prefer the same level of specificity and concrete phrasing demonstrated by the reference examples
- Add niche, domain-specific guidance only when it clearly generalizes across examples
- Prefer concise, durable rules over verbose filler
- Do not include analysis before or after the instruction
- Return the full improved instruction inside triple backticks
""".strip()

        rewritten = self.llm.call_text(
            mutation_prompt,
            model=self.config.reflection_model,
        )
        candidate_prompt = extract_codeblock_or_text(rewritten)

        if "{topic}" not in candidate_prompt:
            # Safety fallback: do not lose the placeholder
            candidate_prompt = parent_prompt

        return candidate_prompt


class EarlyStopTracker:
    """Tracks improvement to enable early stopping."""

    def __init__(self, config: Config):
        self.config = config
        self._best_score = float("-inf")
        self._iterations_without_improvement = 0

    def update(self, score: float) -> None:
        """Update tracker with new score."""
        if score > self._best_score:
            self._best_score = score
            self._iterations_without_improvement = 0
        else:
            self._iterations_without_improvement += 1

    def should_stop(self) -> bool:
        """Check if early stopping criteria is met."""
        if not self.config.early_stop_enabled:
            return False
        return self._iterations_without_improvement >= self.config.early_stop_patience

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def iterations_without_improvement(self) -> int:
        return self._iterations_without_improvement
