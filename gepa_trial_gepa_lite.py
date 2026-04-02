import json
import logging
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# Configuration
# ============================================================
GENERATION_MODEL = "gpt-4o-mini"
EVALUATION_MODEL = "gpt-4o-mini"
REFLECTION_MODEL = "gpt-4o-mini"

MAX_ITERS = 3
FEEDBACK_MINIBATCH_SIZE = 2
ACCEPT_MARGIN = 0.15
RANDOM_SEED = 42
OUTPUT_DIR = Path("gepa_lite_runs")

TASKS: Dict[str, str] = {
    "3.mp4": "A person correctly using a French Press to brew coffee, countertop view.",
    "4.mp4": "A close-up of a Newton’s Cradle in motion on a desk, macro view.",
    "5.mp4": "A person tying a Bowline knot with a thick rope, hands-only close-up.",
}

SEED_PROMPT = """Write an 8-second video prompt for Veo 3.
Rules:
1. Output valid JSON: {"script": "...", "narration": "..."}.
2. One continuous shot, no cuts.
3. Specify camera angle and lighting.
4. Narration must be under 18 words and provide context not seen on screen.
Topic: {topic}"""

# Optional local heuristics to cheaply enforce obvious constraints.
CAMERA_HINTS = [
    "close-up", "macro", "wide shot", "overhead", "low angle", "high angle",
    "eye-level", "countertop view", "hands-only", "tracking shot", "medium shot"
]
LIGHTING_HINTS = [
    "lighting", "lit", "backlit", "soft light", "daylight", "warm light",
    "cool light", "natural light", "studio light", "diffused light"
]
CUT_HINTS = ["cut to", "cuts to", "scene 1", "scene 2", "montage", "jump cut", "crossfade"]


# ============================================================
# Data classes
# ============================================================
@dataclass
class TopicEval:
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


@dataclass
class Candidate:
    cid: int
    prompt: str
    parent_id: Optional[int]
    accepted_from_minibatch: Optional[float] = None
    minibatch_topics: List[str] = field(default_factory=list)
    topic_results: Dict[str, TopicEval] = field(default_factory=dict)
    pareto_avg: float = 0.0


# ============================================================
# LLM utilities
# ============================================================
def call_llm_json(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    last_error = None
    for _ in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:  # pragma: no cover - runtime guard
            last_error = exc
    raise RuntimeError(f"JSON LLM call failed after retries: {last_error}")



def call_llm_text(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    last_error = None
    for _ in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - runtime guard
            last_error = exc
    raise RuntimeError(f"Text LLM call failed after retries: {last_error}")



def extract_codeblock_or_text(text: str) -> str:
    match = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ============================================================
# Generation + evaluation
# ============================================================
def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None



def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)?\b", text))



def local_hard_checks(raw_generation: str, parsed: Optional[Dict[str, Any]]) -> Tuple[Dict[str, bool], List[str]]:
    failures: List[str] = []
    script = ""
    narration = ""

    valid_json = parsed is not None and isinstance(parsed.get("script"), str) and isinstance(parsed.get("narration"), str)
    if not valid_json:
        checks = {
            "valid_json": False,
            "single_shot": False,
            "camera_angle": False,
            "lighting": False,
            "narration_under_18": False,
        }
        failures.append("Output is not valid JSON with string fields script and narration.")
        return checks, failures

    script = parsed["script"].strip()
    narration = parsed["narration"].strip()
    script_l = script.lower()

    single_shot = not any(hint in script_l for hint in CUT_HINTS)
    camera_angle = any(hint in script_l for hint in CAMERA_HINTS)
    lighting = any(hint in script_l for hint in LIGHTING_HINTS)
    narration_under_18 = word_count(narration) <= 18

    checks = {
        "valid_json": True,
        "single_shot": single_shot,
        "camera_angle": camera_angle,
        "lighting": lighting,
        "narration_under_18": narration_under_18,
    }

    if not single_shot:
        failures.append("Script implies multiple shots or editing instead of one continuous shot.")
    if not camera_angle:
        failures.append("Script does not clearly specify a camera angle or viewpoint.")
    if not lighting:
        failures.append("Script does not clearly specify lighting.")
    if not narration_under_18:
        failures.append("Narration exceeds 18 words.")

    return checks, failures



def generate_video_prompt(system_prompt: str, topic: str) -> Tuple[str, Dict[str, Any]]:
    full_prompt = system_prompt.replace("{topic}", topic)
    raw = call_llm_text(full_prompt, model=GENERATION_MODEL, temperature=0.7)
    parsed = safe_json_loads(raw) or {}
    return raw, parsed



def llm_evaluate_output(topic: str, raw_generation: str, parsed_generation: Dict[str, Any], local_failures: List[str]) -> Dict[str, Any]:
    eval_prompt = f"""
You are evaluating a Veo 3 prompt-generation result.

Topic:
{topic}

Generated output:
{raw_generation}

Locally detected issues:
{json.dumps(local_failures, ensure_ascii=False)}

Evaluate the result in two stages.

Stage 1 hard checks:
- topic_fidelity: the script matches the requested topic and does not drift.
- narration_adds_context: narration adds useful context not already obvious from the visuals.
- single_subject_focus: the shot does not introduce unnecessary extra subjects or actions.

Stage 2 quality scores from 0 to 10:
- visual_richness
- temporal_depth
- veo_prompt_craft
- narration_insight
- topic_precision

Return strict JSON with this schema:
{{
  "hard_checks": {{
    "topic_fidelity": true,
    "narration_adds_context": true,
    "single_subject_focus": true
  }},
  "quality_scores": {{
    "visual_richness": 0,
    "temporal_depth": 0,
    "veo_prompt_craft": 0,
    "narration_insight": 0,
    "topic_precision": 0
  }},
  "feedback": "2 to 4 sentences with concrete fixes",
  "strengths": ["short bullet", "short bullet"],
  "fixes": ["short bullet", "short bullet"]
}}
""".strip()

    result = call_llm_json(eval_prompt, model=EVALUATION_MODEL, temperature=0.0)
    result.setdefault("hard_checks", {})
    result.setdefault("quality_scores", {})
    result.setdefault("feedback", "")
    result.setdefault("strengths", [])
    result.setdefault("fixes", [])
    return result



def evaluate_on_topic(task_id: str, system_prompt: str, topic: str) -> TopicEval:
    raw_generation, parsed_generation = generate_video_prompt(system_prompt, topic)
    local_checks, local_failures = local_hard_checks(raw_generation, parsed_generation if parsed_generation else None)
    llm_result = llm_evaluate_output(topic, raw_generation, parsed_generation, local_failures)

    llm_hard = llm_result.get("hard_checks", {})
    combined_hard_checks = {
        **local_checks,
        "topic_fidelity": bool(llm_hard.get("topic_fidelity", False)),
        "narration_adds_context": bool(llm_hard.get("narration_adds_context", False)),
        "single_subject_focus": bool(llm_hard.get("single_subject_focus", False)),
    }

    quality = llm_result.get("quality_scores", {})
    normalized_quality = {
        "visual_richness": float(quality.get("visual_richness", 0)),
        "temporal_depth": float(quality.get("temporal_depth", 0)),
        "veo_prompt_craft": float(quality.get("veo_prompt_craft", 0)),
        "narration_insight": float(quality.get("narration_insight", 0)),
        "topic_precision": float(quality.get("topic_precision", 0)),
    }

    base_quality = mean(normalized_quality.values())
    failed_hard_checks = [name for name, passed in combined_hard_checks.items() if not passed]
    penalty = 1.2 * len(failed_hard_checks)
    final_score = round(max(0.0, min(10.0, base_quality - penalty)), 2)

    feedback_chunks: List[str] = []
    if local_failures:
        feedback_chunks.append("Local hard-check failures: " + " ".join(local_failures))
    if llm_result.get("feedback"):
        feedback_chunks.append(str(llm_result["feedback"]).strip())
    if failed_hard_checks:
        feedback_chunks.append("Failed checks: " + ", ".join(failed_hard_checks))

    feedback = "\n".join(chunk for chunk in feedback_chunks if chunk).strip()

    return TopicEval(
        task_id=task_id,
        topic=topic,
        raw_generation=raw_generation,
        parsed_generation=parsed_generation,
        hard_checks=combined_hard_checks,
        quality_scores=normalized_quality,
        base_quality_score=round(base_quality, 2),
        final_score=final_score,
        feedback=feedback,
        strengths=[str(x) for x in llm_result.get("strengths", [])][:4],
        fixes=[str(x) for x in llm_result.get("fixes", [])][:6],
    )



def evaluate_candidate_on_topics(prompt: str, topics: List[Tuple[str, str]]) -> Dict[str, TopicEval]:
    return {
        task_id: evaluate_on_topic(task_id, prompt, topic)
        for task_id, topic in topics
    }



def average_score(results: Dict[str, TopicEval]) -> float:
    if not results:
        return 0.0
    return round(mean(item.final_score for item in results.values()), 2)


# ============================================================
# GEPA-lite selection and mutation
# ============================================================
def dominates(a: Candidate, b: Candidate, task_ids: List[str]) -> bool:
    a_scores = [a.topic_results[t].final_score for t in task_ids]
    b_scores = [b.topic_results[t].final_score for t in task_ids]
    return all(x >= y for x, y in zip(a_scores, b_scores)) and any(x > y for x, y in zip(a_scores, b_scores))



def pareto_frontier(candidates: List[Candidate], task_ids: List[str]) -> List[Candidate]:
    # Step 1: task-wise winners.
    winners: List[Candidate] = []
    for task_id in task_ids:
        best_score = max(c.topic_results[task_id].final_score for c in candidates)
        winners.extend([c for c in candidates if c.topic_results[task_id].final_score == best_score])

    unique_winners = list({c.cid: c for c in winners}.values())

    # Step 2: prune dominated winners.
    frontier: List[Candidate] = []
    for cand in unique_winners:
        if not any(other.cid != cand.cid and dominates(other, cand, task_ids) for other in unique_winners):
            frontier.append(cand)
    return frontier



def sample_parent_from_frontier(candidates: List[Candidate], task_ids: List[str]) -> Candidate:
    frontier = pareto_frontier(candidates, task_ids)
    weights = []
    for cand in frontier:
        wins = 0
        for task_id in task_ids:
            best_score = max(c.topic_results[task_id].final_score for c in frontier)
            if cand.topic_results[task_id].final_score == best_score:
                wins += 1
        weights.append(max(1, wins))
    return random.choices(frontier, weights=weights, k=1)[0]



def summarize_minibatch_feedback(results: Dict[str, TopicEval]) -> str:
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



def reflective_mutate(parent_prompt: str, minibatch_results: Dict[str, TopicEval]) -> str:
    feedback_blob = summarize_minibatch_feedback(minibatch_results)

    mutation_prompt = f"""
I provided an assistant with the following instructions to perform a task for me:
```
{parent_prompt}
```

The following are examples of task inputs, assistant responses, hard-check results, rubric scores,
and feedback about how the response could be better:
```
{feedback_blob}
```

Write a new improved instruction for the assistant.

Requirements:
- Preserve the exact topic placeholder: {{topic}}
- The output must still require valid JSON with keys "script" and "narration"
- Keep the task focused on a single 8-second Veo 3 shot
- Make the instruction more robust against the failures shown above
- Add niche, domain-specific guidance only when it clearly generalizes across examples
- Prefer concise, durable rules over verbose filler
- Do not include analysis before or after the instruction
- Return the full improved instruction inside triple backticks
""".strip()

    rewritten = call_llm_text(mutation_prompt, model=REFLECTION_MODEL, temperature=0.2)
    candidate_prompt = extract_codeblock_or_text(rewritten)
    if "{topic}" not in candidate_prompt:
        # Safety fallback: do not lose the placeholder.
        candidate_prompt = parent_prompt
    return candidate_prompt


# ============================================================
# Main search loop
# ============================================================
def choose_feedback_topics(task_items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if FEEDBACK_MINIBATCH_SIZE >= len(task_items):
        return task_items[:]
    return random.sample(task_items, FEEDBACK_MINIBATCH_SIZE)



def write_run_artifacts(candidates: List[Candidate], best_candidate: Candidate) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "best_prompt_.txt").write_text(best_candidate.prompt, encoding="utf-8")

    history = []
    for c in candidates:
        history.append({
            "candidate_id": c.cid,
            "parent_id": c.parent_id,
            "pareto_avg": c.pareto_avg,
            "accepted_from_minibatch": c.accepted_from_minibatch,
            "minibatch_topics": c.minibatch_topics,
            "task_scores": {task_id: result.final_score for task_id, result in c.topic_results.items()},
        })

    (OUTPUT_DIR / "search_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    detailed = {
        c.cid: {
            task_id: {
                "topic": res.topic,
                "final_score": res.final_score,
                "base_quality_score": res.base_quality_score,
                "hard_checks": res.hard_checks,
                "quality_scores": res.quality_scores,
                "feedback": res.feedback,
                "strengths": res.strengths,
                "fixes": res.fixes,
                "generated_output": res.parsed_generation or res.raw_generation,
            }
            for task_id, res in c.topic_results.items()
        }
        for c in candidates
    }
    (OUTPUT_DIR / "detailed_evaluations.json").write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )



def run_gepa_lite() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "run_trace.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    task_items = list(TASKS.items())
    task_ids = [task_id for task_id, _ in task_items]

    logging.info(f"Running GEPA-lite for {MAX_ITERS} iterations on {len(task_items)} tasks...")

    # Seed candidate evaluated on full Pareto set.
    seed_results = evaluate_candidate_on_topics(SEED_PROMPT, task_items)
    seed_candidate = Candidate(
        cid=0,
        prompt=SEED_PROMPT,
        parent_id=None,
        topic_results=seed_results,
        pareto_avg=average_score(seed_results),
    )

    candidates: List[Candidate] = [seed_candidate]
    best_candidate = seed_candidate
    next_id = 1

    logging.info(f"Seed candidate average score: {seed_candidate.pareto_avg:.2f}")

    for step in range(1, MAX_ITERS + 1):
        parent = sample_parent_from_frontier(candidates, task_ids)
        minibatch = choose_feedback_topics(task_items)
        minibatch_ids = [task_id for task_id, _ in minibatch]

        logging.info("-" * 60)
        logging.info(f"Iteration {step}")
        logging.info(f"Selected parent: C{parent.cid} | full avg = {parent.pareto_avg:.2f} | minibatch = {minibatch_ids}")

        # Reuse parent scores on the minibatch from the cached full evaluation.
        parent_minibatch_results = {task_id: parent.topic_results[task_id] for task_id in minibatch_ids}
        parent_minibatch_avg = average_score(parent_minibatch_results)

        child_prompt = reflective_mutate(parent.prompt, parent_minibatch_results)
        child_minibatch_results = evaluate_candidate_on_topics(child_prompt, minibatch)
        child_minibatch_avg = average_score(child_minibatch_results)

        logging.info(f"Parent minibatch avg: {parent_minibatch_avg:.2f}")
        logging.info(f"Child  minibatch avg: {child_minibatch_avg:.2f}")

        if child_minibatch_avg <= parent_minibatch_avg + ACCEPT_MARGIN:
            logging.info("Rejected child on minibatch gate.")
            continue

        # Accepted on minibatch. Evaluate on full Pareto set before adding.
        full_child_results = evaluate_candidate_on_topics(child_prompt, task_items)
        full_child_avg = average_score(full_child_results)

        child = Candidate(
            cid=next_id,
            prompt=child_prompt,
            parent_id=parent.cid,
            accepted_from_minibatch=child_minibatch_avg,
            minibatch_topics=minibatch_ids,
            topic_results=full_child_results,
            pareto_avg=full_child_avg,
        )
        next_id += 1
        candidates.append(child)

        logging.info(f"Accepted child C{child.cid} | full avg = {child.pareto_avg:.2f}")

        if child.pareto_avg > best_candidate.pareto_avg:
            best_candidate = child
            logging.info(f"New best candidate: C{child.cid}")

    logging.info("=" * 60)
    logging.info(f"Search complete. Best candidate: C{best_candidate.cid}")
    logging.info(f"Best average score: {best_candidate.pareto_avg:.2f}")

    write_run_artifacts(candidates, best_candidate)
    logging.info(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        run_gepa_lite()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
