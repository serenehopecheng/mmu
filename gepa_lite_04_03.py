import json
import os
import random
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
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
GENERATION_MODEL = "gpt-4o"
EVALUATION_MODEL = "gpt-4o"
REFLECTION_MODEL = "gpt-4o"

MAX_ITERS = 12
FEEDBACK_MINIBATCH_SIZE = 2
ACCEPT_MARGIN = 0.15
RANDOM_SEED = 42
OUTPUT_DIR = Path("gepa_lite_runs")
TRACE_FILENAME = "gepa_lite_run_trace.txt"

TASKS: Dict[str, str] = {
  "0.mp4": "A person unboxing the latest iPhone 16 Pro in Desert Titanium on a wooden table, front view.",
  "1.mp4": "A close-up of a Tesla Cybertruck driving through a puddle, low angle.",
  "2.mp4": "A person opening a bottle of Coca-Cola with the new attached tethered cap, side view.",
  "3.mp4": "A person correctly using a French Press to brew coffee, countertop view.",
  "4.mp4": "A close-up of a Newton's Cradle in motion on a desk, macro view.",
 
  "5.mp4": "A person tying a Bowline knot with a thick rope, hands-only close-up.",
  "7.mp4": "A yellow New York City taxi driving through Times Square, street-level view.",
  "8.mp4": "A person walking across the Abbey Road zebra crossing, eye-level view.",
  "9.mp4": "A Mongolian horseman using an Uurga to catch a wild horse, wide landscape shot.",
 
  "10.mp4": "A samurai practicing with a nodachi.",
  "11.mp4": "The lifecycle of a Monarch butterfly: the chrysalis stage.",
  "12.mp4": "A chef preparing Hand-Pulled Lanzhou Ramen (Lamian).",
  "13.mp4": "A person playing a game of 'Jenga' and pulling out a middle block.",
  "14.mp4": "A chemist performing a 'Titration' until the exact 'End Point' is reached.",
 
  "15.mp4": "A person using a 'Self-Checkout' machine at a grocery store.",
  "16.mp4": "A close-up of a 'Vinyl Record Player' being started.",
  "17.mp4": "A person changing a flat tire on a car.",
  "18.mp4": "A dancer performing the 'Haka' with correct 'Pukana' facial expressions.",
  "19.mp4": "A person wearing a 'Hennin' headdress walking through a 15th-century court.",
 
  "20.mp4": "A close-up of a 'Mantis Shrimp' punching a glass tank.",
  "21.mp4": "A person making a peanut butter and jelly sandwich.",
  "22.mp4": "A person mailing a letter at a blue USPS mailbox.",
  "23.mp4": "A technician performing a 'Three-Point Bend' test on a carbon fiber sample.",
  "24.mp4": "A barista preparing a traditional Turkish coffee using a cezve in hot sand.",
 
  "25.mp4": "An archer demonstrating the Khatra technique upon releasing an arrow, slow-motion side view.",
  "26.mp4": "An artisan practicing Kintsugi on a broken ceramic bowl, soft-lit tabletop scene."
 }


SEED_PROMPT = """Given the topic, background_info, generate a video script. Output valid JSON: {"script": "...", "narration": "..."}
Topic: {topic}
Background info: {background_info}"""


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



def local_hard_checks(_raw_generation: str, parsed: Optional[Dict[str, Any]]) -> Tuple[Dict[str, bool], List[str]]:
    valid_json = parsed is not None and isinstance(parsed.get("script"), str) and isinstance(parsed.get("narration"), str)
    if not valid_json:
        return {"valid_json": False}, ["Output is not valid JSON with string fields script and narration."]
    return {"valid_json": True}, []



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

    (OUTPUT_DIR / "best_prompt.txt").write_text(best_candidate.prompt, encoding="utf-8")

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



def _trace_banner(title: str) -> str:
    line = "=" * 72
    return f"\n{line}\n{title}\n{line}\n"


def format_topic_eval_block(te: TopicEval) -> str:
    gen = te.parsed_generation if te.parsed_generation else te.raw_generation
    if isinstance(gen, dict):
        gen_text = json.dumps(gen, ensure_ascii=False, indent=2)
    else:
        gen_text = str(gen)
    parts = [
        f"task_id: {te.task_id}",
        f"final_score: {te.final_score}  (base_quality: {te.base_quality_score})",
        f"hard_checks: {json.dumps(te.hard_checks, ensure_ascii=False)}",
        f"quality_scores: {json.dumps(te.quality_scores, ensure_ascii=False)}",
        "generated (script/narration or raw):",
        textwrap.indent(gen_text.strip(), "  "),
        "feedback:",
        textwrap.indent(te.feedback.strip() or "(none)", "  "),
    ]
    if te.strengths:
        parts.append("strengths: " + "; ".join(te.strengths))
    if te.fixes:
        parts.append("fixes: " + "; ".join(te.fixes))
    return "\n".join(parts)


def format_results_block(results: Dict[str, TopicEval]) -> str:
    blocks = [format_topic_eval_block(te) for te in results.values()]
    return "\n\n---\n\n".join(blocks)


def run_gepa_lite() -> None:
    random.seed(RANDOM_SEED)
    task_items = list(TASKS.items())
    task_ids = [task_id for task_id, _ in task_items]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = OUTPUT_DIR / TRACE_FILENAME

    print(f"Running GEPA-lite for {MAX_ITERS} iterations on {len(task_items)} tasks...")
    print(f"Trace log: {trace_path.resolve()}")

    with trace_path.open("w", encoding="utf-8") as tf:

        def log_trace(text: str) -> None:
            tf.write(text)
            tf.flush()

        log_trace(
            _trace_banner("GEPA-lite run")
            + f"started_at: {datetime.now().isoformat(timespec='seconds')}\n"
            + f"MAX_ITERS={MAX_ITERS}  FEEDBACK_MINIBATCH_SIZE={FEEDBACK_MINIBATCH_SIZE}  "
            + f"ACCEPT_MARGIN={ACCEPT_MARGIN}  RANDOM_SEED={RANDOM_SEED}\n"
            + f"tasks: {len(task_items)}\n"
            + f"models: gen={GENERATION_MODEL} eval={EVALUATION_MODEL} reflect={REFLECTION_MODEL}\n"
        )

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

        print(f"Seed candidate average score: {seed_candidate.pareto_avg:.2f}")
        log_trace(
            _trace_banner("Seed candidate C0 (full task evaluation)")
            + f"pareto_avg (mean final_score): {seed_candidate.pareto_avg:.2f}\n\n"
            + "SEED_PROMPT:\n"
            + textwrap.indent(seed_candidate.prompt.rstrip(), "  ")
            + "\n\n"
            + format_results_block(seed_results)
            + "\n"
        )

        for step in range(1, MAX_ITERS + 1):
            parent = sample_parent_from_frontier(candidates, task_ids)
            minibatch = choose_feedback_topics(task_items)
            minibatch_ids = [task_id for task_id, _ in minibatch]

            print("-" * 60)
            print(f"Iteration {step}")
            print(f"Selected parent: C{parent.cid} | full avg = {parent.pareto_avg:.2f} | minibatch = {minibatch_ids}")

            # Reuse parent scores on the minibatch from the cached full evaluation.
            parent_minibatch_results = {task_id: parent.topic_results[task_id] for task_id in minibatch_ids}
            parent_minibatch_avg = average_score(parent_minibatch_results)

            child_prompt = reflective_mutate(parent.prompt, parent_minibatch_results)
            child_minibatch_results = evaluate_candidate_on_topics(child_prompt, minibatch)
            child_minibatch_avg = average_score(child_minibatch_results)

            print(f"Parent minibatch avg: {parent_minibatch_avg:.2f}")
            print(f"Child  minibatch avg: {child_minibatch_avg:.2f}")

            iter_head = (
                _trace_banner(f"Iteration {step}/{MAX_ITERS}")
                + f"parent: C{parent.cid}  parent_full_avg={parent.pareto_avg:.2f}\n"
                + f"minibatch task_ids: {minibatch_ids}\n"
                + f"parent_minibatch_avg: {parent_minibatch_avg:.2f}  "
                + f"child_minibatch_avg: {child_minibatch_avg:.2f}  "
                + f"gate: child > parent + {ACCEPT_MARGIN}\n\n"
                + "Parent prompt (excerpt):\n"
                + textwrap.indent(parent.prompt[:1200] + ("..." if len(parent.prompt) > 1200 else ""), "  ")
                + "\n\n"
                + "Proposed child prompt (excerpt):\n"
                + textwrap.indent(child_prompt[:1200] + ("..." if len(child_prompt) > 1200 else ""), "  ")
                + "\n\n"
                + "--- Minibatch: parent topic results ---\n"
                + format_results_block(parent_minibatch_results)
                + "\n\n"
                + "--- Minibatch: child topic results ---\n"
                + format_results_block(child_minibatch_results)
                + "\n"
            )

            if child_minibatch_avg <= parent_minibatch_avg + ACCEPT_MARGIN:
                print("Rejected child on minibatch gate.")
                log_trace(iter_head + "RESULT: rejected (minibatch gate)\n")
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

            print(f"Accepted child C{child.cid} | full avg = {child.pareto_avg:.2f}")

            accepted_tail = (
                f"RESULT: accepted as C{child.cid}\n"
                f"full_child_avg: {child.pareto_avg:.2f}\n\n"
                + "--- Full task evaluation (accepted child) ---\n"
                + format_results_block(full_child_results)
                + "\n"
            )
            log_trace(iter_head + accepted_tail)

            if child.pareto_avg > best_candidate.pareto_avg:
                best_candidate = child
                print(f"New best candidate: C{child.cid}")

        print("=" * 60)
        print(f"Search complete. Best candidate: C{best_candidate.cid}")
        print(f"Best average score: {best_candidate.pareto_avg:.2f}")

        write_run_artifacts(candidates, best_candidate)
        print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")

        log_trace(
            _trace_banner("Run complete")
            + f"finished_at: {datetime.now().isoformat(timespec='seconds')}\n"
            + f"best_candidate: C{best_candidate.cid}  best_avg: {best_candidate.pareto_avg:.2f}\n"
            + f"total_candidates: {len(candidates)}\n"
            + "Also written: best_prompt.txt, search_history.json, detailed_evaluations.json\n"
        )


if __name__ == "__main__":
    try:
        run_gepa_lite()
    except KeyboardInterrupt:
        print("Interrupted by user.")
