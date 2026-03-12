"""
GEPA-style prompt evolution for the Veo 3 ScriptGeneratorTool.

Implements the core GEPA algorithm (Genetic-Pareto, ICLR 2026 Oral):
  1. Maintain a population of candidate system prompts
  2. Evaluate each on a minibatch from prompts.json via critique-based scoring
  3. Reflectively mutate prompts using LLM feedback on failures
  4. Select next candidates via Pareto-based sampling
  5. Persist the full search tree for inspection

Usage:
    python gepa_evolve.py                          # defaults
    python gepa_evolve.py --budget 30 --pop 8      # custom budget & initial population
    python gepa_evolve.py --resume runs/gepa_*.json # resume from a prior run
"""

import os, re, json, time, copy, random, asyncio, argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
PROMPTS_FILE = "prompts.json"
EVAL_MODEL = "gpt-5-nano"
MUTATION_MODEL = "gpt-5-nano"
SCRIPT_GEN_MODEL = "gpt-5-nano"
MINIBATCH_SIZE = 3       # tasks per evaluation round (paper uses b=3)
PARETO_SET_SIZE = None    # None = use all tasks for Pareto scoring

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    id: int
    prompt: str
    parent_id: Optional[int] = None
    generation: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    avg_score: float = 0.0
    mutation_summary: str = ""

    def copy_with_new_id(self, new_id: int) -> "Candidate":
        c = copy.deepcopy(self)
        c.id = new_id
        c.parent_id = self.id
        c.generation = self.generation + 1
        c.scores = {}
        c.avg_score = 0.0
        c.mutation_summary = ""
        return c


@dataclass
class EvalTrace:
    task_key: str
    topic: str
    candidate_id: int
    script_output: Dict[str, str]
    critique: Dict[str, Any]
    score: float
    feedback_text: str


@dataclass
class RunState:
    candidates: List[Candidate]
    traces: List[EvalTrace]
    next_id: int = 0
    rollouts_used: int = 0
    best_candidate_id: int = 0

    def new_id(self) -> int:
        cid = self.next_id
        self.next_id += 1
        return cid


# ---------------------------------------------------------------------------
# Seed prompts — diverse initial population
# ---------------------------------------------------------------------------
SEED_PROMPTS = [
    # 0 — Original prompt from deep.py (baseline)
    """You are writing a prompt for Veo 3, a text-to-video AI model that generates photorealistic real-world footage. 
Your output will be fed directly to the model as its generation prompt, so write in vivid, descriptive, present-tense language—as if narrating what the camera sees moment by moment.

TOPIC: "{topic}"
BACKGROUND RESEARCH (use for factual accuracy): {context}

SCRIPT REQUIREMENTS:
- The clip is exactly 8 seconds of continuous, photorealistic footage. No cuts, transitions, or scene changes.
- Describe a single real-world scene with one clear subject performing one simple, observable action.
- Write in temporal order: what the viewer sees at the start, what unfolds over the 8 seconds, and how the shot ends.
- Use specific, concrete visual details—materials, textures, colors, lighting quality, and spatial relationships.
- Camera: stationary or slow dolly/pan only. Specify the angle (e.g. eye-level, overhead, 45-degree). No zooms, whip pans, handheld shake, or rack focus.
- Lighting: specify the light source (e.g. soft window light, golden hour sun, overhead fluorescents).
- Keep the scene grounded and filmable—nothing fantastical, animated, graphical, or text-overlay based.
- Avoid: multiple competing subjects, complex multi-step actions, grid/pattern layouts, technical measurement setups, or scanning motions.

NARRATION REQUIREMENTS:
- One spoken sentence, max 18 words, that a voiceover narrator would say over the clip.
- Should complement the visuals—not merely describe what is shown, but add insight or context.

OUTPUT (valid JSON only, no other text):
{{
    "script": "Present-tense, temporally ordered description of the 8-second clip with specific visual and cinematic details.",
    "narration": "A single conversational sentence adding insight to the visuals (max 18 words)."
}}"""
]


# ---------------------------------------------------------------------------
# Evaluation: two-stage (hard checks → calibrated quality)
# ---------------------------------------------------------------------------
def generate_script(prompt_template: str, topic: str, context: str = "None provided.") -> Dict[str, str]:
    """Run the script generator with a given system prompt template."""
    filled = prompt_template.replace("{topic}", topic).replace("{context}", context)

    r = client.chat.completions.create(
        model=SCRIPT_GEN_MODEL,
        messages=[{"role": "user", "content": filled}],
        # max_completion_tokens=1200,
        temperature=1,
    )
    content = r.choices[0].message.content
    if not content:
        raise ValueError(f"Empty response from model. Finish reason: {r.choices[0].finish_reason}")
    raw = content.strip()
    raw = _clean_json(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [DEBUG] JSON parse failed. Raw content ({len(raw)} chars):\n{raw[:500]}")
        raise


# ---- Stage 1: deterministic hard-constraint checks ----

def stage1_hard_checks(script_data: Dict[str, str], topic: str) -> Dict[str, Any]:
    """
    Binary pass/fail on verifiable constraints. Each check is phrased as a
    concrete yes/no question with an extraction requirement so the evaluator
    must cite evidence rather than hand-wave.
    """
    script = script_data.get("script", "")
    narration = script_data.get("narration", "")

    prompt = f"""You are a QA tester for Veo 3 video prompts. Your job is to run hard-constraint checks.
For EACH check below, you MUST:
  (a) quote the specific words from the script that are relevant,
  (b) give a verdict of PASS or FAIL,
  (c) if FAIL, state exactly what is wrong.

SCRIPT: "{script}"
NARRATION: "{narration}"
TOPIC: {topic}

CHECKS:

C1_JSON_VALID: Does the output contain exactly two keys "script" and "narration" with non-empty string values?

C2_SINGLE_SUBJECT: Does the script describe exactly ONE primary subject (person, animal, or object)? Quote the subject. FAIL if two or more distinct subjects perform separate actions.

C3_SINGLE_ACTION: Does the subject perform exactly ONE continuous action across the 8 seconds? Quote the action. FAIL if there are multiple sequential steps (e.g., "first X, then Y, finally Z" with 3+ distinct verbs describing different activities).

C4_CAMERA_SPECIFIED: Does the script explicitly name a camera angle (e.g., "eye-level", "overhead", "low angle", "45-degree")? Quote it. FAIL if no angle is stated.

C5_CAMERA_LEGAL: Is the camera movement stationary, slow dolly, or slow pan ONLY? FAIL if any of these appear: zoom, whip pan, rack focus, handheld, tracking shot, crane, drone sweep, or if movement is not stated.

C6_NO_TEXT_OVERLAYS: Does the script avoid describing any on-screen text, titles, labels, captions, subtitles, watermarks, or graphics? FAIL if any text element is described as appearing in the video.

C7_NO_FANTASY: Is every visual element photorealistic and real-world? FAIL if animation, illustration, cartoon, CGI, fantasy creatures, magic, or surreal imagery is described.

C8_LIGHTING_SPECIFIED: Does the script name a concrete light source (e.g., "sunlight", "window light", "fluorescent", "golden hour")? Quote it. FAIL if lighting is not mentioned or is only vague ("good lighting").

C9_NARRATION_WORD_COUNT: Count the exact words in the narration. List them numbered. FAIL if more than 18 words.

C10_NARRATION_IS_ONE_SENTENCE: Does the narration consist of exactly one sentence? FAIL if it contains two or more sentences (look for multiple periods, semicolons used as sentence separators, or coordinating conjunctions joining independent clauses with a comma).

C11_TEMPORAL_ORDER: Does the script describe events in chronological order with some indication of temporal progression (beginning/middle/end, or phrases like "begins", "over the next seconds", "by the end")? FAIL if the description is a static snapshot with no temporal movement.

C12_TOPIC_FIDELITY: Does the script depict the specific subject/action/setting described in the TOPIC? FAIL if the script describes something substantially different from what the topic asks for (e.g., topic says "French Press coffee" but script shows "pour-over coffee").

Respond with ONLY valid JSON:
{{
    "checks": {{
        "C1_JSON_VALID": {{"verdict": "PASS or FAIL", "evidence": "quoted text", "reason": "..."}},
        "C2_SINGLE_SUBJECT": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C3_SINGLE_ACTION": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C4_CAMERA_SPECIFIED": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C5_CAMERA_LEGAL": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C6_NO_TEXT_OVERLAYS": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C7_NO_FANTASY": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C8_LIGHTING_SPECIFIED": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C9_NARRATION_WORD_COUNT": {{"verdict": "...", "evidence": "word count: N", "reason": "..."}},
        "C10_NARRATION_IS_ONE_SENTENCE": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C11_TEMPORAL_ORDER": {{"verdict": "...", "evidence": "...", "reason": "..."}},
        "C12_TOPIC_FIDELITY": {{"verdict": "...", "evidence": "...", "reason": "..."}}
    }},
    "pass_count": 0,
    "fail_count": 0,
    "failed_checks": ["C1_JSON_VALID", "..."]
}}"""

    r = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1500,
        temperature=1,  
    )
    raw = r.choices[0].message.content.strip()
    raw = _clean_json(raw)
    return json.loads(raw)


# ---- Stage 2: calibrated quality scoring with exemplars ----

EXEMPLARS = """
=== EXEMPLAR A (score: 92/100 — strong) ===
TOPIC: "A person correctly using a French Press to brew coffee, countertop view."
SCRIPT: "On a sunlit granite countertop, a pair of hands grips a glass-bodied French Press filled with dark, blooming coffee grounds. Warm morning light from a nearby window catches the amber liquid. The hands slowly press the stainless-steel plunger downward at a steady pace, the mesh filter pushing grounds to the bottom. By the final second the plunger reaches the base and the person's thumb rests on top."
NARRATION: "Four minutes of patience turns coarse grounds into a full-bodied brew."
WHY 92: Strong visual specificity (granite, glass, stainless steel, amber). Clear temporal arc (grip → press → rest). Single subject/action. Camera angle implied but not explicitly named—loses a few points. Narration adds brewing insight. Light source named.

=== EXEMPLAR B (score: 55/100 — mediocre) ===
TOPIC: "A person correctly using a French Press to brew coffee, countertop view."
SCRIPT: "A person makes coffee with a French Press on a kitchen counter. They put in the coffee, add water, wait, and press down the plunger. The camera shows the whole process."
NARRATION: "French Press coffee is easy to make at home."
WHY 55: Extremely vague—no materials, colors, textures, or lighting. Multi-step action compressed ("put in, add, wait, press"). No camera angle or movement specified. Narration is generic description, not insight. Would likely produce an incoherent Veo output due to trying to show 4 steps in 8 seconds.

=== EXEMPLAR C (score: 25/100 — poor) ===
TOPIC: "A person correctly using a French Press to brew coffee, countertop view."
SCRIPT: "An animated infographic shows the French Press brewing process with step-by-step labels: 'Step 1: Add grounds', 'Step 2: Pour water', 'Step 3: Wait 4 minutes', 'Step 4: Press'. Arrows and icons guide the viewer through each stage."
NARRATION: "Follow these four simple steps to brew the perfect cup of French Press coffee every single time at home."
WHY 25: Text overlays and labels (Veo rejection). Animated infographic (not photorealistic). Multi-step with arrows/icons. Narration exceeds 18 words. Would be immediately rejected by Veo 3.

=== EXEMPLAR D (score: 72/100 — decent but flawed) ===
TOPIC: "A close-up of a Newton's Cradle in motion on a desk, macro view."
SCRIPT: "A macro close-up captures a Newton's Cradle on a dark walnut desk. The leftmost chrome sphere swings up and strikes the row, sending the rightmost sphere arcing outward. The camera slowly zooms in as the spheres click rhythmically."
NARRATION: "Each collision transfers momentum perfectly through the stationary spheres between."
WHY 72: Good specificity (walnut, chrome, macro). Reasonable temporal arc. But "slowly zooms in" is a prohibited camera move (FAIL on C5). Single subject, good narration with physics insight. The zoom violation would cause Veo generation issues.
"""

def stage2_quality_scoring(script_data: Dict[str, str], topic: str, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calibrated quality assessment using exemplar anchoring. The evaluator is
    shown scored examples first to calibrate its scale, then scores the candidate.
    """
    script = script_data.get("script", "")
    narration = script_data.get("narration", "")
    failed_checks = stage1_result.get("failed_checks", [])
    checks = stage1_result.get("checks", {})

    stage1_summary = ""
    for name, check in checks.items():
        verdict = check.get("verdict", "?")
        reason = check.get("reason", "")
        if verdict == "FAIL":
            stage1_summary += f"  FAILED {name}: {reason}\n"

    prompt = f"""You are a calibrated evaluator for Veo 3 video generation prompts.

IMPORTANT: Read the scored exemplars below FIRST to calibrate your scoring scale. A score of 92 means excellent with minor flaws. A score of 55 means vague and multi-step. A score of 25 means policy-violating garbage. Score honestly relative to these anchors.

{EXEMPLARS}

--- NOW EVALUATE THIS CANDIDATE ---

TOPIC: {topic}
SCRIPT: "{script}"
NARRATION: "{narration}"

HARD-CHECK FAILURES FROM STAGE 1 (these are already verified — do NOT override them):
{stage1_summary if stage1_summary else "  None — all hard checks passed."}

SCORING RULES:
- Start from 100 and subtract.
- Each FAILED hard check from Stage 1: subtract 8-15 points depending on severity.
- Then evaluate these quality dimensions (each 0-10 scale, where 5 is average and 10 is exceptional):

Q1_VISUAL_RICHNESS: Are specific materials, textures, and colors named? (e.g., "brushed aluminum" not just "metal"; "amber liquid" not just "coffee")
  - 8-10: 4+ specific material/color/texture terms. Example: "granite countertop, glass-bodied French Press, stainless-steel plunger, amber liquid"
  - 5-7: 2-3 specific terms with some vague descriptions mixed in
  - 0-4: Mostly generic ("a person", "a table", "coffee")

Q2_TEMPORAL_DEPTH: Does the description create a sense of motion and change across 8 seconds, or is it a static pose?
  - 8-10: Clear beginning/middle/end with described motion. Reader can mentally "play" the 8 seconds.
  - 5-7: Some progression but rushed or with gaps
  - 0-4: Static scene description or list of features with no temporal flow

Q3_VEO_PROMPT_CRAFT: Would an experienced Veo user consider this a well-crafted generation prompt? Consider: Is it written for a VIDEO MODEL (temporal, motion-focused) rather than an image model? Does it avoid patterns known to cause Veo failures?
  - 8-10: Reads like a professional Veo prompt. Motion-oriented, avoids all known failure patterns.
  - 5-7: Functional but could be improved. Minor risk of Veo issues.
  - 0-4: Reads like an image description, contains known Veo failure patterns.

Q4_NARRATION_INSIGHT: Does the narration add a fact, context, or perspective that the visuals alone don't convey?
  - 8-10: Adds specific knowledge (a measurement, a historical fact, a causal explanation). Example: "Four minutes of patience turns coarse grounds into a full-bodied brew."
  - 5-7: Somewhat insightful but could be more specific
  - 0-4: Merely restates what's visible, or is generic ("This is how it's done")

Q5_TOPIC_PRECISION: Does the output capture the specific subject, action, and framing described in the topic — not just the general category?
  - 8-10: Matches the exact subject, action, viewpoint, and setting from the topic
  - 5-7: Captures the general idea but misses specific details from the topic
  - 0-4: Only loosely related to the topic

Respond with ONLY valid JSON:
{{
    "quality_scores": {{
        "Q1_VISUAL_RICHNESS": {{"score": 0-10, "evidence": "quote key terms or note absence"}},
        "Q2_TEMPORAL_DEPTH": {{"score": 0-10, "evidence": "describe the temporal arc or lack thereof"}},
        "Q3_VEO_PROMPT_CRAFT": {{"score": 0-10, "evidence": "note motion language or static/image-like phrasing"}},
        "Q4_NARRATION_INSIGHT": {{"score": 0-10, "evidence": "what insight does it add?"}},
        "Q5_TOPIC_PRECISION": {{"score": 0-10, "evidence": "what topic details are captured vs missed?"}}
    }},
    "hard_check_penalty": 0,
    "final_score": 0,
    "feedback": "2-3 sentences: what specific instruction changes in the system prompt would improve this output? Focus on the weakest dimension."
}}"""

    r = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1000,
        temperature=1, 
    )
    raw = r.choices[0].message.content.strip()
    raw = _clean_json(raw)
    return json.loads(raw)


def compute_score(stage1: Dict[str, Any], stage2: Dict[str, Any]) -> float:
    """
    Combine both stages into a single 0-100 fitness score.
    Hard-check failures impose a ceiling; quality scores fill the rest.
    """
    checks = stage1.get("checks", {})
    num_checks = len(checks)
    fail_count = sum(1 for c in checks.values() if c.get("verdict") == "FAIL")
    pass_rate = (num_checks - fail_count) / num_checks if num_checks else 0

    q_scores = stage2.get("quality_scores", {})
    q_values = [v.get("score", 0) if isinstance(v, dict) else v for v in q_scores.values()]
    quality_avg = sum(q_values) / len(q_values) if q_values else 0

    # Hard checks gate the score: each failure reduces the ceiling
    # 0 failures → ceiling 100, 1 failure → ceiling ~88, 2 → ~76, etc.
    ceiling = 100 * (pass_rate ** 0.5)

    # Quality component: scale 0-10 average to 0-100 range, then cap at ceiling
    quality_pct = quality_avg * 10
    raw_score = min(quality_pct, ceiling)

    # Apply the model's own penalty as a cross-check
    model_penalty = stage2.get("hard_check_penalty", 0)
    model_final = stage2.get("final_score", raw_score)

    # Average our computed score with the model's to reduce single-point bias
    blended = (raw_score + model_final) / 2
    return round(max(0, min(100, blended)), 2)


def evaluate_candidate(candidate: Candidate, tasks: List[Dict[str, str]]) -> List[EvalTrace]:
    """Evaluate a candidate prompt on a list of (key, topic) tasks."""
    traces = []
    for task in tasks:
        key = task["key"]
        topic = task["topic"]
        try:
            script_data = generate_script(candidate.prompt, topic)
        except Exception as e:
            script_data = {"script": f"GENERATION_ERROR: {e}", "narration": ""}

        try:
            stage1 = stage1_hard_checks(script_data, topic)
        except Exception as e:
            stage1 = {"checks": {}, "pass_count": 0, "fail_count": 12, "failed_checks": ["EVAL_ERROR"]}

        try:
            stage2 = stage2_quality_scoring(script_data, topic, stage1)
        except Exception as e:
            stage2 = {"quality_scores": {}, "hard_check_penalty": 50, "final_score": 0, "feedback": str(e)}

        score = compute_score(stage1, stage2)

        critique = {"stage1": stage1, "stage2": stage2}

        s1_failures = stage1.get("failed_checks", [])
        s2_feedback = stage2.get("feedback", "")
        s2_scores = stage2.get("quality_scores", {})
        weakest = ""
        if s2_scores:
            weakest_dim = min(s2_scores.items(), key=lambda x: x[1].get("score", 10) if isinstance(x[1], dict) else x[1])
            weakest = f"Weakest: {weakest_dim[0]}"

        feedback_parts = []
        if s1_failures:
            feedback_parts.append("HARD-CHECK FAILURES:\n" + "\n".join(f"- {f}" for f in s1_failures))
        if weakest:
            feedback_parts.append(weakest)
        if s2_feedback:
            feedback_parts.append(f"QUALITY FEEDBACK:\n{s2_feedback}")
        feedback = "\n\n".join(feedback_parts)

        trace = EvalTrace(
            task_key=key,
            topic=topic,
            candidate_id=candidate.id,
            script_output=script_data,
            critique=critique,
            score=score,
            feedback_text=feedback,
        )
        traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Reflective prompt mutation (GEPA Appendix C meta-prompt)
# ---------------------------------------------------------------------------
def reflective_mutate(candidate: Candidate, traces: List[EvalTrace]) -> str:
    """
    Use the GEPA reflection meta-prompt to propose an improved system prompt
    based on the candidate's current prompt and its evaluation traces.
    """
    examples_block = ""
    for t in traces:
        examples_block += f"""
---
Task topic: {t.topic}
Assistant output:
  script: {t.script_output.get("script", "N/A")}
  narration: {t.script_output.get("narration", "N/A")}
Score: {t.score}/100
Feedback: {t.feedback_text}
---
"""

    meta_prompt = f"""I provided an assistant with the following instructions to perform a task for me:
```
{candidate.prompt}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
{examples_block}
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Important domain knowledge for this task:
- The assistant writes prompts for Veo 3, a text-to-video AI that generates photorealistic real-world footage.
- Output must be valid JSON with "script" and "narration" keys.
- Veo 3 rejects: text overlays, graphics, animation, fantasy, zooms, whip pans, rack focus, scanning motions, grid layouts, multi-step processes, abstract imagery, multiple competing subjects.
- Camera must be stationary or slow dolly/pan. Angle must be specified.
- The clip is exactly 8 seconds of continuous footage with no cuts.
- Narration is one sentence, max 18 words, adding insight beyond the visuals.

Focus your improvements on the specific issues identified in the feedback. Make targeted changes rather than rewriting from scratch. Preserve what works well and fix what doesn't.

Provide the new instructions within ``` blocks."""

    r = client.chat.completions.create(
        model=MUTATION_MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        max_completion_tokens=2000,
        temperature=1, 
    )
    response = r.choices[0].message.content.strip()

    match = re.search(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response


# ---------------------------------------------------------------------------
# Pareto-based candidate selection (Algorithm 2 from the paper)
# ---------------------------------------------------------------------------
def select_candidate_pareto(candidates: List[Candidate], task_keys: List[str]) -> Candidate:
    """
    Pareto-based selection: find candidates that are best on at least one task,
    prune dominated ones, then sample proportionally to how many tasks each leads.
    """
    if len(candidates) == 1:
        return candidates[0]

    best_per_task: Dict[str, List[Candidate]] = {}
    for key in task_keys:
        best_score = -1.0
        for c in candidates:
            s = c.scores.get(key, 0.0)
            if s > best_score:
                best_score = s
        best_per_task[key] = [c for c in candidates if c.scores.get(key, 0.0) == best_score]

    pareto_set = set()
    for winners in best_per_task.values():
        for c in winners:
            pareto_set.add(c.id)

    pareto_candidates = [c for c in candidates if c.id in pareto_set]

    dominated = set()
    for c in pareto_candidates:
        for other in pareto_candidates:
            if c.id == other.id:
                continue
            if all(other.scores.get(k, 0) >= c.scores.get(k, 0) for k in task_keys) and \
               any(other.scores.get(k, 0) > c.scores.get(k, 0) for k in task_keys):
                dominated.add(c.id)
                break

    non_dominated = [c for c in pareto_candidates if c.id not in dominated]
    if not non_dominated:
        non_dominated = pareto_candidates if pareto_candidates else candidates

    freq: Dict[int, int] = {c.id: 0 for c in non_dominated}
    for key in task_keys:
        surviving = [c for c in non_dominated if c in best_per_task.get(key, [])]
        for c in surviving:
            freq[c.id] += 1

    total = sum(freq.values())
    if total == 0:
        return random.choice(non_dominated)

    weights = [freq[c.id] / total for c in non_dominated]
    return random.choices(non_dominated, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_state(state: RunState, path: Path):
    data = {
        "candidates": [asdict(c) for c in state.candidates],
        "traces": [asdict(t) for t in state.traces],
        "next_id": state.next_id,
        "rollouts_used": state.rollouts_used,
        "best_candidate_id": state.best_candidate_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_state(path: Path) -> RunState:
    with open(path) as f:
        data = json.load(f)
    candidates = [Candidate(**c) for c in data["candidates"]]
    traces = [EvalTrace(**t) for t in data["traces"]]
    return RunState(
        candidates=candidates,
        traces=traces,
        next_id=data["next_id"],
        rollouts_used=data["rollouts_used"],
        best_candidate_id=data["best_candidate_id"],
    )


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def load_tasks(path: str = PROMPTS_FILE) -> List[Dict[str, str]]:
    with open(path) as f:
        raw = json.load(f)
    tasks = []
    for key, topic in raw.items():
        tasks.append({"key": key, "topic": topic})
    return tasks


def sample_minibatch(tasks: List[Dict[str, str]], size: int) -> List[Dict[str, str]]:
    return random.sample(tasks, min(size, len(tasks)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_json(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def print_candidate_summary(c: Candidate, prefix: str = ""):
    task_scores = ", ".join(f"{k}: {v}" for k, v in sorted(c.scores.items())[:5])
    print(f"{prefix}[C{c.id}] gen={c.generation} avg={c.avg_score:.1f} parent=C{c.parent_id} | {task_scores}...")


def summarize_mutation(old_prompt: str, new_prompt: str) -> str:
    """Ask the LLM for a one-line diff summary."""
    r = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": f"In one sentence, what changed between these two prompts?\n\nOLD:\n{old_prompt[:1500]}\n\nNEW:\n{new_prompt[:1500]}"}],
        max_completion_tokens=100,
        temperature=1,  
    )
    return r.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------
def run_gepa(
    budget: int = 20,
    initial_pop_size: int = 6,
    minibatch_size: int = MINIBATCH_SIZE,
    resume_path: Optional[str] = None,
    output_dir: str = "runs",
):
    tasks = load_tasks()
    all_task_keys = [t["key"] for t in tasks]

    if resume_path:
        print(f"Resuming from {resume_path}")
        state = load_state(Path(resume_path))
    else:
        state = RunState(candidates=[], traces=[], next_id=0, rollouts_used=0)

        # --- Initialize population ---
        print(f"Initializing population with {initial_pop_size} seed prompts...")
        for i in range(min(initial_pop_size, len(SEED_PROMPTS))):
            cid = state.new_id()
            c = Candidate(id=cid, prompt=SEED_PROMPTS[i], generation=0)
            state.candidates.append(c)

        # --- Evaluate initial population on full Pareto set ---
        print(f"Evaluating initial population on {len(tasks)} tasks...")
        for c in state.candidates:
            print(f"  Evaluating C{c.id}...")
            traces = evaluate_candidate(c, tasks)
            state.traces.extend(traces)
            for t in traces:
                c.scores[t.task_key] = t.score
            c.avg_score = sum(c.scores.values()) / len(c.scores) if c.scores else 0
            state.rollouts_used += len(traces)
            print_candidate_summary(c, prefix="    ")

    # --- Find best so far ---
    best = max(state.candidates, key=lambda c: c.avg_score)
    state.best_candidate_id = best.id
    print(f"\nBest after init: C{best.id} (avg={best.avg_score:.1f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(output_dir) / f"gepa_{timestamp}.json"
    save_state(state, save_path)

    # --- Evolution loop ---
    iteration = 0
    while state.rollouts_used < budget * len(tasks):
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration} | rollouts={state.rollouts_used} | budget={budget * len(tasks)}")
        print(f"{'='*70}")

        # 1. Select candidate via Pareto sampling
        parent = select_candidate_pareto(state.candidates, all_task_keys)
        print(f"Selected C{parent.id} (avg={parent.avg_score:.1f}) for mutation")

        # 2. Sample minibatch and gather feedback
        minibatch = sample_minibatch(tasks, minibatch_size)
        print(f"Evaluating C{parent.id} on minibatch of {len(minibatch)} tasks for feedback...")
        mb_traces = evaluate_candidate(parent, minibatch)
        state.rollouts_used += len(mb_traces)

        mb_avg = sum(t.score for t in mb_traces) / len(mb_traces)
        print(f"  Minibatch avg score: {mb_avg:.1f}")

        # 3. Reflective mutation
        print(f"Reflecting and proposing new prompt...")
        new_prompt = reflective_mutate(parent, mb_traces)
        child = parent.copy_with_new_id(state.new_id())
        child.prompt = new_prompt

        # 4. Evaluate child on same minibatch
        print(f"Evaluating mutated C{child.id} on minibatch...")
        child_mb_traces = evaluate_candidate(child, minibatch)
        state.rollouts_used += len(child_mb_traces)

        child_mb_avg = sum(t.score for t in child_mb_traces) / len(child_mb_traces)
        print(f"  Mutated minibatch avg: {child_mb_avg:.1f} (parent was {mb_avg:.1f})")

        if child_mb_avg <= 0:
            print("  EARLY STOP: iteration returned 0.0, exiting search.")
            state.traces.extend(mb_traces)
            save_state(state, save_path)
            print(f"  State saved to {save_path}")
            break

        # 5. Accept if improved on minibatch
        if child_mb_avg > mb_avg:
            print(f"  IMPROVED +{child_mb_avg - mb_avg:.1f} — evaluating on full Pareto set...")

            full_traces = evaluate_candidate(child, tasks)
            state.traces.extend(full_traces)
            state.rollouts_used += len(full_traces)
            for t in full_traces:
                child.scores[t.task_key] = t.score
            child.avg_score = sum(child.scores.values()) / len(child.scores) if child.scores else 0

            child.mutation_summary = summarize_mutation(parent.prompt, child.prompt)
            state.candidates.append(child)
            print_candidate_summary(child, prefix="    ACCEPTED: ")
            print(f"    Mutation: {child.mutation_summary}")

            if child.avg_score > best.avg_score:
                best = child
                state.best_candidate_id = best.id
                print(f"    *** NEW BEST: C{best.id} (avg={best.avg_score:.1f}) ***")
        else:
            print(f"  REJECTED (no improvement on minibatch)")
            state.traces.extend(mb_traces)

        save_state(state, save_path)
        print(f"  State saved to {save_path}")

    # --- Final report ---
    print(f"\n{'='*70}")
    print(f"EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total rollouts: {state.rollouts_used}")
    print(f"Candidates explored: {len(state.candidates)}")
    print(f"Best candidate: C{best.id} (gen={best.generation}, avg={best.avg_score:.1f})")
    print(f"\nBest prompt:\n{best.prompt}")

    # Write best prompt to a standalone file
    best_path = Path(output_dir) / f"best_prompt_{timestamp}.txt"
    with open(best_path, "w") as f:
        f.write(best.prompt)
    print(f"\nBest prompt saved to: {best_path}")

    # Write leaderboard
    sorted_candidates = sorted(state.candidates, key=lambda c: c.avg_score, reverse=True)
    print(f"\nLeaderboard:")
    for rank, c in enumerate(sorted_candidates, 1):
        lineage = f"C{c.parent_id}→C{c.id}" if c.parent_id is not None else f"C{c.id} (seed)"
        print(f"  #{rank}: avg={c.avg_score:.1f} gen={c.generation} {lineage} | {c.mutation_summary[:80]}")

    save_state(state, save_path)
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEPA prompt evolution for Veo 3 script generator")
    parser.add_argument("--budget", type=int, default=20,
                        help="Max iterations of the evolution loop (each uses ~2 minibatch + 1 full eval)")
    parser.add_argument("--pop", type=int, default=6,
                        help="Initial population size (max 6 seed prompts available)")
    parser.add_argument("--minibatch", type=int, default=MINIBATCH_SIZE,
                        help="Tasks per minibatch evaluation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a prior run JSON to resume from")
    parser.add_argument("--output", type=str, default="runs",
                        help="Output directory for run state files")
    args = parser.parse_args()

    best = run_gepa(
        budget=args.budget,
        initial_pop_size=args.pop,
        minibatch_size=args.minibatch,
        resume_path=args.resume,
        output_dir=args.output,
    )
