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

# GEPA paper hyperparameters (Appendix E.2, E.4)
INFERENCE_TEMPERATURE = 0.7   # Paper uses 0.6-1.0; lower = more reliable JSON
MUTATION_TEMPERATURE = 0.9    # Slightly higher for creative mutations
MAX_RETRIES = 3               # Retry failed LLM calls
MERGE_INTERVAL = 5            # Invoke merge every N iterations (paper: up to 5 per run)

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
# Seed prompts — diverse initial population (GEPA requires diversity!)
# ---------------------------------------------------------------------------
SEED_PROMPTS = [
    # 0 — Original baseline (cinematography-focused)
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
}}""",

    # 1 — Temporal structure emphasis (different strategy: explicit beat breakdown)
    """You generate Veo 3 video prompts. Veo 3 creates photorealistic 8-second video clips.

TOPIC: "{topic}"
CONTEXT: {context}

STRUCTURE YOUR SCRIPT IN THREE BEATS:
- BEAT 1 (0-2s): Establish the scene. Name the setting, lighting, and introduce the subject at rest or mid-action.
- BEAT 2 (3-6s): The core action unfolds. Describe the single continuous motion with physical details.
- BEAT 3 (7-8s): Resolution. The action completes or reaches a natural pause.

TECHNICAL CONSTRAINTS:
- Camera: State angle (eye-level/overhead/low-angle/45-degree) and movement (stationary OR slow dolly/pan only).
- Lighting: Name the source (sunlight, window light, fluorescent, golden hour, etc.).
- Materials: Use specific terms (brushed steel, oak grain, matte ceramic) not generic (metal, wood, cup).
- FORBIDDEN: cuts, zooms, rack focus, text overlays, graphics, animation, fantasy, multiple subjects.

NARRATION: One sentence (max 18 words) that teaches something or provides context the visuals cannot show.

Return ONLY valid JSON:
{{"script": "...", "narration": "..."}}""",

    # 2 — Minimalist/constraint-focused (different strategy: what NOT to do)
    """Write a Veo 3 video generation prompt for an 8-second photorealistic clip.

TOPIC: "{topic}"
RESEARCH: {context}

GOLDEN RULES:
1. ONE subject, ONE action, ONE continuous shot.
2. Camera angle MUST be stated (eye-level, overhead, 45-degree, low-angle).
3. Camera movement: stationary or slow dolly/pan ONLY. No zoom, no rack focus, no handheld.
4. Light source MUST be named (e.g., "soft daylight from left," "overhead fluorescent").
5. Use 3+ specific material/texture words (e.g., "polished granite," "brushed aluminum," "worn leather").
6. Temporal flow: describe start → middle → end of the 8 seconds.

ABSOLUTE BANS (Veo will reject):
- Text, labels, captions, watermarks, infographics
- Animation, CGI, fantasy, surreal imagery
- Multiple sequential actions ("first... then... finally...")
- Grid layouts, measurement setups, scanning motions

NARRATION: One sentence, ≤18 words, adds insight (a fact, measurement, or context) beyond what's visible.

Output JSON only: {{"script": "...", "narration": "..."}}""",

    # 3 — Sensory immersion (different strategy: texture/atmosphere first)
    """You are a cinematographer writing a shot description for Veo 3's AI video generator.

TOPIC: "{topic}"
BACKGROUND: {context}

START WITH ATMOSPHERE:
First, establish the sensory environment: What does the light feel like? What textures dominate? What's the ambient mood? Then introduce your subject within this environment.

DESCRIBE THE MOTION:
A single subject performs one fluid action across 8 seconds. Capture the physics: weight, momentum, resistance. Use verbs that imply duration (glides, settles, presses, unfurls) not instant actions (snaps, flashes, pops).

CAMERA RULES:
- Angle: explicitly state (eye-level, overhead, low-angle, 45-degree, macro close-up)
- Movement: stationary OR slow dolly/pan. Nothing else.
- Veo rejects: zooms, whip pans, rack focus, handheld, drone shots, tracking shots.

FORBIDDEN CONTENT: text overlays, graphics, animation, CGI, fantasy, multiple subjects, multi-step sequences.

NARRATION: A single sentence (max 18 words) that reveals something the image alone cannot tell.

JSON output only: {{"script": "...", "narration": "..."}}""",

    # 4 — Checklist-driven (different strategy: explicit verification steps)
    """Generate a Veo 3 prompt for the topic below. Before outputting, verify each checklist item.

TOPIC: "{topic}"
CONTEXT: {context}

PRE-OUTPUT CHECKLIST (verify each before writing):
□ Single subject identified? (one person, animal, or object)
□ Single continuous action? (not multi-step: avoid "first X, then Y")
□ Camera angle named? (eye-level / overhead / low-angle / 45-degree / macro)
□ Camera movement valid? (stationary OR slow dolly/pan — NO zoom, rack focus, handheld)
□ Light source specified? (natural/artificial, direction)
□ 3+ material/texture descriptors? (specific: "walnut," "chrome," "linen" — not "wood," "metal," "cloth")
□ Temporal arc present? (beginning state → action → end state over 8 seconds)
□ No banned elements? (text, graphics, animation, fantasy, multiple subjects)

NARRATION CHECKLIST:
□ One sentence only?
□ ≤18 words?
□ Adds information beyond visuals? (a fact, measurement, or insight)

Output valid JSON only:
{{"script": "...", "narration": "..."}}""",

    # 5 — Example-anchored (different strategy: show before tell)
    """You write Veo 3 video prompts. Study this example of a GOOD prompt:

EXAMPLE (score 92/100):
Topic: "French Press coffee brewing"
Script: "On a sunlit granite countertop, a pair of hands grips a glass-bodied French Press filled with dark, blooming coffee grounds. Warm morning light from a nearby window catches the amber liquid. The hands slowly press the stainless-steel plunger downward at a steady pace, the mesh filter pushing grounds to the bottom. By the final second the plunger reaches the base."
Narration: "Four minutes of patience turns coarse grounds into a full-bodied brew."

WHY IT WORKS: Specific materials (granite, glass, stainless steel). Single subject/action. Named light source. Clear temporal arc. Narration adds brewing time fact.

---

NOW WRITE FOR:
TOPIC: "{topic}"
CONTEXT: {context}

REQUIREMENTS:
- 8-second continuous photorealistic footage, no cuts
- One subject, one action
- Camera: state angle + movement (stationary or slow dolly/pan only)
- Light source named
- 3+ specific materials/textures
- Temporal flow (start → middle → end)
- No text/graphics/animation/fantasy
- Narration: 1 sentence, ≤18 words, adds insight

JSON only: {{"script": "...", "narration": "..."}}\n"""
]


# ---------------------------------------------------------------------------
# Evaluation: two-stage (hard checks → calibrated quality)
# ---------------------------------------------------------------------------
def _llm_call_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int = 4000,
    parse_json: bool = True,
    label: str = "llm_call",
) -> Any:
    """
    Robust LLM call with retry logic for JSON parsing failures.
    GEPA paper emphasizes reliability of evaluation - retries are essential.
    """
    last_error = None
    # Some OpenAI models (including gpt-5-nano) only support the default temperature.
    include_temperature = not model.startswith("gpt-5")
    current_max_tokens = max_tokens
    for attempt in range(MAX_RETRIES):
        try:
            request_kwargs = dict(
                model=model,
                messages=messages,
                max_completion_tokens=current_max_tokens,
            )
            if include_temperature:
                request_kwargs["temperature"] = temperature
            if parse_json:
                request_kwargs["response_format"] = {"type": "json_object"}

            r = client.chat.completions.create(**request_kwargs)
            finish_reason = r.choices[0].finish_reason
            content = r.choices[0].message.content
            if finish_reason == "length" and not content:
                raise ValueError(f"Empty response. Finish reason: {finish_reason}")
            if not content:
                raise ValueError(f"Empty response. Finish reason: {finish_reason}")
            
            raw = content.strip()
            if parse_json:
                raw = _clean_json(raw)
                return json.loads(raw)
            return raw
        except json.JSONDecodeError as e:
            last_error = e
            # Lower temperature on retry for more deterministic output
            temperature = max(0.3, temperature - 0.2)
            if attempt < MAX_RETRIES - 1:
                print(f"    [RETRY {attempt+1}/{MAX_RETRIES}] JSON parse failed, retrying with temp={temperature:.1f}")
                time.sleep(0.5)  # Brief backoff
        except Exception as e:
            last_error = e
            if "temperature' does not support" in str(e):
                include_temperature = False
            if "Finish reason: length" in str(e):
                current_max_tokens = min(current_max_tokens * 2, 8000)
            if attempt < MAX_RETRIES - 1:
                print(f"    [RETRY {attempt+1}/{MAX_RETRIES}] {label}: {e}")
                time.sleep(1.0)
    
    raise last_error


def generate_script(prompt_template: str, topic: str, context: str = "None provided.") -> Dict[str, str]:
    """Run the script generator with a given system prompt template."""
    filled = prompt_template.replace("{topic}", topic).replace("{context}", context)

    return _llm_call_with_retry(
        model=SCRIPT_GEN_MODEL,
        messages=[{"role": "user", "content": filled}],
        temperature=INFERENCE_TEMPERATURE,
        max_tokens=4000,
        parse_json=True,
        label="generate_script",
    )


# ---- Stage 1: deterministic hard-constraint checks ----

def stage1_hard_checks(script_data: Dict[str, str], topic: str) -> Dict[str, Any]:
    """
    Binary pass/fail on verifiable constraints. Each check is phrased as a
    concrete yes/no question with an extraction requirement so the evaluator
    must cite evidence rather than hand-wave.
    """
    script = script_data.get("script", "")
    narration = script_data.get("narration", "")

    prompt = f"""You are a QA tester for Veo 3 video prompts. Run compact hard-constraint checks.
Be brief. Only give reasons for failed checks. Keep each reason under 12 words.

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
        "C1_JSON_VALID": "PASS or FAIL",
        "C2_SINGLE_SUBJECT": "PASS or FAIL",
        "C3_SINGLE_ACTION": "PASS or FAIL",
        "C4_CAMERA_SPECIFIED": "PASS or FAIL",
        "C5_CAMERA_LEGAL": "PASS or FAIL",
        "C6_NO_TEXT_OVERLAYS": "PASS or FAIL",
        "C7_NO_FANTASY": "PASS or FAIL",
        "C8_LIGHTING_SPECIFIED": "PASS or FAIL",
        "C9_NARRATION_WORD_COUNT": "PASS or FAIL",
        "C10_NARRATION_IS_ONE_SENTENCE": "PASS or FAIL",
        "C11_TEMPORAL_ORDER": "PASS or FAIL",
        "C12_TOPIC_FIDELITY": "PASS or FAIL"
    }},
    "pass_count": 0,
    "fail_count": 0,
    "failed_checks": ["C1_JSON_VALID", "..."],
    "reasons": {{
        "C1_JSON_VALID": "reason only if failed"
    }}
}}"""

    return _llm_call_with_retry(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=INFERENCE_TEMPERATURE,
        max_tokens=1200,
        parse_json=True,
        label="stage1_hard_checks",
    )


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
    reasons = stage1_result.get("reasons", {})

    stage1_summary = ""
    for name, check in checks.items():
        verdict = check if isinstance(check, str) else check.get("verdict", "?")
        reason = reasons.get(name, "") if isinstance(check, str) else check.get("reason", "")
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

    return _llm_call_with_retry(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=INFERENCE_TEMPERATURE,
        max_tokens=1800,
        parse_json=True,
        label="stage2_quality_scoring",
    )


def compute_score(stage1: Dict[str, Any], stage2: Dict[str, Any]) -> float:
    """
    Combine both stages into a single 0-100 fitness score.
    Hard-check failures impose a ceiling; quality scores fill the rest.
    """
    checks = stage1.get("checks", {})
    num_checks = len(checks)
    fail_count = sum(
        1
        for c in checks.values()
        if (c == "FAIL") or (isinstance(c, dict) and c.get("verdict") == "FAIL")
    )
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
    
    Key insight from GEPA paper: The reflection prompt should:
    1. Present execution traces (what the LLM produced)
    2. Present evaluation traces (scores + textual feedback)
    3. Ask for targeted improvements based on failure patterns
    """
    # Separate successful vs failed traces for better signal
    failed_traces = [t for t in traces if t.score < 60]
    good_traces = [t for t in traces if t.score >= 60]
    
    examples_block = ""
    
    # Show failures first with detailed feedback
    if failed_traces:
        examples_block += "\n=== CASES NEEDING IMPROVEMENT ===\n"
        for t in failed_traces:
            examples_block += f"""
---
Topic: {t.topic}
Output:
  script: {t.script_output.get("script", "N/A")[:500]}
  narration: {t.script_output.get("narration", "N/A")}
Score: {t.score}/100
Feedback: {t.feedback_text}
---
"""
    
    # Show successes to preserve what works
    if good_traces:
        examples_block += "\n=== SUCCESSFUL CASES (preserve these patterns) ===\n"
        for t in good_traces:
            examples_block += f"""
---
Topic: {t.topic}
Score: {t.score}/100
Key strengths from feedback: {t.feedback_text[:200] if t.feedback_text else "Good overall"}
---
"""

    # Analyze failure patterns
    failure_patterns = {}
    for t in traces:
        critique = t.critique.get("stage1", {})
        for check_name, check_data in critique.get("checks", {}).items():
            if check_data == "FAIL" or (isinstance(check_data, dict) and check_data.get("verdict") == "FAIL"):
                failure_patterns[check_name] = failure_patterns.get(check_name, 0) + 1
    
    pattern_summary = ""
    if failure_patterns:
        sorted_failures = sorted(failure_patterns.items(), key=lambda x: -x[1])
        pattern_summary = "Most common failures: " + ", ".join(f"{k}({v}x)" for k, v in sorted_failures[:5])

    meta_prompt = f"""I provided an assistant with the following instructions to perform a task for me:
```
{candidate.prompt}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and evaluation feedback:
```
{examples_block}
```

{pattern_summary}

Your task is to write an IMPROVED version of the instruction that addresses the failures while preserving successful patterns.

ANALYSIS STEPS:
1. Identify the specific failure modes from the feedback (e.g., missing camera angle, multi-step actions, word count violations).
2. For each failure mode, determine what instruction change would prevent it.
3. Check what patterns appear in successful outputs and ensure they are reinforced.
4. Extract any domain-specific knowledge from the feedback that should be codified.

DOMAIN KNOWLEDGE (must be incorporated):
- Veo 3 generates 8-second photorealistic video clips with no cuts.
- Output format: JSON with exactly "script" and "narration" keys.
- Camera: MUST state angle (eye-level/overhead/low-angle/45-degree) and movement (stationary OR slow dolly/pan).
- Veo 3 REJECTS: text overlays, graphics, animation, fantasy, zooms, whip pans, rack focus, handheld, tracking shots, crane, drone.
- Single subject performing single continuous action (no "first X, then Y, finally Z").
- Light source MUST be named explicitly.
- Narration: exactly one sentence, maximum 18 words, adds insight beyond visuals.

MUTATION GUIDELINES:
- Make TARGETED changes to fix specific failures. Do not rewrite from scratch.
- If the prompt already has a strength (e.g., good structure), preserve it.
- Add explicit examples or counter-examples for common failure modes.
- Consider adding a brief checklist or verification step if failures are systematic.

Provide the complete new instruction within ``` blocks."""

    response = _llm_call_with_retry(
        model=MUTATION_MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=MUTATION_TEMPERATURE,
        max_tokens=2500,
        parse_json=False,
    )

    match = re.search(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response


# ---------------------------------------------------------------------------
# System-Aware Crossover / Merge (GEPA Algorithm 3-4)
# ---------------------------------------------------------------------------
def find_merge_candidates(candidates: List[Candidate], min_generation: int = 2) -> List[tuple]:
    """
    Find pairs of candidates suitable for merging (Algorithm 3: Desirable).
    
    Criteria from paper:
    - Both candidates should have evolved (generation >= min_generation)
    - They should have different strengths (complementary on different tasks)
    - They should not be direct parent-child
    """
    merge_pairs = []
    
    for i, c1 in enumerate(candidates):
        if c1.generation < min_generation:
            continue
        for c2 in candidates[i+1:]:
            if c2.generation < min_generation:
                continue
            # Skip if direct parent-child relationship
            if c1.parent_id == c2.id or c2.parent_id == c1.id:
                continue
            
            # Check for complementary strengths: each is best on different tasks
            c1_best_tasks = set()
            c2_best_tasks = set()
            
            common_tasks = set(c1.scores.keys()) & set(c2.scores.keys())
            for task in common_tasks:
                if c1.scores.get(task, 0) > c2.scores.get(task, 0):
                    c1_best_tasks.add(task)
                elif c2.scores.get(task, 0) > c1.scores.get(task, 0):
                    c2_best_tasks.add(task)
            
            # Both should have some tasks where they excel
            if c1_best_tasks and c2_best_tasks:
                complementarity = len(c1_best_tasks) + len(c2_best_tasks)
                merge_pairs.append((c1, c2, complementarity))
    
    # Sort by complementarity (higher is better)
    merge_pairs.sort(key=lambda x: -x[2])
    return merge_pairs


def merge_candidates(c1: Candidate, c2: Candidate, state: "RunState") -> Candidate:
    """
    Merge two candidates by combining their instruction strategies (Algorithm 4).
    
    The paper describes module-level merging, but since we have a single prompt,
    we use LLM to intelligently combine the best aspects of both prompts.
    """
    # Identify which tasks each excels at
    c1_strengths = []
    c2_strengths = []
    
    common_tasks = set(c1.scores.keys()) & set(c2.scores.keys())
    for task in common_tasks:
        s1, s2 = c1.scores.get(task, 0), c2.scores.get(task, 0)
        if s1 > s2:
            c1_strengths.append(f"{task}: {s1:.0f} vs {s2:.0f}")
        elif s2 > s1:
            c2_strengths.append(f"{task}: {s2:.0f} vs {s1:.0f}")
    
    merge_prompt = f"""You are merging two evolved prompt strategies that each excel at different tasks.

=== PROMPT A (excels at: {', '.join(c1_strengths[:3]) or 'general'}) ===
```
{c1.prompt}
```

=== PROMPT B (excels at: {', '.join(c2_strengths[:3]) or 'general'}) ===
```
{c2.prompt}
```

Your task: Create a MERGED prompt that combines the best strategies from both.

GUIDELINES:
1. Identify what makes each prompt successful on its strong tasks.
2. Combine complementary strategies (e.g., if A has better structure and B has better examples, use both).
3. Resolve conflicts by keeping the more specific/detailed instruction.
4. Preserve all critical constraints (camera rules, narration limits, JSON format).
5. The merged prompt should be coherent, not a simple concatenation.

DOMAIN REQUIREMENTS (must all be present):
- 8-second photorealistic video, no cuts
- Single subject, single action
- Camera angle stated + movement (stationary or slow dolly/pan)
- Light source named
- JSON output with "script" and "narration" keys
- Narration: 1 sentence, ≤18 words

Output the merged prompt within ``` blocks."""

    response = _llm_call_with_retry(
        model=MUTATION_MODEL,
        messages=[{"role": "user", "content": merge_prompt}],
        temperature=MUTATION_TEMPERATURE,
        max_tokens=2500,
        parse_json=False,
    )

    match = re.search(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    merged_prompt = match.group(1).strip() if match else response
    
    # Create new candidate
    child = Candidate(
        id=state.new_id(),
        prompt=merged_prompt,
        parent_id=c1.id,  # Primary parent
        generation=max(c1.generation, c2.generation) + 1,
        mutation_summary=f"Merged C{c1.id}+C{c2.id}",
    )
    
    return child


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
    try:
        return _llm_call_with_retry(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": f"In one sentence, what changed between these two prompts?\n\nOLD:\n{old_prompt[:1500]}\n\nNEW:\n{new_prompt[:1500]}"}],
            temperature=INFERENCE_TEMPERATURE,
            max_tokens=100,
            parse_json=False,
        )
    except Exception:
        return "Mutation applied"


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
    merges_performed = 0
    max_merges = 5  # Paper suggests up to 5 merges per run
    
    while state.rollouts_used < budget * len(tasks):
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration} | rollouts={state.rollouts_used} | budget={budget * len(tasks)}")
        print(f"{'='*70}")

        # --- MERGE CHECK (GEPA Algorithm 3-4) ---
        # Attempt merge every MERGE_INTERVAL iterations if we have enough evolved candidates
        if iteration % MERGE_INTERVAL == 0 and merges_performed < max_merges:
            merge_pairs = find_merge_candidates(state.candidates, min_generation=1)
            if merge_pairs:
                c1, c2, complementarity = merge_pairs[0]
                print(f"MERGE: Combining C{c1.id} (avg={c1.avg_score:.1f}) + C{c2.id} (avg={c2.avg_score:.1f})")
                print(f"  Complementarity score: {complementarity}")
                
                merged_child = merge_candidates(c1, c2, state)
                
                # Evaluate merged candidate on full task set
                print(f"  Evaluating merged C{merged_child.id}...")
                merged_traces = evaluate_candidate(merged_child, tasks)
                state.traces.extend(merged_traces)
                state.rollouts_used += len(merged_traces)
                
                for t in merged_traces:
                    merged_child.scores[t.task_key] = t.score
                merged_child.avg_score = sum(merged_child.scores.values()) / len(merged_child.scores) if merged_child.scores else 0
                
                # Accept if it's competitive
                parent_avg = (c1.avg_score + c2.avg_score) / 2
                if merged_child.avg_score >= parent_avg * 0.95:  # Accept if within 5% of parent average
                    state.candidates.append(merged_child)
                    merges_performed += 1
                    print_candidate_summary(merged_child, prefix="    MERGE ACCEPTED: ")
                    
                    if merged_child.avg_score > best.avg_score:
                        best = merged_child
                        state.best_candidate_id = best.id
                        print(f"    *** NEW BEST FROM MERGE: C{best.id} (avg={best.avg_score:.1f}) ***")
                else:
                    print(f"    MERGE REJECTED: {merged_child.avg_score:.1f} < {parent_avg * 0.95:.1f}")
                
                save_state(state, save_path)
                continue  # Skip regular mutation this iteration

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
            print("  WARNING: Zero score, but continuing search...")
            # Don't early stop - try to recover with more iterations
            state.traces.extend(mb_traces)
            save_state(state, save_path)
            continue

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
