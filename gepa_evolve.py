"""
Usage:
    python gepa_evolve.py                           # defaults
    python gepa_evolve.py --budget 30               # custom budget
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

PROMPTS_FILE = "prompts.json"
EVAL_MODEL = "gpt-5-nano"
MUTATION_MODEL = "gpt-5-nano"
SCRIPT_GEN_MODEL = "gpt-5-nano"
MINIBATCH_SIZE = 3   
PARETO_SET_SIZE = None    
ACCEPT_TOLERANCE = 2.0  

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

SEED_PROMPT = """You are writing a prompt for Veo 3, a text-to-video AI model that generates photorealistic real-world footage.
                Your output will be fed directly to the model as its generation prompt.

                TOPIC: "{topic}"
                CONTEXT: {context}

                RULES — every output MUST satisfy all of these:
                1. SINGLE SCENE: Exactly 8 seconds of continuous, photorealistic footage. No cuts, transitions, or scene changes.
                2. ONE SUBJECT, ONE ACTION: Describe one clear subject performing one simple, observable action. No competing elements or multi-step processes.
                3. VISUAL SPECIFICITY: Name concrete materials, textures, colors, lighting source/quality, and spatial relationships. Replace vague adjectives with tangible details.
                4. TEMPORAL BEATS: Describe what the viewer sees at the start, what unfolds over the 8 seconds, and how the shot ends. Use phrases like "The shot opens with…", "Over the next few seconds…", "The clip ends as…".
                5. CAMERA: State the angle (eye-level, overhead, 45-degree, etc.) and movement (stationary, slow dolly, slow pan). No zooms, whip pans, handheld shake, rack focus, or scanning.
                6. LIGHTING: Specify the light source (soft window light, golden hour sun, overhead fluorescents, etc.).
                7. FORBIDDEN: No text overlays, graphics, animation, fantasy, abstract imagery, grid/pattern layouts, or technical measurement setups.
                8. NARRATION: One spoken sentence, MAXIMUM 18 words. Must add insight or context beyond what is visible — never just restate the visuals. Count your words.

                Output valid JSON only, no other text:
                {{
                    "script": "Present-tense, temporally ordered 8-second clip description with specific visual and cinematic details.",
                    "narration": "One sentence, max 18 words, adding insight beyond the visuals."
                }}"""

def generate_script(prompt_template: str, topic: str, context: str = "None provided.") -> Dict[str, str]:
    """Run the script generator with a given system prompt template."""
    filled = prompt_template.replace("{topic}", topic).replace("{context}", context)

    r = client.chat.completions.create(
        model=SCRIPT_GEN_MODEL,
        messages=[{"role": "user", "content": filled}],
    )
    choice = r.choices[0]
    raw = choice.message.content or ""
    raw = raw.strip()
    if raw:
        output_dir = Path("runs") / "generate_script"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{topic[:20]}_{prompt_template[:20]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== SYSTEM PROMPT ===\n{prompt_template}\n\n")
            f.write(f"=== TOPIC ===\n{topic}\n\n")
            f.write("=== MODEL OUTPUT ===\n")
            f.write(raw)
    if not raw:
        raise ValueError(f"Empty response from {SCRIPT_GEN_MODEL} (finish_reason={choice.finish_reason})")
    try:
        return json.loads(_clean_json(raw))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    [generate_script] JSON parse failed: {e}")
        print(f"    [generate_script] finish_reason={choice.finish_reason}")
        print(f"    [generate_script] Raw ({len(raw)} chars): {raw[:300]}")
        raise

def critique_script(script_data: Dict[str, str], topic: str) -> Dict[str, Any]:
    """Run the critique evaluator. Returns structured feedback + numeric score."""
    critique_prompt = f"""You are a strict evaluator for Veo 3 video generation prompts.

    TOPIC: {topic}
    VIDEO SCRIPT: {script_data.get("script", "")}
    NARRATION: {script_data.get("narration", "")}

    Score this prompt on each dimension (0-10) and list any issues found.

    SCORING DIMENSIONS:
    1. policy_compliance: No sensitive content, text overlays, graphics, animation, fantasy, or prohibited camera moves.
    2. visual_specificity: Concrete materials, textures, colors, lighting, spatial relationships named explicitly.
    3. temporal_clarity: Clear start-to-end progression across 8 seconds.
    4. single_subject_focus: One subject, one action, no competing elements.
    5. camera_feasibility: Camera angle and movement are specified, realistic, and simple (stationary or slow pan/dolly).
    6. narration_quality: One sentence, max 18 words, adds insight beyond what is shown.
    7. veo_compatibility: Likely to succeed with Veo 3 without triggering policy rejection. Penalize abstract, multi-step, grid, scanning, technical setups.

    For each issue found, explain what part of the prompt caused it and what instruction change would prevent it.

    Respond with ONLY valid JSON:
    {{
        "scores": {{
            "policy_compliance": 0-10,
            "visual_specificity": 0-10,
            "temporal_clarity": 0-10,
            "single_subject_focus": 0-10,
            "camera_feasibility": 0-10,
            "narration_quality": 0-10,
            "veo_compatibility": 0-10
        }},
        "issues": ["issue 1 description", "..."],
        "feedback": "Detailed paragraph: what went wrong, what instruction in the system prompt was insufficient, and what change would fix it."
    }}"""

    r = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": critique_prompt}],
    )
    choice = r.choices[0]
    raw = choice.message.content or ""
    raw = raw.strip()
    if raw:
        output_dir = Path("runs") / "critique_script"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{topic[:20]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== TOPIC ===\n{topic}\n\n")
            f.write("=== SCRIPT DATA ===\n")
            f.write(f"{json.dumps(script_data, indent=2)}\n\n")
            f.write("=== CRITIQUE PROMPT ===\n{json.dumps(critique_prompt, indent=2)}\n\n")
            f.write("=== MODEL OUTPUT ===\n")
            f.write(raw)
    if not raw:
        raise ValueError(f"Empty response from {EVAL_MODEL} (finish_reason={choice.finish_reason})")
    try:
        return json.loads(_clean_json(raw))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    [critique_script] JSON parse failed: {e}")
        print(f"    [critique_script] finish_reason={choice.finish_reason}")
        print(f"    [critique_script] Raw ({len(raw)} chars): {raw[:300]}")
        raise

def compute_score(critique: Dict[str, Any]) -> float:
    """Aggregate critique sub-scores into a single 0-100 fitness value."""
    scores = critique.get("scores", {})
    if not scores:
        return 0.0
    weights = {
        "policy_compliance": 2.0,
        "veo_compatibility": 2.0,
        "visual_specificity": 1.0,
        "temporal_clarity": 1.0,
        "single_subject_focus": 1.5,
        "camera_feasibility": 1.0,
        "narration_quality": 0.5,
    }
    total_w = sum(weights.get(k, 1.0) for k in scores)
    weighted = sum(scores.get(k, 0) * weights.get(k, 1.0) for k in scores)
    return round((weighted / total_w) * 10, 2)  # scale to 0-100

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
            critique = critique_script(script_data, topic)
        except Exception as e:
            print(f"    [eval C{candidate.id} / {key}] critique failed: {e}")
            critique = {"scores": {}, "issues": [str(e)], "feedback": str(e)}

        score = compute_score(critique)
        feedback = critique.get("feedback", "")
        issues = critique.get("issues", [])
        if issues:
            feedback = "ISSUES:\n" + "\n".join(f"- {i}" for i in issues) + "\n\nFEEDBACK:\n" + feedback

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
        filename = Path("runs") / "traces" / f"{key}_{topic}_{candidate.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== TASK KEY ===\n{key}\n\n")
            f.write(f"=== TASK TOPIC ===\n{topic}\n\n")
            f.write(f"=== CANDIDATE ID ===\n{candidate.id}\n\n")
            f.write(f"=== SCRIPT OUTPUT ===\n{json.dumps(script_data, indent=2)}\n\n")
            f.write(f"=== CRITIQUE ===\n{json.dumps(critique, indent=2)}\n\n")
            f.write(f"=== SCORE ===\n{score}\n\n")
            f.write(f"=== FEEDBACK TEXT ===\n{feedback}\n\n")

    return traces

def reflective_mutate(candidate: Candidate, traces: List[EvalTrace]) -> str:
    """
    Use the GEPA reflection meta-prompt to propose an improved system prompt
    based on the candidate's current prompt and its evaluation traces.
    """
    examples_block = ""
    all_dim_scores: Dict[str, List[float]] = {}
    for t in traces:
        dim_scores = t.critique.get("scores", {})
        dim_lines = ""
        for dim, val in sorted(dim_scores.items()):
            dim_lines += f"      {dim}: {val}/10\n"
            all_dim_scores.setdefault(dim, []).append(val)
        examples_block += f"""
        ---
        Task topic: {t.topic}
        Assistant output:
        script: {t.script_output.get("script", "N/A")}
        narration: {t.script_output.get("narration", "N/A")}
        Per-dimension scores:
        {dim_lines}Overall score: {t.score}/100
        Feedback: {t.feedback_text}
        ---
        """

    # Identify weakest dimensions to focus on
    dim_avgs = {dim: sum(vals) / len(vals) for dim, vals in all_dim_scores.items() if vals}
    sorted_dims = sorted(dim_avgs.items(), key=lambda x: x[1])
    weakest = [f"  - {dim}: avg {avg:.1f}/10" for dim, avg in sorted_dims[:3]]
    weakest_block = "\n".join(weakest)

    meta_prompt = f"""I provided an assistant with the following instructions to perform a task for me:
    ```
    {candidate.prompt}
    ```

    The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
    ```
    {examples_block}
    ```

    WEAKEST SCORING DIMENSIONS (focus your improvements here):
    {weakest_block}

    Your task is to write an improved instruction for the assistant. Follow these guidelines:

    1. STRUCTURE: The instruction MUST contain explicit numbered rules that the assistant can check against. Do not write vague prose — write concrete, testable constraints.

    2. FOCUS ON WEAK DIMENSIONS: The scores above reveal which dimensions are consistently low. Add or strengthen rules that directly address those weaknesses. For example:
    - Low narration_quality → add a rule like "Narration must be one sentence, max 18 words. It must teach or contextualize, never restate visuals. Count your words before outputting."
    - Low visual_specificity → add a rule requiring explicit materials, textures, colors, and lighting source in every script.
    - Low temporal_clarity → add a rule requiring temporal beat markers ("The shot opens with…", "Over the next few seconds…", "The clip ends as…").
    - Low camera_feasibility → add a rule stating the camera angle and movement type explicitly.

    3. PRESERVE WHAT WORKS: Keep rules and language from the current instruction that score well. Only modify or add rules targeting the weak dimensions.

    4. INCLUDE A WORKED EXAMPLE: After the rules, include one brief example of a good output so the assistant has a concrete reference for quality.

    5. DOMAIN KNOWLEDGE: The assistant writes prompts for Veo 3, a text-to-video AI that generates photorealistic 8-second real-world footage. Keep these hard constraints:
    - Output must be valid JSON with "script" and "narration" keys only.
    - Veo 3 rejects: text overlays, graphics, animation, fantasy, zooms, whip pans, rack focus, scanning motions, grid layouts, multi-step processes, abstract imagery, multiple competing subjects.
    - Camera must be stationary or slow dolly/pan. Angle must be specified (eye-level, overhead, 45-degree, etc.).
    - The clip is exactly 8 seconds of continuous footage with no cuts.
    - Narration is one sentence, max 18 words, adding insight beyond the visuals.

    Provide the new instructions within ``` blocks."""

    r = client.chat.completions.create(
        model=MUTATION_MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
    )
    response = (r.choices[0].message.content or "").strip()
    if not response:
        raise ValueError(f"Empty mutation response from {MUTATION_MODEL} (finish_reason={r.choices[0].finish_reason})")

    match = re.search(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response

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

def load_tasks(path: str = PROMPTS_FILE) -> List[Dict[str, str]]:
    with open(path) as f:
        raw = json.load(f)
    tasks = []
    for key, topic in raw.items():
        tasks.append({"key": key, "topic": topic})
    return tasks

def sample_minibatch(tasks: List[Dict[str, str]], size: int) -> List[Dict[str, str]]:
    return random.sample(tasks, min(size, len(tasks)))

def _clean_json(s: str) -> str:
    """Extract a JSON object from model output, tolerating markdown fences and prose."""
    s = s.strip()
    if not s:
        raise ValueError("Model returned empty response")

    # Strip markdown fences
    s = re.sub(r"^```(?:json)?\s*\n?", "", s)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    s = s.strip()

    # If the result already looks like JSON, return it
    if s.startswith("{"):
        return s

    # Try to extract the first JSON object from surrounding prose
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError(f"No JSON object found in model response: {s[:200]}")

def print_candidate_summary(c: Candidate, prefix: str = ""):
    task_scores = ", ".join(f"{k}: {v}" for k, v in sorted(c.scores.items())[:5])
    print(f"{prefix}[C{c.id}] gen={c.generation} avg={c.avg_score:.1f} parent=C{c.parent_id} | {task_scores}...")

def summarize_mutation(old_prompt: str, new_prompt: str) -> str:
    """Ask the LLM for a one-line diff summary."""
    r = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": f"In one sentence, what changed between these two prompts?\n\nOLD:\n{old_prompt[:1500]}\n\nNEW:\n{new_prompt[:1500]}"}],
    )
    return (r.choices[0].message.content or "(empty response)").strip()

def run_gepa(
    budget: int = 2,
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

        print("Initializing with bare-minimum seed prompt (generation 0)...")
        cid = state.new_id()
        seed = Candidate(id=cid, prompt=SEED_PROMPT, generation=0)
        state.candidates.append(seed)

        print(f"Evaluating seed C{seed.id} on {len(tasks)} tasks...")
        traces = evaluate_candidate(seed, tasks)
        state.traces.extend(traces)
        for t in traces:
            seed.scores[t.task_key] = t.score
        seed.avg_score = sum(seed.scores.values()) / len(seed.scores) if seed.scores else 0
        state.rollouts_used += len(traces)
        print_candidate_summary(seed, prefix="    ")

    best = max(state.candidates, key=lambda c: c.avg_score)
    state.best_candidate_id = best.id
    print(f"\nBest after init: C{best.id} (avg={best.avg_score:.1f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(output_dir) / f"gepa_{timestamp}.json"
    save_state(state, save_path)

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
        print(f"  Parent minibatch avg: {mb_avg:.1f}")
        for t in mb_traces:
            dims = t.critique.get("scores", {})
            dim_str = " ".join(f"{k[:3]}={v}" for k, v in sorted(dims.items()))
            print(f"    {t.task_key}: {t.score:.1f} | {dim_str}")

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
        print(f"  Child minibatch avg: {child_mb_avg:.1f} (parent was {mb_avg:.1f})")
        for t in child_mb_traces:
            dims = t.critique.get("scores", {})
            dim_str = " ".join(f"{k[:3]}={v}" for k, v in sorted(dims.items()))
            print(f"    {t.task_key}: {t.score:.1f} | {dim_str}")

        # 5. Accept if improved or within tolerance (allows lateral exploration)
        delta = child_mb_avg - mb_avg
        if delta >= -ACCEPT_TOLERANCE:
            label = f"IMPROVED +{delta:.1f}" if delta > 0 else f"LATERAL {delta:+.1f} (within tolerance)"
            print(f"  {label} — evaluating on full Pareto set...")

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
            print(f"  REJECTED (delta={delta:+.1f}, below tolerance={-ACCEPT_TOLERANCE:.1f})")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEPA prompt evolution for Veo 3 script generator")
    parser.add_argument("--budget", type=int, default=40,
                        help="Max iterations of the evolution loop (each uses ~2 minibatch + 1 full eval)")
    parser.add_argument("--minibatch", type=int, default=MINIBATCH_SIZE,
                        help="Tasks per minibatch evaluation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a prior run JSON to resume from")
    parser.add_argument("--output", type=str, default="runs",
                        help="Output directory for run state files")
    args = parser.parse_args()

    best = run_gepa(
        budget=args.budget,
        minibatch_size=args.minibatch,
        resume_path=args.resume,
        output_dir=args.output,
    )
