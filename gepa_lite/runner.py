"""Main GEPA-lite search loop."""

import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gepa_lite.cache import EvalCache
from gepa_lite.config import Config
from gepa_lite.evaluation import average_score, Evaluator
from gepa_lite.llm import LLMClient
from gepa_lite.logging_utils import (
    MetricsTracker,
    prompt_for_console,
    setup_logging,
    TraceLogger,
)
from gepa_lite.models import Candidate, TopicEval
from gepa_lite.mutation import (
    EarlyStopTracker,
    Mutator,
    sample_parent_from_frontier,
)


def choose_feedback_topics(
    task_items: List[Tuple[str, str]],
    size: int,
) -> List[Tuple[str, str]]:
    """Select a random minibatch of topics for feedback."""
    if size >= len(task_items):
        return task_items[:]
    return random.sample(task_items, size)


def write_run_artifacts(
    config: Config,
    candidates: List[Candidate],
    best_candidate: Candidate,
    metrics: Optional[MetricsTracker] = None,
) -> None:
    """Write output files at the end of a run."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    (config.output_dir / "best_prompt.txt").write_text(
        best_candidate.prompt,
        encoding="utf-8",
    )

    history = [c.to_history_dict() for c in candidates]
    (config.output_dir / "search_history.json").write_text(
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
    (config.output_dir / "detailed_evaluations.json").write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if metrics:
        (config.output_dir / "run_metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (config.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_gold_standards(config: Config) -> Dict[str, str]:
    """Load gold standard prompts from scripts.json."""
    gold_path = config.gold_standard_path
    if gold_path.exists():
        return json.loads(gold_path.read_text(encoding="utf-8"))
    return {}


def run_gepa_lite(
    config: Optional[Config] = None,
    llm_client: Optional[LLMClient] = None,
) -> Candidate:
    """Run the GEPA-lite evolutionary prompt search.

    Args:
        config: Configuration options. Uses defaults if not provided.
        llm_client: LLM client instance. Creates new one if not provided.

    Returns:
        The best candidate found during the search.
    """
    config = config or Config()
    random.seed(config.random_seed)

    logger = setup_logging(config)
    metrics = MetricsTracker()
    metrics.start()

    llm_client = llm_client or LLMClient(config=config)
    cache = EvalCache(config)
    
    gold_standards = load_gold_standards(config)
    evaluator = Evaluator(config, llm_client, cache, gold_standards=gold_standards)
    mutator = Mutator(config, llm_client)
    early_stop = EarlyStopTracker(config)

    task_items = list(config.tasks.items())
    task_ids = [task_id for task_id, _ in task_items]

    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running GEPA-lite for {config.max_iters} iterations on {len(task_items)} tasks...")

    with TraceLogger(config) as trace:
        logger.info(f"Trace log: {trace.path}")
        trace.log_run_start(config, len(task_items))

        # Seed candidate evaluated on full Pareto set
        seed_results = evaluator.evaluate_on_topics(config.seed_prompt, task_items)
        seed_candidate = Candidate(
            cid=0,
            prompt=config.seed_prompt,
            parent_id=None,
            topic_results=seed_results,
            pareto_avg=average_score(seed_results),
        )

        candidates: List[Candidate] = [seed_candidate]
        best_candidate = seed_candidate
        next_id = 1
        early_stopped = False

        existing_prompts = [seed_candidate.prompt]

        logger.info(f"Seed candidate average score: {seed_candidate.pareto_avg:.2f}")
        trace.log_seed_candidate(seed_candidate, seed_results)
        early_stop.update(seed_candidate.pareto_avg)

        for step in range(1, config.max_iters + 1):
            # Check early stopping
            if early_stop.should_stop():
                logger.info(
                    f"Early stopping triggered after {early_stop.iterations_without_improvement} "
                    f"iterations without improvement"
                )
                early_stopped = True
                break

            parent = sample_parent_from_frontier(candidates, task_ids)
            minibatch = choose_feedback_topics(task_items, config.feedback_minibatch_size)
            minibatch_ids = [task_id for task_id, _ in minibatch]

            logger.info("-" * 60)
            logger.info(f"Iteration {step}")
            logger.info(
                f"Selected parent: C{parent.cid} | full avg = {parent.pareto_avg:.2f} | "
                f"minibatch = {minibatch_ids}"
            )

            # Reuse parent scores on the minibatch from the cached full evaluation
            parent_minibatch_results = {
                task_id: parent.topic_results[task_id] for task_id in minibatch_ids
            }
            parent_minibatch_avg = average_score(parent_minibatch_results)

            child_prompt = mutator.reflective_mutate(parent.prompt, parent_minibatch_results)

            # Check diversity before evaluating
            if not mutator.is_sufficiently_diverse(child_prompt, existing_prompts):
                logger.info("Rejected child due to insufficient diversity")
                metrics.record_iteration(accepted=False)
                continue

            child_minibatch_results = evaluator.evaluate_on_topics(child_prompt, minibatch)
            child_minibatch_avg = average_score(child_minibatch_results)

            logger.info(f"Parent minibatch avg: {parent_minibatch_avg:.2f}")
            logger.info(f"Child  minibatch avg: {child_minibatch_avg:.2f}")

            if config.print_prompt_each_iteration:
                print("Proposed child prompt:")
                print(textwrap.indent(
                    prompt_for_console(child_prompt, config.prompt_print_char_limit),
                    "  ",
                ))

            trace.log_banner(f"Iteration {step}/{config.max_iters}")
            iter_header = trace.log_iteration_start(
                step=step,
                max_iters=config.max_iters,
                parent=parent,
                minibatch_ids=minibatch_ids,
                parent_minibatch_avg=parent_minibatch_avg,
                child_minibatch_avg=child_minibatch_avg,
                accept_margin=config.accept_margin,
                parent_prompt=parent.prompt,
                child_prompt=child_prompt,
                parent_results=parent_minibatch_results,
                child_results=child_minibatch_results,
            )

            if child_minibatch_avg <= parent_minibatch_avg + config.accept_margin:
                logger.info("Rejected child on minibatch gate.")
                trace.log_iteration_rejected(iter_header)
                metrics.record_iteration(accepted=False)
                continue

            # Accepted on minibatch. Reuse minibatch evaluations and only score missing tasks.
            full_child_results = evaluator.evaluate_on_topics_with_cache(
                child_prompt,
                task_items,
                cached_results=child_minibatch_results,
            )
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
            existing_prompts.append(child_prompt)

            logger.info(f"Accepted child C{child.cid} | full avg = {child.pareto_avg:.2f}")

            if config.print_prompt_each_iteration:
                print(f"Accepted prompt C{child.cid}:")
                print(textwrap.indent(
                    prompt_for_console(child.prompt, config.prompt_print_char_limit),
                    "  ",
                ))

            trace.log_iteration_accepted(iter_header, child, full_child_results)
            metrics.record_iteration(accepted=True)

            early_stop.update(child.pareto_avg)

            if child.pareto_avg > best_candidate.pareto_avg:
                best_candidate = child
                logger.info(f"New best candidate: C{child.cid}")

        logger.info("=" * 60)
        logger.info(f"Search complete. Best candidate: C{best_candidate.cid}")
        logger.info(f"Best average score: {best_candidate.pareto_avg:.2f}")

        metrics.end()
        metrics.update_from_llm_stats(llm_client.stats)
        metrics.update_from_cache_stats(cache.stats)

        write_run_artifacts(config, candidates, best_candidate, metrics)
        logger.info(f"Artifacts written to: {config.output_dir.resolve()}")

        trace.log_run_complete(best_candidate, len(candidates), early_stopped)

    return best_candidate
