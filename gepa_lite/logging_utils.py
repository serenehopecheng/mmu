"""Structured logging utilities for GEPA-lite."""

import json
import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

from gepa_lite.config import Config
from gepa_lite.models import Candidate, TopicEval


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{color}[{timestamp}] {record.levelname:<8}{reset} {record.getMessage()}"


def setup_logging(config: Config) -> logging.Logger:
    """Configure logging based on config settings."""
    logger = logging.getLogger("gepa_lite")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    if config.log_to_file:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = config.output_dir / "gepa_lite.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


class TraceLogger:
    """Handles detailed run tracing to file."""

    def __init__(self, config: Config):
        self.config = config
        self._file: Optional[IO] = None
        self._path: Optional[Path] = None

    def __enter__(self) -> "TraceLogger":
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.config.output_dir / self.config.trace_filename
        self._file = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._file:
            self._file.close()
            self._file = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def log(self, text: str) -> None:
        """Write text to trace file."""
        if self._file:
            self._file.write(text)
            self._file.flush()

    def log_banner(self, title: str) -> None:
        """Write a section banner."""
        line = "=" * 72
        self.log(f"\n{line}\n{title}\n{line}\n")

    def log_run_start(self, config: Config, num_tasks: int) -> None:
        """Log run initialization."""
        self.log_banner("GEPA-lite run")
        self.log(
            f"started_at: {datetime.now().isoformat(timespec='seconds')}\n"
            f"MAX_ITERS={config.max_iters}  "
            f"FEEDBACK_MINIBATCH_SIZE={config.feedback_minibatch_size}  "
            f"ACCEPT_MARGIN={config.accept_margin}  "
            f"RANDOM_SEED={config.random_seed}\n"
            f"early_stop_enabled={config.early_stop_enabled}  "
            f"early_stop_patience={config.early_stop_patience}\n"
            f"diversity_enabled={config.diversity_enabled}  "
            f"diversity_threshold={config.diversity_threshold}\n"
            f"parallel_evaluation={config.parallel_evaluation}  "
            f"max_workers={config.max_workers}\n"
            f"tasks: {num_tasks}\n"
            f"models: gen={config.generation_model} "
            f"eval={config.evaluation_model} "
            f"reflect={config.reflection_model}\n"
        )

    def log_seed_candidate(self, candidate: Candidate, results: Dict[str, TopicEval]) -> None:
        """Log seed candidate evaluation."""
        self.log_banner("Seed candidate C0 (full task evaluation)")
        self.log(
            f"pareto_avg (mean final_score): {candidate.pareto_avg:.2f}\n\n"
            f"SEED_PROMPT:\n"
            f"{textwrap.indent(candidate.prompt.rstrip(), '  ')}\n\n"
            f"{format_results_block(results)}\n"
        )

    def log_iteration_start(
        self,
        step: int,
        max_iters: int,
        parent: Candidate,
        minibatch_ids: List[str],
        parent_minibatch_avg: float,
        child_minibatch_avg: float,
        accept_margin: float,
        parent_prompt: str,
        child_prompt: str,
        parent_results: Dict[str, TopicEval],
        child_results: Dict[str, TopicEval],
    ) -> str:
        """Log iteration start and return header text for later use."""
        header = (
            f"\nparent: C{parent.cid}  parent_full_avg={parent.pareto_avg:.2f}\n"
            f"minibatch task_ids: {minibatch_ids}\n"
            f"parent_minibatch_avg: {parent_minibatch_avg:.2f}  "
            f"child_minibatch_avg: {child_minibatch_avg:.2f}  "
            f"gate: child > parent + {accept_margin}\n\n"
            f"Parent prompt (excerpt):\n"
            f"{textwrap.indent(parent_prompt[:1200] + ('...' if len(parent_prompt) > 1200 else ''), '  ')}\n\n"
            f"Proposed child prompt (excerpt):\n"
            f"{textwrap.indent(child_prompt[:1200] + ('...' if len(child_prompt) > 1200 else ''), '  ')}\n\n"
            f"--- Minibatch: parent topic results ---\n"
            f"{format_results_block(parent_results)}\n\n"
            f"--- Minibatch: child topic results ---\n"
            f"{format_results_block(child_results)}\n"
        )
        return header

    def log_iteration_rejected(self, header: str, reason: str = "minibatch gate") -> None:
        """Log rejected iteration."""
        self.log(header + f"RESULT: rejected ({reason})\n")

    def log_iteration_accepted(
        self,
        header: str,
        child: Candidate,
        full_results: Dict[str, TopicEval],
    ) -> None:
        """Log accepted iteration with full evaluation."""
        tail = (
            f"RESULT: accepted as C{child.cid}\n"
            f"full_child_avg: {child.pareto_avg:.2f}\n\n"
            f"--- Full task evaluation (accepted child) ---\n"
            f"{format_results_block(full_results)}\n"
        )
        self.log(header + tail)

    def log_run_complete(
        self,
        best_candidate: Candidate,
        total_candidates: int,
        early_stopped: bool = False,
    ) -> None:
        """Log run completion."""
        self.log_banner("Run complete")
        status = " (early stopped)" if early_stopped else ""
        self.log(
            f"finished_at: {datetime.now().isoformat(timespec='seconds')}{status}\n"
            f"best_candidate: C{best_candidate.cid}  "
            f"best_avg: {best_candidate.pareto_avg:.2f}\n"
            f"total_candidates: {total_candidates}\n"
            f"Also written: best_prompt.txt, search_history.json, detailed_evaluations.json\n"
        )


class MetricsTracker:
    """Tracks runtime metrics for observability."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Any] = {
            "iterations": 0,
            "accepted": 0,
            "rejected": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "start_time": None,
            "end_time": None,
        }

    def start(self) -> None:
        self._metrics["start_time"] = datetime.now()

    def end(self) -> None:
        self._metrics["end_time"] = datetime.now()

    def record_iteration(self, accepted: bool) -> None:
        self._metrics["iterations"] += 1
        if accepted:
            self._metrics["accepted"] += 1
        else:
            self._metrics["rejected"] += 1

    def update_from_llm_stats(self, stats: Dict[str, int]) -> None:
        self._metrics["api_calls"] = stats.get("api_calls", 0)
        self._metrics["cache_hits"] = stats.get("cache_hits", 0)

    def update_from_cache_stats(self, stats: Dict[str, Any]) -> None:
        self._metrics["cache_hits"] = stats.get("hits", 0)
        self._metrics["cache_hit_rate"] = stats.get("hit_rate", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        metrics = self._metrics.copy()
        if metrics["start_time"] and metrics["end_time"]:
            elapsed = (metrics["end_time"] - metrics["start_time"]).total_seconds()
            metrics["elapsed_seconds"] = elapsed
        metrics["start_time"] = (
            metrics["start_time"].isoformat() if metrics["start_time"] else None
        )
        metrics["end_time"] = (
            metrics["end_time"].isoformat() if metrics["end_time"] else None
        )
        return metrics


def format_topic_eval_block(te: TopicEval) -> str:
    """Format a single topic evaluation for display."""
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
    """Format multiple topic evaluations for display."""
    blocks = [format_topic_eval_block(te) for te in results.values()]
    return "\n\n---\n\n".join(blocks)


def prompt_for_console(prompt: str, limit: int = 4000) -> str:
    """Truncate prompt for console display."""
    if limit <= 0 or len(prompt) <= limit:
        return prompt
    return prompt[:limit] + "\n...[truncated for console output]..."
