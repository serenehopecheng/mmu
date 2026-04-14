"""CLI entry point for GEPA-lite."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from gepa_lite.config import Config
from gepa_lite.runner import run_gepa_lite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEPA-lite: Genetic Evolution of Prompts with Acceptance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--generation-model",
        default="gpt-5-nano",
        help="Model for generating video scripts",
    )
    parser.add_argument(
        "--evaluation-model",
        default="gpt-5-nano",
        help="Model for evaluating outputs",
    )
    parser.add_argument(
        "--reflection-model",
        default="gpt-5-nano",
        help="Model for reflective mutation",
    )

    # Search parameters
    parser.add_argument(
        "--max-iters",
        type=int,
        default=12,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=2,
        help="Number of topics per feedback minibatch",
    )
    parser.add_argument(
        "--accept-margin",
        type=float,
        default=0.15,
        help="Margin required for minibatch acceptance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Early stopping
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Iterations without improvement before early stopping",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping",
    )

    # Diversity
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.3,
        help="Minimum edit distance ratio for diversity",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable diversity checking",
    )

    # Parallelization
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers for evaluation",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel evaluation",
    )

    # Caching
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk caching",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gepa_lite_runs"),
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress prompt printing during iterations",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        help="Load configuration from JSON file",
    )

    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    # Load base config from file or defaults
    if args.config and args.config.exists():
        config = Config.from_file(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    config.generation_model = args.generation_model
    config.evaluation_model = args.evaluation_model
    config.reflection_model = args.reflection_model
    config.max_iters = args.max_iters
    config.feedback_minibatch_size = args.minibatch_size
    config.accept_margin = args.accept_margin
    config.random_seed = args.seed
    config.early_stop_patience = args.early_stop_patience
    config.early_stop_enabled = not args.no_early_stop
    config.diversity_threshold = args.diversity_threshold
    config.diversity_enabled = not args.no_diversity
    config.max_workers = args.max_workers
    config.parallel_evaluation = not args.no_parallel
    config.enable_disk_cache = not args.no_cache
    config.output_dir = args.output_dir
    config.print_prompt_each_iteration = not args.quiet
    config.log_level = args.log_level
    config.log_to_file = not args.no_log_file

    # Clear cache if requested
    if args.clear_cache:
        from gepa_lite.cache import EvalCache
        cache = EvalCache(config)
        cleared = cache.clear()
        print(f"Cleared {cleared} cached evaluations")

    try:
        best = run_gepa_lite(config)
        print(f"\nBest candidate: C{best.cid}")
        print(f"Best score: {best.pareto_avg:.2f}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
