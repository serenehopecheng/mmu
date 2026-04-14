#!/usr/bin/env python3
"""
GEPA-lite: Genetic Evolution of Prompts with Acceptance

This is a simplified wrapper around the gepa_lite package.
For full configuration options, use: python -m gepa_lite --help
"""

from dotenv import load_dotenv

load_dotenv()

from gepa_lite import Config, run_gepa_lite


def main() -> None:
    """Run GEPA-lite with default configuration."""
    config = Config(
        # Model configuration
        generation_model="gpt-5-nano",
        evaluation_model="gpt-5-nano",
        reflection_model="gpt-5-nano",

        # Search parameters
        max_iters=12,
        feedback_minibatch_size=2,
        accept_margin=0.15,
        random_seed=42,

        # Early stopping (new feature)
        early_stop_enabled=True,
        early_stop_patience=5,

        # Diversity tracking (new feature)
        diversity_enabled=True,
        diversity_threshold=0.3,

        # Parallelization (new feature)
        parallel_evaluation=True,
        max_workers=8,

        # Output
        print_prompt_each_iteration=True,
        prompt_print_char_limit=4000,

        # Logging
        log_level="INFO",
        log_to_file=True,
    )

    try:
        best = run_gepa_lite(config)
        print(f"\nFinal best candidate: C{best.cid}")
        print(f"Final best score: {best.pareto_avg:.2f}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
