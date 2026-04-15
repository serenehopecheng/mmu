"""Configuration management for GEPA-lite."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import json
import os


@dataclass
class Config:
    """Central configuration for GEPA-lite runs."""

    # Model configuration
    generation_model: str = "gpt-5-nano"
    evaluation_model: str = "gpt-5-nano"
    reflection_model: str = "gpt-5-nano"

    # Search parameters
    max_iters: int = 12
    feedback_minibatch_size: int = 2
    accept_margin: float = 0.15
    random_seed: int = 42

    # Early stopping
    early_stop_patience: int = 5  # Stop after N iterations without improvement
    early_stop_enabled: bool = True

    # Diversity tracking
    diversity_threshold: float = 0.3  # Minimum edit distance ratio to accept similar prompts
    diversity_enabled: bool = True

    # Parallelization
    max_workers: int = 8
    parallel_evaluation: bool = True

    # Scoring
    hard_check_penalty: float = 1.2  # Penalty per failed hard check

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("gepa_lite_runs"))
    trace_filename: str = "gepa_lite_run_trace.txt"

    # Gold standard reference prompts (from scripts.json)
    gold_standard_path: Path = field(default_factory=lambda: Path("scripts.json"))
    reference_example_limit: int = 3

    # Caching
    enable_disk_cache: bool = True
    cache_schema_version: str = "topic_eval_v2"
    cache_dirname: str = "api_cache"

    # Console output
    print_prompt_each_iteration: bool = True
    prompt_print_char_limit: int = 4000

    # Rate limiting
    api_call_delay: float = 0.0  # Seconds between API calls (0 = no delay)
    max_retries: int = 3
    retry_base_delay: float = 1.0  # Base delay for exponential backoff

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    # Tasks (can be overridden)
    tasks: Dict[str, str] = field(default_factory=dict)

    # Seed prompt template
    seed_prompt: str = field(default_factory=lambda: (
        'Given the topic, generate a video script. '
        'Output valid JSON: {"script": "...", "narration": "..."}\n'
        'Topic: {topic}'
    ))

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.gold_standard_path, str):
            self.gold_standard_path = Path(self.gold_standard_path)

        if not self.tasks:
            self.tasks = DEFAULT_TASKS.copy()

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from a JSON or YAML file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Create config with environment variable overrides."""
        config = cls()

        env_mappings = {
            "GEPA_GENERATION_MODEL": "generation_model",
            "GEPA_EVALUATION_MODEL": "evaluation_model",
            "GEPA_REFLECTION_MODEL": "reflection_model",
            "GEPA_MAX_ITERS": ("max_iters", int),
            "GEPA_MINIBATCH_SIZE": ("feedback_minibatch_size", int),
            "GEPA_ACCEPT_MARGIN": ("accept_margin", float),
            "GEPA_RANDOM_SEED": ("random_seed", int),
            "GEPA_MAX_WORKERS": ("max_workers", int),
            "GEPA_LOG_LEVEL": "log_level",
        }

        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(mapping, tuple):
                    attr_name, type_fn = mapping
                    setattr(config, attr_name, type_fn(value))
                else:
                    setattr(config, mapping, value)

        return config

    def to_dict(self) -> Dict:
        """Serialize config to dictionary."""
        return {
            "generation_model": self.generation_model,
            "evaluation_model": self.evaluation_model,
            "reflection_model": self.reflection_model,
            "max_iters": self.max_iters,
            "feedback_minibatch_size": self.feedback_minibatch_size,
            "accept_margin": self.accept_margin,
            "random_seed": self.random_seed,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_enabled": self.early_stop_enabled,
            "diversity_threshold": self.diversity_threshold,
            "diversity_enabled": self.diversity_enabled,
            "max_workers": self.max_workers,
            "parallel_evaluation": self.parallel_evaluation,
            "hard_check_penalty": self.hard_check_penalty,
            "output_dir": str(self.output_dir),
            "trace_filename": self.trace_filename,
            "gold_standard_path": str(self.gold_standard_path),
            "reference_example_limit": self.reference_example_limit,
            "enable_disk_cache": self.enable_disk_cache,
            "cache_schema_version": self.cache_schema_version,
            "cache_dirname": self.cache_dirname,
            "print_prompt_each_iteration": self.print_prompt_each_iteration,
            "prompt_print_char_limit": self.prompt_print_char_limit,
            "api_call_delay": self.api_call_delay,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "seed_prompt": self.seed_prompt,
        }


DEFAULT_TASKS: Dict[str, str] = {
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
    "26.mp4": "An artisan practicing Kintsugi on a broken ceramic bowl, soft-lit tabletop scene.",
}
