# GEPA-lite

**Genetic Evolution of Prompts with Acceptance** - An evolutionary algorithm for optimizing LLM prompts.

## Features

- **Evolutionary search**: Uses Pareto frontier selection and reflective mutation to evolve prompts
- **Parallel evaluation**: Evaluates topics concurrently for faster iteration
- **Early stopping**: Automatically stops when no improvement is detected
- **Diversity tracking**: Prevents convergence to identical prompts
- **Disk caching**: Caches evaluations to avoid redundant API calls
- **Structured logging**: JSON logs for observability and debugging
- **CLI interface**: Full command-line control with sensible defaults
- **Type-safe**: Pydantic validation for LLM outputs

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

1. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

2. Run with defaults:

```bash
python -m gepa_lite
```

Or use the wrapper script:

```bash
python gepa_trial_gepa_lite.py
```

## CLI Options

```bash
python -m gepa_lite --help
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `--max-iters N` | Maximum iterations | 12 |
| `--minibatch-size N` | Topics per feedback batch | 2 |
| `--accept-margin F` | Acceptance threshold | 0.15 |
| `--max-workers N` | Parallel workers | 8 |
| `--no-early-stop` | Disable early stopping | enabled |
| `--no-diversity` | Disable diversity checks | enabled |
| `--no-cache` | Disable disk caching | enabled |
| `--quiet` | Suppress prompt printing | disabled |
| `--config FILE` | Load config from JSON | none |

## Configuration

### Environment Variables

- `GEPA_GENERATION_MODEL`: Model for generating scripts
- `GEPA_EVALUATION_MODEL`: Model for evaluation
- `GEPA_REFLECTION_MODEL`: Model for mutation
- `GEPA_MAX_ITERS`: Maximum iterations
- `GEPA_MINIBATCH_SIZE`: Minibatch size
- `GEPA_MAX_WORKERS`: Parallel workers
- `GEPA_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Config File

Create a JSON config file:

```json
{
  "generation_model": "gpt-5-nano",
  "max_iters": 20,
  "feedback_minibatch_size": 3,
  "early_stop_patience": 8,
  "parallel_evaluation": true,
  "max_workers": 16
}
```

Load with:

```bash
python -m gepa_lite --config my_config.json
```

## Programmatic Usage

```python
from gepa_lite import Config, run_gepa_lite

config = Config(
    max_iters=10,
    parallel_evaluation=True,
    early_stop_enabled=True,
)

best_candidate = run_gepa_lite(config)
print(f"Best score: {best_candidate.pareto_avg}")
print(f"Best prompt: {best_candidate.prompt}")
```

## Output Files

After a run, find these in `gepa_lite_runs/`:

- `best_prompt.txt`: The winning prompt
- `search_history.json`: All candidates and scores
- `detailed_evaluations.json`: Per-topic evaluation details
- `run_metrics.json`: Runtime statistics
- `config.json`: Configuration used
- `gepa_lite_run_trace.txt`: Detailed trace log
- `gepa_lite.log`: Structured JSON logs

## Testing

```bash
pytest gepa_lite/tests/ -v
```

With coverage:

```bash
pytest gepa_lite/tests/ --cov=gepa_lite --cov-report=term-missing
```

## Architecture

```
gepa_lite/
├── __init__.py        # Package exports
├── __main__.py        # CLI entry point
├── config.py          # Configuration management
├── llm.py             # LLM client with retries
├── models.py          # Pydantic data models
├── cache.py           # Disk caching
├── evaluation.py      # Scoring and evaluation
├── mutation.py        # Pareto selection + mutation
├── logging_utils.py   # Structured logging
├── runner.py          # Main search loop
└── tests/             # Test suite
```

## Algorithm Overview

1. **Initialize**: Evaluate seed prompt on all tasks
2. **Select parent**: Sample from Pareto frontier (weighted by task wins)
3. **Generate child**: Reflectively mutate based on minibatch feedback
4. **Gate check**: Accept only if child > parent + margin on minibatch
5. **Full evaluation**: Score accepted children on all tasks
6. **Update best**: Track best overall candidate
7. **Repeat**: Until max iterations or early stopping

## License

MIT
