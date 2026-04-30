import json
import os
import sys
from typing import Dict, List

import torch

# Prevent local files (e.g., ./clip.py) from shadowing third-party packages used by VBench.
_cwd = os.path.abspath(os.getcwd())
sys.path = [p for p in sys.path if os.path.abspath(p or _cwd) != _cwd]

# PyTorch 2.6 changed torch.load default weights_only=True; VBench checkpoints
# expect the previous behavior, so default to False when not provided.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from vbench import VBench


MODELS = ["baseline", "deep", "naive"]
CUSTOM_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def count_videos(folder: str) -> int:
    return len([f for f in os.listdir(folder) if f.lower().endswith(".mp4")])


def main() -> None:
    full_info_path = "VBench/vbench/VBench_full_info.json"
    output_path = "results/vbench_custom"
    os.makedirs(output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vbench = VBench(
        device=device,
        full_info_dir=full_info_path,
        output_path=output_path,
    )

    run_manifest: Dict[str, Dict[str, List[str] or int]] = {}

    for model_name in MODELS:
        videos_path = model_name
        if not os.path.isdir(videos_path):
            raise FileNotFoundError(f"Missing folder: {videos_path}")

        mp4_count = count_videos(videos_path)
        if mp4_count != 10:
            print(f"[WARN] {model_name} has {mp4_count} mp4 files (expected 10).")

        print(f"Evaluating {model_name} ({mp4_count} videos)...")
        vbench.evaluate(
            videos_path=videos_path,
            name=f"custom_{model_name}",
            dimension_list=CUSTOM_DIMENSIONS,
            mode="custom_input",
        )

        run_manifest[model_name] = {
            "videos_path": videos_path,
            "mp4_count": mp4_count,
            "dimensions": CUSTOM_DIMENSIONS,
        }

    manifest_path = os.path.join(output_path, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Done. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
