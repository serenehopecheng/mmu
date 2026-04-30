#!/usr/bin/env python3
"""
Evaluate videos using VBench for deep vs naive comparison.
"""

import sys
import os

# eva-decord installs as 'decord', so no monkey-patching needed

import json
import torch
from pathlib import Path
from datetime import datetime

# Now import VBench
from vbench import VBench

# Configuration
BASE_DIR = Path("/Users/serenecheng/mmu/evaluation")
OUTPUT_DIR = BASE_DIR / "vbench_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# VBench dimensions that support custom input
CUSTOM_DIMENSIONS = [
    'subject_consistency',
    'background_consistency', 
    'motion_smoothness',
    'dynamic_degree',
    'aesthetic_quality',
    'imaging_quality',
]

def load_prompts(folder_name: str) -> dict:
    """Load prompt file for a folder."""
    prompt_file = BASE_DIR / f"prompts_{folder_name}.json"
    with open(prompt_file, "r") as f:
        return json.load(f)

def evaluate_folder(folder_name: str, dimensions: list = None):
    """Run VBench evaluation on a folder."""
    if dimensions is None:
        dimensions = CUSTOM_DIMENSIONS
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {folder_name}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load prompts
    prompts = load_prompts(folder_name)
    print(f"Loaded {len(prompts)} video prompts")
    
    # Initialize VBench
    vbench = VBench(device, None, str(OUTPUT_DIR))
    
    videos_path = str(BASE_DIR / folder_name)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    
    # Run evaluation
    vbench.evaluate(
        videos_path=videos_path,
        name=f"{folder_name}_{timestamp}",
        prompt_list=prompts,
        dimension_list=dimensions,
        mode="custom_input",
    )
    
    print(f"\nCompleted evaluation for {folder_name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run VBench evaluation on deep vs naive videos")
    parser.add_argument("--folders", nargs="+", default=["deep", "naive"], 
                        help="Folders to evaluate (default: deep naive)")
    parser.add_argument("--dimensions", nargs="+", default=None,
                        help=f"Dimensions to evaluate (default: all custom dimensions)")
    args = parser.parse_args()
    
    print("VBench Evaluation: Deep vs Naive")
    print(f"Dimensions: {args.dimensions or CUSTOM_DIMENSIONS}")
    print(f"Folders: {args.folders}")
    
    for folder in args.folders:
        try:
            evaluate_folder(folder, args.dimensions)
        except Exception as e:
            print(f"Error evaluating {folder}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
