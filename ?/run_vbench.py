#!/usr/bin/env python3
"""
Create VBench prompt files for deep vs naive video folders.
Maps video names to prompts from prompts.json.
"""

import json
from pathlib import Path

# Load prompts
with open("prompts.json", "r") as f:
    prompts_data = json.load(f)

# Create prompt files for both folders
base_dir = Path("/Users/serenecheng/mmu/evaluation")

def create_prompt_file(folder_name: str) -> str:
    """Create a VBench-compatible prompt file for a folder."""
    folder_path = base_dir / folder_name
    prompt_dict = {}
    
    for video_name, prompt in prompts_data.items():
        video_path = folder_path / video_name
        if video_path.exists():
            prompt_dict[str(video_path)] = prompt
    
    output_path = base_dir / f"prompts_{folder_name}.json"
    with open(output_path, "w") as f:
        json.dump(prompt_dict, f, indent=2)
    
    print(f"Created {output_path} with {len(prompt_dict)} videos")
    return str(output_path)

# Create prompt files
deep_prompts = create_prompt_file("deep")
naive_prompts = create_prompt_file("naive")

# VBench dimensions that support custom input
CUSTOM_DIMENSIONS = [
    'subject_consistency',
    'background_consistency', 
    'motion_smoothness',
    'dynamic_degree',
    'aesthetic_quality',
    'imaging_quality',
]

print(f"\nPrompt files created:")
print(f"  Deep: {deep_prompts}")
print(f"  Naive: {naive_prompts}")
print(f"\nSupported custom dimensions: {CUSTOM_DIMENSIONS}")
