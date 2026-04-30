#!/usr/bin/env python3
"""
Run VBench evaluation on deep vs naive folders.
"""
import sys
import traceback
import json
import os

print("Starting evaluation...", flush=True)

import torch
print(f"PyTorch version: {torch.__version__}", flush=True)

from vbench import VBench
print("VBench imported successfully", flush=True)

device = torch.device('cpu')
print(f'Device: {device}', flush=True)

with open('prompts_deep.json', 'r') as f:
    prompts_deep = json.load(f)
with open('prompts_naive.json', 'r') as f:
    prompts_naive = json.load(f)

print(f'Loaded prompts: deep={len(prompts_deep)}, naive={len(prompts_naive)}', flush=True)

# Create output directory
os.makedirs('vbench_results', exist_ok=True)

print('Initializing VBench...', flush=True)
vbench = VBench(device, None, 'vbench_results')
print('VBench initialized', flush=True)

# Dimensions to evaluate (those that support custom input)
dimensions = ['aesthetic_quality', 'imaging_quality']

for dim in dimensions:
    print(f'\n=== Evaluating dimension: {dim} ===', flush=True)
    
    try:
        # Evaluate deep folder
        print(f'Evaluating DEEP folder for {dim}...', flush=True)
        vbench.evaluate(
            videos_path='deep',
            name=f'deep_{dim}',
            prompt_list=prompts_deep,
            dimension_list=[dim],
            mode='custom_input',
        )
        print(f'DEEP {dim} evaluation complete!', flush=True)
    except Exception as e:
        print(f'Error evaluating DEEP {dim}: {e}', flush=True)
        traceback.print_exc()

    try:
        # Evaluate naive folder  
        print(f'Evaluating NAIVE folder for {dim}...', flush=True)
        vbench.evaluate(
            videos_path='naive',
            name=f'naive_{dim}',
            prompt_list=prompts_naive,
            dimension_list=[dim],
            mode='custom_input',
        )
        print(f'NAIVE {dim} evaluation complete!', flush=True)
    except Exception as e:
        print(f'Error evaluating NAIVE {dim}: {e}', flush=True)
        traceback.print_exc()

print('\n=== ALL DONE! ===', flush=True)
