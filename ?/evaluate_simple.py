#!/usr/bin/env python3
"""
Simplified VBench evaluation script for comparing deep vs naive videos.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import traceback

# Add VBench to path
VBENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VBench')
sys.path.insert(0, VBENCH_PATH)

import cv2

# Create mock decord module before any VBench imports
class MockVideoReader:
    def __init__(self, video_path, width=None, height=None, num_threads=1):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(video_path)
        self.vlen = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def __len__(self):
        return self.vlen
    
    def get_avg_fps(self):
        return self.fps
    
    def get_batch(self, indices):
        import torch
        frames = []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.width and self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
        
        class FrameArray:
            def __init__(self, data):
                self._data = np.array(data)
            def asnumpy(self):
                return self._data
            def permute(self, *args):
                return torch.from_numpy(self._data).permute(*args)
        
        return FrameArray(frames)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

class MockBridge:
    @staticmethod
    def set_bridge(name):
        pass

class MockDecord:
    VideoReader = MockVideoReader
    bridge = MockBridge
    
    @staticmethod
    def cpu(idx=0):
        return idx

sys.modules['decord'] = MockDecord()

import torch
from datetime import datetime

def load_prompts(prompts_path):
    with open(prompts_path, 'r') as f:
        return json.load(f)

def evaluate_single_dimension(videos_path, dimension, prompts_dict, output_dir):
    """Evaluate videos on a specific dimension."""
    print(f"Importing VBench...")
    from vbench import VBench
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    full_info_path = os.path.join(VBENCH_PATH, 'vbench', 'VBench_full_info.json')
    
    print(f"Creating VBench instance...")
    my_VBench = VBench(device, full_info_path, output_dir)
    
    current_time = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    name = f'{os.path.basename(videos_path)}_{dimension}_{current_time}'
    
    # Build video-prompt mapping
    video_prompt_map = {}
    for video_file in os.listdir(videos_path):
        if video_file.endswith('.mp4'):
            full_path = os.path.join(videos_path, video_file)
            if video_file in prompts_dict:
                video_prompt_map[full_path] = prompts_dict[video_file]
    
    print(f"Evaluating {len(video_prompt_map)} videos for dimension: {dimension}")
    print(f"Videos: {list(video_prompt_map.keys())}")
    
    my_VBench.evaluate(
        videos_path=videos_path,
        name=name,
        prompt_list=video_prompt_map,
        dimension_list=[dimension],
        mode='custom_input'
    )
    
    # Find and return results
    results_pattern = os.path.join(output_dir, f'*{dimension}*.json')
    import glob
    result_files = sorted(glob.glob(results_pattern), key=os.path.getmtime, reverse=True)
    
    if result_files:
        with open(result_files[0], 'r') as f:
            return json.load(f)
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=str, required=True,
                       choices=['motion_smoothness', 'aesthetic_quality', 'human_action', 'subject_consistency'])
    parser.add_argument('--video_set', type=str, required=True, choices=['deep', 'naive', 'both'])
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    deep_dir = os.path.join(base_dir, 'deep')
    naive_dir = os.path.join(base_dir, 'naive')
    prompts_path = os.path.join(base_dir, 'prompts.json')
    output_dir = os.path.join(base_dir, 'evaluation_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")
    
    results = {}
    
    if args.video_set in ['deep', 'both']:
        print(f"\n{'='*60}")
        print(f"Evaluating DEEP videos for {args.dimension}")
        print('='*60)
        try:
            deep_results = evaluate_single_dimension(deep_dir, args.dimension, prompts, output_dir)
            results['deep'] = deep_results
            print(f"Deep results: {deep_results}")
        except Exception as e:
            print(f"Error evaluating deep videos: {e}")
            traceback.print_exc()
    
    if args.video_set in ['naive', 'both']:
        print(f"\n{'='*60}")
        print(f"Evaluating NAIVE videos for {args.dimension}")
        print('='*60)
        try:
            naive_results = evaluate_single_dimension(naive_dir, args.dimension, prompts, output_dir)
            results['naive'] = naive_results
            print(f"Naive results: {naive_results}")
        except Exception as e:
            print(f"Error evaluating naive videos: {e}")
            traceback.print_exc()
    
    # Save results
    results_path = os.path.join(output_dir, f'{args.dimension}_{args.video_set}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
