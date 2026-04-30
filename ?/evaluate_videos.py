#!/usr/bin/env python3
"""
VBench evaluation script for comparing deep vs naive videos.
Evaluates: motion_smoothness, aesthetic_quality, human_action, subject_consistency
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add VBench to path
VBENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VBench')
sys.path.insert(0, VBENCH_PATH)

# Patch decord with opencv-based video reading before importing VBench
import cv2

def load_video_opencv(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    """Load video using OpenCV instead of decord."""
    import torch
    from PIL import Image, ImageSequence
    
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if width and height:
                frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        cap.release()
        buffer = np.array(frames).astype(np.uint8)
    else:
        raise NotImplementedError(f"Unsupported video format: {video_path}")
    
    frames = buffer
    
    # Sample frames if num_frames is specified
    if num_frames and len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = frames[indices]
    
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        import torch
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    
    return frames

def read_frames_opencv_by_fps(video_path, sample_fps=2, sample='rand', fix_start=None, 
                               max_num_frames=-1, trimmed30=False, num_frames=8):
    """Read video frames using OpenCV with FPS sampling."""
    import torch
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = vlen / float(fps) if fps > 0 else 0
    
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))
    
    # Calculate frame indices based on sampling strategy
    if "fps" in sample:
        output_fps = float(sample[3:])
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        # Uniform sampling
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = [(intervals[i], intervals[i + 1] - 1) for i in range(len(intervals) - 1)]
        if sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            frame_indices = [x[0] for x in ranges]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")
    
    frames = np.array(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
    return frames

# Create a mock decord module
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

# Create mock decord module
class MockDecord:
    VideoReader = MockVideoReader
    bridge = MockBridge
    
    @staticmethod
    def cpu(idx=0):
        return idx

# Install the mock
sys.modules['decord'] = MockDecord()

import torch
from datetime import datetime

def load_prompts(prompts_path):
    """Load prompts from JSON file."""
    with open(prompts_path, 'r') as f:
        return json.load(f)

def evaluate_dimension(videos_path, dimension, prompts_dict, output_dir):
    """Evaluate videos on a specific dimension."""
    from vbench import VBench
    
    # Patch the utils module
    import vbench.utils as vbench_utils
    vbench_utils.load_video = load_video_opencv
    vbench_utils.read_frames_decord_by_fps = read_frames_opencv_by_fps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    full_info_path = os.path.join(VBENCH_PATH, 'vbench', 'VBench_full_info.json')
    
    my_VBench = VBench(device, full_info_path, output_dir)
    
    current_time = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    name = f'{os.path.basename(videos_path)}_{dimension}_{current_time}'
    
    # Convert prompts_dict to the format VBench expects
    # VBench expects {"video_path": "prompt", ...}
    video_prompt_map = {}
    for video_file in os.listdir(videos_path):
        if video_file.endswith('.mp4'):
            full_path = os.path.join(videos_path, video_file)
            if video_file in prompts_dict:
                video_prompt_map[full_path] = prompts_dict[video_file]
    
    print(f"Evaluating {len(video_prompt_map)} videos for dimension: {dimension}")
    
    try:
        my_VBench.evaluate(
            videos_path=videos_path,
            name=name,
            prompt_list=video_prompt_map,
            dimension_list=[dimension],
            mode='custom_input'
        )
        
        # Load results
        results_file = os.path.join(output_dir, f'{name}_eval_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error evaluating {dimension}: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    deep_dir = os.path.join(base_dir, 'deep')
    naive_dir = os.path.join(base_dir, 'naive')
    prompts_path = os.path.join(base_dir, 'prompts.json')
    output_dir = os.path.join(base_dir, 'evaluation_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")
    
    # Dimensions to evaluate
    dimensions = [
        'motion_smoothness',
        'aesthetic_quality', 
        'human_action',
        'subject_consistency'
    ]
    
    results = {
        'deep': {},
        'naive': {},
        'comparison': {}
    }
    
    for dimension in dimensions:
        print(f"\n{'='*60}")
        print(f"Evaluating dimension: {dimension}")
        print('='*60)
        
        # Evaluate deep videos
        print(f"\nEvaluating DEEP videos for {dimension}...")
        deep_results = evaluate_dimension(deep_dir, dimension, prompts, output_dir)
        if deep_results:
            results['deep'][dimension] = deep_results
        
        # Evaluate naive videos
        print(f"\nEvaluating NAIVE videos for {dimension}...")
        naive_results = evaluate_dimension(naive_dir, dimension, prompts, output_dir)
        if naive_results:
            results['naive'][dimension] = naive_results
        
        # Compare results
        if deep_results and naive_results:
            deep_scores = deep_results.get(dimension, [])
            naive_scores = naive_results.get(dimension, [])
            
            if isinstance(deep_scores, list) and isinstance(naive_scores, list):
                deep_avg = np.mean([s for s in deep_scores if isinstance(s, (int, float))])
                naive_avg = np.mean([s for s in naive_scores if isinstance(s, (int, float))])
            else:
                deep_avg = deep_scores if isinstance(deep_scores, (int, float)) else 0
                naive_avg = naive_scores if isinstance(naive_scores, (int, float)) else 0
            
            results['comparison'][dimension] = {
                'deep_avg': float(deep_avg),
                'naive_avg': float(naive_avg),
                'difference': float(deep_avg - naive_avg),
                'winner': 'deep' if deep_avg > naive_avg else 'naive' if naive_avg > deep_avg else 'tie'
            }
            
            print(f"\n{dimension} Results:")
            print(f"  Deep average:  {deep_avg:.4f}")
            print(f"  Naive average: {naive_avg:.4f}")
            print(f"  Winner: {results['comparison'][dimension]['winner']}")
    
    # Save final results
    final_results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY")
    print('='*60)
    for dim, comp in results['comparison'].items():
        print(f"{dim}:")
        print(f"  Deep:  {comp['deep_avg']:.4f}")
        print(f"  Naive: {comp['naive_avg']:.4f}")
        print(f"  Winner: {comp['winner']} (diff: {comp['difference']:+.4f})")
        print()
    
    print(f"Results saved to: {final_results_path}")

if __name__ == "__main__":
    main()
