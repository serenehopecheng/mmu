import json
import os
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}

def build_map(video_dir, prompts):
    video_dir = Path(video_dir)
    out = {}
    missing = []

    for p in sorted(video_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        # prompts.json may key by full filename ("0.mp4") or stem ("0")
        key = p.name if p.name in prompts else p.stem
        if key not in prompts:
            missing.append(p.name)
            continue
        out[p.name] = prompts[key]  # VBench expects {"video_path": prompt}

    if missing:
        print(f"[warn] missing prompts for: {missing}")
    return out

def main():
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    deep_map = build_map("deep", prompts)
    naive_map = build_map("naive", prompts)

    with open("deep_prompts_vbench.json", "w", encoding="utf-8") as f:
        json.dump(deep_map, f, indent=2, ensure_ascii=False)

    with open("naive_prompts_vbench.json", "w", encoding="utf-8") as f:
        json.dump(naive_map, f, indent=2, ensure_ascii=False)

    print(f"deep:  {len(deep_map)} videos mapped")
    print(f"naive: {len(naive_map)} videos mapped")

if __name__ == "__main__":
    main()