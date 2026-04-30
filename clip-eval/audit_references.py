"""Audit the references/<video_id>/ folders for visually inconsistent reference images.

For each video, encodes every reference image with CLIP, computes pairwise cosine
similarity, and flags the image whose mean similarity to the others is lowest as
the likely outlier (the one polluting the eval signal).

Run:
    python3 audit_references.py             # report only
    python3 audit_references.py --quarantine  # move outliers to references_outliers/
"""

import argparse
import json
import os
import shutil
from typing import Dict, List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

REF_ROOT = "references"
OUTLIER_DIR = "references_outliers"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_NAME = "clip-ViT-L-14"


def list_refs(video_dir: str) -> List[str]:
    if not os.path.isdir(video_dir):
        return []
    files = []
    for f in sorted(os.listdir(video_dir)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarantine", action="store_true",
                    help="Move flagged outlier images to references_outliers/.")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="If a ref's mean sim to its peers is below this, "
                         "it's flagged as an outlier (default 0.55).")
    ap.add_argument("--video-ids", nargs="*", default=None,
                    help="Only audit these video ids (e.g. 4.mp4 15.mp4). "
                         "Defaults to all subfolders of references/.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    if args.video_ids:
        video_ids = args.video_ids
    else:
        video_ids = sorted(
            [d for d in os.listdir(REF_ROOT) if os.path.isdir(os.path.join(REF_ROOT, d))],
            key=lambda x: int(x.split('.')[0]),
        )

    report: Dict[str, Dict] = {}
    flagged_total = 0

    print(f"\n{'video':<10} {'file':<22} {'mean_sim_to_peers':>18}   note")
    print("-" * 70)
    for vid in video_ids:
        vdir = os.path.join(REF_ROOT, vid)
        files = list_refs(vdir)
        if len(files) < 2:
            print(f"{vid:<10} (only {len(files)} ref, skipping)")
            continue

        images = []
        valid_files = []
        for f in files:
            try:
                img = Image.open(os.path.join(vdir, f)).convert("RGB")
                images.append(img)
                valid_files.append(f)
            except Exception as e:
                print(f"  ! failed to read {f}: {e}")

        if len(images) < 2:
            continue

        embeds = model.encode(images, convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(embeds, embeds)  # [N, N]
        n = sim.shape[0]
        mean_sims: List[float] = []
        for i in range(n):
            others = [sim[i, j].item() for j in range(n) if j != i]
            mean_sims.append(sum(others) / len(others))

        outlier_idx = int(min(range(n), key=lambda i: mean_sims[i]))
        outlier_score = mean_sims[outlier_idx]
        outlier_file = valid_files[outlier_idx]
        is_flagged = outlier_score < args.threshold

        for i, f in enumerate(valid_files):
            tag = ""
            if i == outlier_idx and is_flagged:
                tag = "<-- OUTLIER (below threshold)"
                flagged_total += 1
            elif i == outlier_idx:
                tag = "(lowest, but above threshold)"
            print(f"{vid:<10} {f:<22} {mean_sims[i]:>18.4f}   {tag}")

        report[vid] = {
            "files": valid_files,
            "mean_sim_to_peers": mean_sims,
            "outlier_file": outlier_file,
            "outlier_mean_sim": outlier_score,
            "flagged": is_flagged,
        }

        if is_flagged and args.quarantine:
            dst_dir = os.path.join(OUTLIER_DIR, vid)
            os.makedirs(dst_dir, exist_ok=True)
            src = os.path.join(vdir, outlier_file)
            dst = os.path.join(dst_dir, outlier_file)
            shutil.move(src, dst)
            print(f"  -> moved {src} -> {dst}")

    with open("reference_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFlagged {flagged_total} outlier reference(s) below threshold {args.threshold}")
    print("Saved: reference_audit.json")
    if not args.quarantine and flagged_total:
        print("Re-run with --quarantine to move flagged refs to ./references_outliers/")


if __name__ == "__main__":
    main()
