import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import cv2
import json
import os
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NUM_FRAMES = 12
# ViT-L/14 is far more discriminative than ViT-B/32. Swap to
# 'clip-ViT-B-32' if you need the old behaviour or to save VRAM.
MODEL_NAME = "clip-ViT-L-14"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)


def get_video_frames(video_path, num_frames=NUM_FRAMES):
    """Extracts N evenly spaced frames from the video as RGB PIL images."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def list_reference_images(ref_folder):
    if not os.path.exists(ref_folder):
        return []
    paths = []
    for f in sorted(os.listdir(ref_folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(os.path.join(ref_folder, f))
    return paths


def compute_similarity(video_path, reference_image_paths, prompt_text):
    """Computes image-image (max-over-refs) and text-image CLIP similarities."""
    video_frames = get_video_frames(video_path)
    ref_images = []
    for p in reference_image_paths:
        if not os.path.exists(p):
            continue
        try:
            ref_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  ! skipping unreadable ref {p}: {e}")

    empty = {
        "image_score": 0.0,
        "text_score": 0.0,
        "num_frames": len(video_frames),
        "num_references": len(ref_images),
        "frame_image_scores": [],
        "frame_text_scores": [],
        "pairwise_similarity": [],
    }
    if not video_frames:
        return empty

    video_embeddings = model.encode(video_frames, convert_to_tensor=True, show_progress_bar=False)

    # --- image <-> image (max over references per frame) ---
    if ref_images:
        ref_embeddings = model.encode(ref_images, convert_to_tensor=True, show_progress_bar=False)
        cos_sim = util.cos_sim(video_embeddings, ref_embeddings)  # [F, R]
        # Use max over references so a frame is judged by its best-matching
        # reference, instead of being dragged down by visually conflicting refs.
        frame_image_scores = torch.max(cos_sim, dim=1).values.tolist()
        pairwise = cos_sim.tolist()
    else:
        frame_image_scores = []
        pairwise = []

    # --- text <-> image (prompt fidelity) ---
    if prompt_text:
        text_embedding = model.encode([prompt_text], convert_to_tensor=True, show_progress_bar=False)
        text_sim = util.cos_sim(video_embeddings, text_embedding).squeeze(-1)  # [F]
        frame_text_scores = text_sim.tolist()
    else:
        frame_text_scores = []

    image_score = float(np.mean(frame_image_scores)) if frame_image_scores else 0.0
    text_score = float(np.mean(frame_text_scores)) if frame_text_scores else 0.0

    return {
        "image_score": image_score,
        "text_score": text_score,
        "num_frames": len(video_frames),
        "num_references": len(ref_images),
        "frame_image_scores": frame_image_scores,
        "frame_text_scores": frame_text_scores,
        "pairwise_similarity": pairwise,
    }


def main():
    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    summary = {}
    detailed_results = {}

    for model_name in ["baseline", "naive", "deep"]:
        image_scores = []
        text_scores = []
        model_details = {}
        for video_id, prompt_text in prompts.items():
            video_file = f"{model_name}/{video_id}"
            ref_folder = f"./references/{video_id}/"
            ref_paths = list_reference_images(ref_folder)

            if not os.path.exists(video_file):
                print(f"  ! missing {video_file}")
                continue

            print(f"[{model_name}] {video_id} ({len(ref_paths)} refs)")
            sim = compute_similarity(video_file, ref_paths, prompt_text)
            image_scores.append(sim["image_score"])
            text_scores.append(sim["text_score"])
            model_details[video_id] = {
                "video_path": video_file,
                "prompt": prompt_text,
                "reference_paths": ref_paths,
                "image_score": sim["image_score"],
                "text_score": sim["text_score"],
                "num_frames": sim["num_frames"],
                "num_references": sim["num_references"],
                "frame_image_scores": sim["frame_image_scores"],
                "frame_text_scores": sim["frame_text_scores"],
                "pairwise_similarity": sim["pairwise_similarity"],
            }

        detailed_results[model_name] = {
            "image_mean": float(np.mean(image_scores)) if image_scores else 0.0,
            "text_mean": float(np.mean(text_scores)) if text_scores else 0.0,
            "num_videos": len(image_scores),
            "videos": model_details,
        }
        summary[model_name] = {
            "image_mean": detailed_results[model_name]["image_mean"],
            "text_mean": detailed_results[model_name]["text_mean"],
        }

    print("\n--- Final CLIP Comparison (max-over-refs image / prompt text) ---")
    print(f"{'model':<10} {'image':>10} {'text':>10}")
    for m, s in summary.items():
        print(f"{m.upper():<10} {s['image_mean']:>10.4f} {s['text_mean']:>10.4f}")

    print("\n--- Per-video breakdown ---")
    video_ids = sorted(prompts.keys(), key=lambda x: int(x.split('.')[0]))
    print(f"{'video':<8} | {'baseline':>18} {'naive':>18} {'deep':>18}")
    print(f"{'':<8} | {'img':>9}{'txt':>9} {'img':>9}{'txt':>9} {'img':>9}{'txt':>9}")
    for v in video_ids:
        row = f"{v:<8} |"
        for m in ["baseline", "naive", "deep"]:
            d = detailed_results[m]["videos"].get(v)
            if d is None:
                row += f" {'-':>9}{'-':>9}"
            else:
                row += f" {d['image_score']:>9.4f}{d['text_score']:>9.4f}"
        print(row)

    with open("clip_results.json", "w") as f:
        json.dump(summary, f, indent=4)

    with open("clip_results_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=4)


if __name__ == "__main__":
    main()
