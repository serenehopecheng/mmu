import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import cv2
import json
import os
import numpy as np

# Load the CLIP model
model = SentenceTransformer('clip-ViT-B-32')

def get_video_frames(video_path, num_frames=5):
    """Extracts N evenly spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def compute_similarity(video_path, reference_image_paths):
    """Computes avg similarity between video frames and reference images."""
    video_frames = get_video_frames(video_path)
    ref_images = [Image.open(p) for p in reference_image_paths if os.path.exists(p)]
    
    if not ref_images: return 0.0

    # Encode images to vectors
    video_embeddings = model.encode(video_frames)
    ref_embeddings = model.encode(ref_images)

    # Compute cosine similarity (matrix multiplication)
    # Resulting shape: [num_video_frames, num_ref_images]
    cos_sim = util.cos_sim(video_embeddings, ref_embeddings)
    
    # Return average similarity across all frame-reference pairs
    return torch.mean(cos_sim).item()

def main():
    # models_to_test = ["baseline", "naive_rag", "deep_research"]
    # Assuming prompts.json links video_id to the reference image set
    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    results = {}

    for model_name in ["baseline", "naive", "deep"]:
        model_scores = []
        for video_id in prompts.keys():
            video_file = f"{model_name}/{video_id}"
            # Logic: Look for reference images named {video_id}_ref1.jpg etc.
            ref_folder = f"./references/{video_id}/"
            ref_paths = [os.path.join(ref_folder, f) for f in os.listdir(ref_folder)] if os.path.exists(ref_folder) else []
            
            if os.path.exists(video_file):
                score = compute_similarity(video_file, ref_paths)
                model_scores.append(score)
        
        results[model_name] = np.mean(model_scores) if model_scores else 0

    print("\n--- Final CLIP Similarity Comparison ---")
    for m, score in results.items():
        print(f"{m.upper()}: {score:.4f}")

    with open("clip_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()