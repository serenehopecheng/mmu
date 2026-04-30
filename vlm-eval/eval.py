import base64
import json
import os
from typing import Any, Dict, List

import cv2
import openai

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_frames_to_base64(video_path: str, max_frames: int = 8) -> List[str]:
    """Sample up to max_frames uniformly across the video and return base64 JPEGs."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        video.release()
        return []

    sample_count = min(max_frames, total_frames)
    indices = [
        int(i * (total_frames - 1) / max(1, sample_count - 1)) for i in range(sample_count)
    ]

    base64_frames: List[str] = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video.read()
        if not ok:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return base64_frames


def build_gpt4o_message(
    prompt: str, expected_elements: List[str], frame_base64_list: List[str]
) -> List[Dict[str, Any]]:
    checklist = "\n".join([f"- {item}" for item in expected_elements])
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Target prompt:\n{prompt}\n\n"
                "Check whether each expected visual element appears in the generated video.\n"
                f"Expected elements:\n{checklist}"
            ),
        }
    ]

    for base64_img in frame_base64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "high",
                },
            }
        )
    return content


def score_video_gpt4o(video_path: str, prompt: str, expected_elements: List[str]) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = extract_frames_to_base64(video_path, max_frames=8)
    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    message_content = build_gpt4o_message(prompt, expected_elements, frames)

    system_prompt = """You are a strict video evaluation judge.

You receive:
1) A target prompt.
2) A checklist of expected visual elements.
3) Sampled frames from a generated video.

Evaluate visual grounding and prompt correctness.

Return ONLY valid JSON with this exact schema:
{
  "prompt_alignment_score": <integer 1-5>,
  "correctness_score": <integer 1-5>,
  "elements": [
    {
      "element": "<string copied from checklist>",
      "present": <true or false>,
      "confidence": <number between 0 and 1>,
      "evidence": "<brief evidence from frames>"
    }
  ],
  "summary": "<1-3 sentence justification>"
}

Scoring guidance:
- prompt_alignment_score: How well the video matches the overall prompt intent.
- correctness_score: Fidelity and procedural/attribute correctness across checklist items.
- Be conservative: if unclear, mark present=false with lower confidence.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def evaluate_model(
    model_name: str, prompts: Dict[str, str], expectations: Dict[str, List[str]]
) -> Dict[str, Any]:
    model_results: Dict[str, Any] = {}
    alignment_scores: List[float] = []
    correctness_scores: List[float] = []
    element_hits = 0
    element_total = 0

    for video_id in expectations.keys():
        prompt = prompts.get(video_id, "")
        expected_elements = expectations.get(video_id, [])
        video_path = os.path.join(model_name, video_id)

        print(f"[{model_name}] Evaluating {video_id}...")
        try:
            result = score_video_gpt4o(video_path, prompt, expected_elements)
            model_results[video_id] = result
            alignment_scores.append(float(result.get("prompt_alignment_score", 0)))
            correctness_scores.append(float(result.get("correctness_score", 0)))

            for item in result.get("elements", []):
                element_total += 1
                if item.get("present") is True:
                    element_hits += 1
        except Exception as exc:
            model_results[video_id] = {"error": str(exc)}
            print(f"  [ERROR] {video_id}: {exc}")

    avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
    checklist_recall = (element_hits / element_total) if element_total else 0.0

    return {
        "aggregate": {
            "videos_evaluated": len(alignment_scores),
            "avg_prompt_alignment_score": avg_alignment,
            "avg_correctness_score": avg_correctness,
            "checklist_element_recall": checklist_recall,
        },
        "per_video": model_results,
    }


def main() -> None:
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts: Dict[str, str] = json.load(f)
    with open("expectations.json", "r", encoding="utf-8") as f:
        expectations: Dict[str, List[str]] = json.load(f)

    # Use only the 10 target IDs defined in expectations.json.
    target_ids = list(expectations.keys())
    prompts = {video_id: prompts.get(video_id, "") for video_id in target_ids}

    models_to_eval = ["baseline", "deep", "naive"]

    final_results = {
        "config": {
            "models": models_to_eval,
            "target_video_ids": target_ids,
            "model_used": "gpt-4o",
        },
        "models": {},
    }

    for model_name in models_to_eval:
        final_results["models"][model_name] = evaluate_model(model_name, prompts, expectations)

    out_path = "evaluation_results_gpt4o.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {out_path}")


if __name__ == "__main__":
    main()