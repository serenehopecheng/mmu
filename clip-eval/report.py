"""Combine CLIP, GPT-4o, and VBench results into a single readable report.

Outputs:
  - report.json   (machine-readable, normalized per-video / per-model)
  - report.md     (human-readable summary with tables)

Run: python3 report.py
"""

import json
import os
from typing import Any, Dict, List

MODELS = ["baseline", "naive", "deep"]
CLIP_DETAILED = "clip_results_detailed.json"
GPT4O_FILE = "evaluation_results_gpt4o.json"
VBENCH_DIR = "results/vbench_custom"
PROMPTS_FILE = "prompts.json"

VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def safe_load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_clip() -> Dict[str, Dict[str, Any]]:
    d = safe_load(CLIP_DETAILED)
    if not d:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for m in MODELS:
        if m not in d:
            continue
        videos = d[m].get("videos", {})
        out[m] = {
            "image_mean": d[m].get("image_mean"),
            "text_mean": d[m].get("text_mean"),
            "per_video": {
                v: {
                    "image_score": info.get("image_score"),
                    "text_score": info.get("text_score"),
                }
                for v, info in videos.items()
            },
        }
    return out


def load_gpt4o() -> Dict[str, Dict[str, Any]]:
    d = safe_load(GPT4O_FILE)
    if not d or "models" not in d:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for m in MODELS:
        if m not in d["models"]:
            continue
        agg = d["models"][m].get("aggregate", {})
        per = d["models"][m].get("per_video", {})
        out[m] = {
            "avg_prompt_alignment": agg.get("avg_prompt_alignment_score"),
            "avg_correctness": agg.get("avg_correctness_score"),
            "checklist_recall": agg.get("checklist_element_recall"),
            "per_video": {
                v: {
                    "prompt_alignment": info.get("prompt_alignment_score"),
                    "correctness": info.get("correctness_score"),
                    "summary": info.get("summary", ""),
                }
                for v, info in per.items()
                if isinstance(info, dict) and "error" not in info
            },
        }
    return out


def load_vbench() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in MODELS:
        path = os.path.join(VBENCH_DIR, f"custom_{m}_eval_results.json")
        d = safe_load(path)
        if not d:
            continue
        agg: Dict[str, float] = {}
        per_video: Dict[str, Dict[str, float]] = {}
        for dim in VBENCH_DIMS:
            entry = d.get(dim)
            if not entry or not isinstance(entry, list) or len(entry) < 2:
                continue
            agg[dim] = entry[0]
            for row in entry[1]:
                vp = row.get("video_path", "")
                vid = os.path.basename(vp)
                per_video.setdefault(vid, {})[dim] = row.get("video_results")
        out[m] = {"aggregate": agg, "per_video": per_video}
    return out


def fmt(x, w=8, prec=4):
    if x is None:
        return f"{'N/A':>{w}}"
    if isinstance(x, float):
        return f"{x:>{w}.{prec}f}"
    return f"{str(x):>{w}}"


def build_combined(prompts, clip, gpt4o, vbench) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        "models": {},
        "per_video": {},
        "video_ids": sorted(prompts.keys(), key=lambda x: int(x.split('.')[0])),
        "prompts": prompts,
    }
    for m in MODELS:
        combined["models"][m] = {
            "clip": (clip.get(m) or {}).get("image_mean"),
            "clip_text": (clip.get(m) or {}).get("text_mean"),
            "gpt4o_alignment": (gpt4o.get(m) or {}).get("avg_prompt_alignment"),
            "gpt4o_correctness": (gpt4o.get(m) or {}).get("avg_correctness"),
            "gpt4o_recall": (gpt4o.get(m) or {}).get("checklist_recall"),
            "vbench": (vbench.get(m) or {}).get("aggregate", {}),
        }

    for v in combined["video_ids"]:
        combined["per_video"][v] = {}
        for m in MODELS:
            entry: Dict[str, Any] = {}
            cv = (clip.get(m, {}).get("per_video") or {}).get(v, {})
            entry["clip"] = cv.get("image_score")
            entry["clip_text"] = cv.get("text_score")
            gv = (gpt4o.get(m, {}).get("per_video") or {}).get(v, {})
            entry["gpt4o_alignment"] = gv.get("prompt_alignment")
            entry["gpt4o_correctness"] = gv.get("correctness")
            vbv = (vbench.get(m, {}).get("per_video") or {}).get(v, {})
            entry["vbench"] = {dim: vbv.get(dim) for dim in VBENCH_DIMS}
            combined["per_video"][v][m] = entry

    return combined


def render_markdown(combined: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Evaluation Report\n")
    lines.append(f"Models: {', '.join(MODELS)}  ")
    lines.append(f"Videos: {len(combined['video_ids'])}\n")

    # Per-model summary
    lines.append("## Per-model summary\n")
    header = (
        "| model | CLIP image | CLIP text | GPT-4o align | GPT-4o correct | Recall | "
        "subj cons | bg cons | motion | dyn deg | aesth | img qual |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append(header)
    lines.append(sep)
    for m in MODELS:
        d = combined["models"][m]
        vb = d.get("vbench") or {}
        cells = [
            f"**{m}**",
            fmt_cell(d.get("clip"), 4),
            fmt_cell(d.get("clip_text"), 4),
            fmt_cell(d.get("gpt4o_alignment"), 2),
            fmt_cell(d.get("gpt4o_correctness"), 2),
            fmt_cell(d.get("gpt4o_recall"), 3),
            fmt_cell(vb.get("subject_consistency"), 4),
            fmt_cell(vb.get("background_consistency"), 4),
            fmt_cell(vb.get("motion_smoothness"), 4),
            fmt_cell(vb.get("dynamic_degree"), 3),
            fmt_cell(vb.get("aesthetic_quality"), 4),
            fmt_cell(vb.get("imaging_quality"), 4),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # Winners summary
    lines.append("\n## Best model per metric\n")
    metric_keys = [
        ("CLIP image (max-over-refs)", "clip", True),
        ("CLIP text (prompt fidelity)", "clip_text", True),
        ("GPT-4o prompt alignment", "gpt4o_alignment", True),
        ("GPT-4o correctness", "gpt4o_correctness", True),
        ("GPT-4o checklist recall", "gpt4o_recall", True),
    ]
    vb_keys = [
        ("VBench subject_consistency", "subject_consistency", True),
        ("VBench background_consistency", "background_consistency", True),
        ("VBench motion_smoothness", "motion_smoothness", True),
        ("VBench dynamic_degree", "dynamic_degree", True),
        ("VBench aesthetic_quality", "aesthetic_quality", True),
        ("VBench imaging_quality", "imaging_quality", True),
    ]
    lines.append("| metric | winner | values |")
    lines.append("|---|---|---|")
    for label, key, higher in metric_keys:
        vals = {m: combined["models"][m].get(key) for m in MODELS}
        winner = pick_winner(vals, higher)
        val_str = ", ".join(f"{m}={fmt_cell(vals[m], 4)}" for m in MODELS)
        lines.append(f"| {label} | **{winner}** | {val_str} |")
    for label, key, higher in vb_keys:
        vals = {m: (combined["models"][m].get("vbench") or {}).get(key) for m in MODELS}
        winner = pick_winner(vals, higher)
        val_str = ", ".join(f"{m}={fmt_cell(vals[m], 4)}" for m in MODELS)
        lines.append(f"| {label} | **{winner}** | {val_str} |")

    # Per-video CLIP image winners
    lines.append("\n## Per-video winners (CLIP image)\n")
    lines.append("| video | baseline | naive | deep | winner |")
    lines.append("|---|---|---|---|---|")
    for v in combined["video_ids"]:
        vals = {m: combined["per_video"][v][m].get("clip") for m in MODELS}
        winner = pick_winner(vals, higher=True)
        lines.append(
            f"| {v} | {fmt_cell(vals['baseline'], 4)} | {fmt_cell(vals['naive'], 4)} "
            f"| {fmt_cell(vals['deep'], 4)} | **{winner}** |"
        )

    # Per-video GPT-4o (alignment)
    lines.append("\n## Per-video GPT-4o prompt alignment\n")
    lines.append("| video | baseline | naive | deep |")
    lines.append("|---|---|---|---|")
    for v in combined["video_ids"]:
        cells = [v]
        for m in MODELS:
            cells.append(fmt_cell(combined["per_video"][v][m].get("gpt4o_alignment"), 2))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def fmt_cell(x, prec=4):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def pick_winner(vals: Dict[str, Any], higher: bool = True) -> str:
    available = {k: v for k, v in vals.items() if v is not None}
    if not available:
        return "N/A"
    return max(available, key=available.get) if higher else min(available, key=available.get)


def render_text(combined: Dict[str, Any]) -> str:
    """Plain-text terminal summary."""
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("EVALUATION REPORT".center(88))
    lines.append("=" * 88)

    lines.append("\nPer-model aggregate:")
    lines.append(
        f"{'model':<10} {'clip_img':>9} {'clip_txt':>9} {'gpt_align':>10} {'gpt_corr':>9} "
        f"{'recall':>8} {'subj':>8} {'bg':>8} {'motion':>8} {'dyn':>6} {'aesth':>8} {'imgq':>8}"
    )
    for m in MODELS:
        d = combined["models"][m]
        vb = d.get("vbench") or {}
        lines.append(
            f"{m:<10} "
            f"{_t(d.get('clip'), 9, 4)} {_t(d.get('clip_text'), 9, 4)} "
            f"{_t(d.get('gpt4o_alignment'), 10, 2)} {_t(d.get('gpt4o_correctness'), 9, 2)} "
            f"{_t(d.get('gpt4o_recall'), 8, 3)} "
            f"{_t(vb.get('subject_consistency'), 8, 4)} {_t(vb.get('background_consistency'), 8, 4)} "
            f"{_t(vb.get('motion_smoothness'), 8, 4)} {_t(vb.get('dynamic_degree'), 6, 2)} "
            f"{_t(vb.get('aesthetic_quality'), 8, 4)} {_t(vb.get('imaging_quality'), 8, 4)}"
        )

    lines.append("\nPer-video CLIP image (max-over-refs):")
    lines.append(f"{'video':<8} {'baseline':>10} {'naive':>10} {'deep':>10}   winner")
    for v in combined["video_ids"]:
        vals = {m: combined["per_video"][v][m].get("clip") for m in MODELS}
        winner = pick_winner(vals)
        lines.append(
            f"{v:<8} {_t(vals['baseline'], 10, 4)} {_t(vals['naive'], 10, 4)} "
            f"{_t(vals['deep'], 10, 4)}   {winner}"
        )

    lines.append("\nPer-video GPT-4o prompt alignment (1-5):")
    lines.append(f"{'video':<8} {'baseline':>10} {'naive':>10} {'deep':>10}")
    for v in combined["video_ids"]:
        row = f"{v:<8}"
        for m in MODELS:
            row += " " + _t(combined["per_video"][v][m].get("gpt4o_alignment"), 10, 2)
        lines.append(row)

    return "\n".join(lines)


def _t(x, w, prec):
    if x is None:
        return f"{'N/A':>{w}}"
    if isinstance(x, float):
        return f"{x:>{w}.{prec}f}"
    return f"{str(x):>{w}}"


def main():
    prompts = safe_load(PROMPTS_FILE) or {}
    clip = load_clip()
    gpt4o = load_gpt4o()
    vbench = load_vbench()

    if not clip:
        print(f"warning: no CLIP results at {CLIP_DETAILED}")
    if not gpt4o:
        print(f"warning: no GPT-4o results at {GPT4O_FILE}")
    missing_gpt4o = [m for m in MODELS if m not in gpt4o]
    if missing_gpt4o:
        print(f"warning: GPT-4o eval missing for: {missing_gpt4o}")
    missing_vb = [m for m in MODELS if m not in vbench]
    if missing_vb:
        print(f"warning: VBench eval missing for: {missing_vb}")

    combined = build_combined(prompts, clip, gpt4o, vbench)

    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    md = render_markdown(combined)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(render_text(combined))
    print("\nSaved: report.json, report.md")


if __name__ == "__main__":
    main()
