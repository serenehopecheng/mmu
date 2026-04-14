import os, re, asyncio, requests, nest_asyncio, openai, subprocess, tempfile
import sys, io
import subprocess, time, uuid, base64
import aiohttp
from urllib.parse import quote_plus
from typing import Dict, Any, Optional, List
import numpy as np
from pydantic import BaseModel
from autogen_core.tools import BaseTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import Image as AutogenImage
from PIL import Image as PILImage
from dotenv import load_dotenv
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx, CompositeAudioClip
from elevenlabs import ElevenLabs
import json
import base64
from pathlib import Path

# from naive import PROJECT_ID

load_dotenv()
# nest_asyncio.apply()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# PROJECT_ID = "gen-lang-client-0239546387"
PROJECT_ID = "thermal-petal-487200-j0"
MODEL_ID = "veo-3.0-fast-generate-001"
LOCATION = "us-central1"

def get_access_token():
    result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def generate_video(prompt, duration_seconds=8, frame=None):
    """Generate video with optional reference frame for continuity"""
    access_token = get_access_token()
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:predictLongRunning"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    data = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "durationSeconds": duration_seconds,
            "sampleCount": 1, 
            "generateAudio": False
        }
    }
    
    if frame and isinstance(frame, (bytes, bytearray)):
        frame_b64 = base64.b64encode(frame).decode("utf-8")
        data["instances"][0]["image"] = {
            "bytesBase64Encoded": frame_b64,
            "mimeType": "image/jpeg"
        }
        print("  📸 Using previous frame for continuity")

    response = requests.post(url, headers=headers, json=data)
    print(response.status_code, response.headers.get("content-type"))
    print(response.text[:1000])
    
    response_json = response.json()
    
    if "error" in response_json:
        raise Exception(f"Veo API Error: {response_json['error']['message']}")

    return response_json["name"]

def poll_operation(operation_name):
    access_token = get_access_token()
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:fetchPredictOperation"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {"operationName": operation_name}
    
    while True:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        if result.get("done") and "response" in result and "videos" in result["response"]:
            return result
            
        time.sleep(30)

def extract_frame_bytes(video_path: str):
    vp = Path(video_path)
    out_dir = Path("test/r")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{vp.stem}_last.jpg"

    cmd = [
        "ffmpeg", "-v", "error",
        "-sseof", "-0.25",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
        "-y"
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open(output_path, "rb") as f:
        return f.read()

def download_video(gcs_uri, filename):
    subprocess.run(["gsutil", "cp", gcs_uri, filename], check=True)

def clean_json_block(s: str):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s

class WebKnowledgeRetrieverArgs(BaseModel):
    query: str
    detail_level: str = "comprehensive"

class WebKnowledgeRetrieverTool(BaseTool[WebKnowledgeRetrieverArgs, str]):
    def __init__(self):
        super().__init__(
            args_type=WebKnowledgeRetrieverArgs,
            return_type=str,
            name="web_knowledge_retriever",
            description="Retrieves comprehensive knowledge from the web using MultimodalWebSurfer"
        )
        client = OpenAIChatCompletionClient(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        self.web_surfer = MultimodalWebSurfer(
            name="WebSurfer",
            model_client=client,
            description="Search and browse the web to gather comprehensive knowledge for video generation."
        )

    async def run(self, args: WebKnowledgeRetrieverArgs, context):
        search_queries = [
            f"{args.query} detailed explanation",
            f"{args.query} tutorial guide",
            f"{args.query} step by step",
            f"how {args.query} works visual guide"
        ]
        
        all_knowledge = []
        
        for query in search_queries[:2]:
            print(f"WebSurfer searching: {query}")
            search_message = TextMessage(
                content=f"Search for comprehensive information about: {query}. Focus on visual descriptions, step-by-step processes, and key concepts that would translate well to video format.",
                source="user"
            )
            
            response = await self.web_surfer.on_messages([search_message], context)
            
            content = ""
            if hasattr(response, 'chat_message'):
                if hasattr(response.chat_message, 'content'):
                    content = response.chat_message.content
            elif hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, list) and len(response) > 0:
                if hasattr(response[-1], 'content'):
                    content = response[-1].content
                else:
                    content = str(response[-1])
            else:
                content = str(response)
            
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            
            all_knowledge.append(str(content))
        
        combined_knowledge = "\n\n---SEARCH RESULT---\n\n".join(all_knowledge)
        
        print(f"WebSurfer complete: {len(combined_knowledge)} total chars of knowledge")
        return combined_knowledge

class ImageRetrieverArgs(BaseModel):
    query: str
    iid: str
    num_images: int = 1 

class ImageRetrieverTool(BaseTool[ImageRetrieverArgs, list]):
    MAX_URLS_PER_ENGINE = 5
    MAX_VALIDATION_ATTEMPTS = 6
    ACCEPT_SCORE_THRESHOLD = 6
    EARLY_EXIT_SCORE = 8

    def __init__(self):
        super().__init__(
            args_type=ImageRetrieverArgs,
            return_type=list,
            name="image_retriever",
            description="Retrieves reference images from the web for video generation"
        )

    def _scrape_google_images(self, query: str, num_results: int = 20) -> list[str]:
        """Scrape image URLs from Google Images search results."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&ijn=0"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text
        
        urls = []
        for match in re.finditer(r'\["(https?://[^"]+)",[0-9]+,[0-9]+\]', html):
            img_url = match.group(1)
            if any(img_url.endswith(ext) or f".{ext}?" in img_url or f".{ext}%" in img_url
                   for ext in ("jpg", "jpeg", "png", "webp")):
                if "gstatic.com" not in img_url and "google.com" not in img_url:
                    urls.append(img_url)
        
        if not urls:
            for match in re.finditer(r'(https?://[^\s"\\]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s"\\]*)?)', html):
                img_url = match.group(1).replace("\\u003d", "=").replace("\\u0026", "&")
                if "gstatic.com" not in img_url and "google.com" not in img_url:
                    urls.append(img_url)
        
        return list(dict.fromkeys(urls))[:num_results]

    def _scrape_duckduckgo_images(self, query: str, num_results: int = 20) -> list[str]:
        """Scrape image URLs from DuckDuckGo image search (vqd token + API)."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })
        
        token_resp = session.get(f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images", timeout=15)
        vqd_match = re.search(r'vqd=["\']([^"\']+)', token_resp.text)
        if not vqd_match:
            return []
        vqd = vqd_match.group(1)
        
        api_url = "https://duckduckgo.com/i.js"
        params = {"l": "us-en", "o": "json", "q": query, "vqd": vqd, "f": ",,,,,", "p": "1"}
        api_resp = session.get(api_url, params=params, timeout=15)
        results = api_resp.json().get("results", [])
        
        urls = []
        for r in results:
            img_url = r.get("image", "")
            if img_url and any(ext in img_url.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                urls.append(img_url)
        return list(dict.fromkeys(urls))[:num_results]

    def _validate_image(self, img_bytes: bytes, query: str) -> dict | None:
        """Use GPT-4o to validate an image is suitable as a visual reference."""
        if len(img_bytes) < 5000:
            return None
        
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        validation_prompt = f"""Analyze this image for the query: "{query}"

        This image will be used as a VISUAL REFERENCE for video generation. We need images that show what the subject LOOKS LIKE.

        ACCEPT images that:
        - Show real-world subjects, objects, people, or scenes 
        - Are photographs or photo-realistic images (not pure illustrations/cartoons)
        - Are relevant to the query topic
        - Have acceptable visual quality

        Respond with JSON only:
        {{"is_valid": true/false, "reason": "brief explanation", "relevance_score": 0-10}}"""

        validation_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": validation_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_completion_tokens=100
        )
        validation_text = validation_response.choices[0].message.content.strip()
        validation_text = clean_json_block(validation_text)
        return json.loads(validation_text)

    async def run(self, args: ImageRetrieverArgs, context):
        search_queries = [
            f"{args.query} photo",
            f"{args.query} real",
            f"{args.query} how to",
        ]

        all_urls = []
        for sq in search_queries:
            print(f"Searching images: {sq}")
            try:
                urls = self._scrape_google_images(sq, num_results=self.MAX_URLS_PER_ENGINE)
                print(f"  Google Images: {len(urls)} URLs")
                all_urls.extend(urls)
            except Exception as e:
                print(f"  Google Images failed: {str(e)[:80]}")
            try:
                urls = self._scrape_duckduckgo_images(sq, num_results=self.MAX_URLS_PER_ENGINE)
                print(f"  DuckDuckGo Images: {len(urls)} URLs")
                all_urls.extend(urls)
            except Exception as e:
                print(f"  DuckDuckGo Images failed: {str(e)[:80]}")

        all_urls = list(dict.fromkeys(all_urls))

        unique_dir = Path("test/2_unique")
        unique_dir.mkdir(parents=True, exist_ok=True)
        for i, url in enumerate(all_urls):
            try:
                img_response = requests.get(url, timeout=10, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                })
                if img_response.status_code == 200:
                    content_type = img_response.headers.get("Content-Type", "")
                    ext = "jpg"
                    if "png" in content_type:
                        ext = "png"
                    elif "gif" in content_type:
                        ext = "gif"
                    elif "webp" in content_type:
                        ext = "webp"
                    filename = unique_dir / f"unique{args.iid}_{i}.{ext}"
                    with open(filename, "wb") as f:
                        f.write(img_response.content)
            except Exception as e:
                print(f"  Failed to save unique image from {url[:60]}: {str(e)[:50]}")
        print(f"Found {len(all_urls)} unique candidate image URLs")

        reference_images = []
        validation_attempts = 0
        for url in all_urls:
            if len(reference_images) >= args.num_images:
                break
            if validation_attempts >= self.MAX_VALIDATION_ATTEMPTS:
                print(f"Stopping validation after {validation_attempts} attempts to limit GPT usage")
                break
            try:
                img_response = requests.get(url, timeout=10, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                })
                if img_response.status_code != 200:
                    continue
                img_bytes = img_response.content

                validation_attempts += 1
                validation_data = self._validate_image(img_bytes, args.query)
                if validation_data is None:
                    print(f"✗ Skipped tiny image: {url[:80]}")
                    continue

                relevance_score = validation_data.get("relevance_score", 0)
                if validation_data.get("is_valid") and relevance_score >= self.ACCEPT_SCORE_THRESHOLD:
                    reference_images.append({
                        "url": url,
                        "bytes": img_bytes,
                        "index": len(reference_images),
                        "relevance_score": relevance_score
                    })
                    print(f"✓ Validated reference image {len(reference_images)} (score: {relevance_score}/10)")
                    if relevance_score >= self.EARLY_EXIT_SCORE:
                        print(f"Stopping early after finding a strong match ({relevance_score}/10)")
                        break
                else:
                    print(f"✗ Rejected image: {validation_data.get('reason', 'low quality')}")

            except Exception as e:
                print(f"✗ Failed to process {url[:60]}: {str(e)[:50]}")
                continue

        reference_images.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if not reference_images and all_urls:
            print("No validated images found, falling back to first downloadable image...")
            for url in all_urls:
                try:
                    img_response = requests.get(url, timeout=10, headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36"
                    })
                    if img_response.status_code == 200 and len(img_response.content) >= 5000:
                        reference_images.append({
                            "url": url,
                            "bytes": img_response.content,
                            "index": 0,
                            "relevance_score": 0
                        })
                        print(f"✓ Using fallback image: {url[:80]}")
                        break
                except Exception:
                    continue
        
        print(f"Successfully validated {len(reference_images)} reference images")
        return reference_images

class ScriptGeneratorArgs(BaseModel):
    query: str
    context_info: Optional[str] = None

class ScriptGeneratorTool(BaseTool[ScriptGeneratorArgs, str]):
    def __init__(self):
        super().__init__(
            args_type=ScriptGeneratorArgs,
            return_type=str,
            name="script_generator",
            description="Generates scripts for short, cinematic videos with narration."
        )

    async def run(self, args: ScriptGeneratorArgs, context):
        prompt = f"""You are writing a prompt for Veo 3, a text-to-video AI model that generates photorealistic real-world footage. 
        Your output will be fed directly to the model as its generation prompt, so write in vivid, descriptive, present-tense language—as if narrating what the camera sees moment by moment.

        TOPIC: "{args.query}"
        BACKGROUND RESEARCH (use for factual accuracy): {args.context_info}

        SCRIPT REQUIREMENTS:
        - The clip is exactly 8 seconds of continuous, photorealistic footage. No cuts, transitions, or scene changes.
        - Describe a single real-world scene with one clear subject performing one simple, observable action.
        - Write in temporal order: what the viewer sees at the start, what unfolds over the 8 seconds, and how the shot ends.
        - Use specific, concrete visual details—materials, textures, colors, lighting quality, and spatial relationships.
        - Camera: stationary or slow dolly/pan only. Specify the angle (e.g. eye-level, overhead, 45-degree). No zooms, whip pans, handheld shake, or rack focus.
        - Lighting: specify the light source (e.g. soft window light, golden hour sun, overhead fluorescents).
        - Keep the scene grounded and filmable—nothing fantastical, animated, graphical, or text-overlay based.
        - Avoid: multiple competing subjects, complex multi-step actions, grid/pattern layouts, technical measurement setups, or scanning motions.

        NARRATION REQUIREMENTS:
        - One spoken sentence, max 18 words, that a voiceover narrator would say over the clip.
        - Should complement the visuals—not merely describe what is shown, but add insight or context.

        OUTPUT (valid JSON only, no other text):
        {{
            "script": "Present-tense, temporally ordered description of the 8-second clip with specific visual and cinematic details.",
            "narration": "A single conversational sentence adding insight to the visuals (max 18 words)."
        }}"""

        r = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1200
        )
        return r.choices[0].message.content.strip()

class PromptCritiqueArgs(BaseModel):
    video_prompt: str
    narration: str
    topic: str

class PromptCritiqueTool(BaseTool[PromptCritiqueArgs, Dict[str, Any]]):
    def __init__(self):
        super().__init__(
            args_type=PromptCritiqueArgs,
            return_type=Dict[str, Any],
            name="prompt_critique",
            description="Scores and rewrites a video script across quality dimensions, returning an improved version."
        )

    async def run(self, args: PromptCritiqueArgs, context):
        critique_prompt = f"""You are an expert script editor for Veo 3, a text-to-video AI that generates photorealistic 8-second clips.

    TOPIC: {args.topic}
    VIDEO SCRIPT: {args.video_prompt}
    NARRATION: {args.narration}

    SCORE each dimension 0-10, then REWRITE the script to maximize all scores.

    SCORING DIMENSIONS:
    1. policy_compliance: No sensitive content, text overlays, graphics, animation, fantasy, or prohibited camera moves (zooms, whip pans, rack focus, scanning).
    2. visual_specificity: Names concrete materials, textures, colors, lighting source/quality, and spatial relationships. Avoids vague words like "beautiful", "amazing", "stunning".
    3. temporal_clarity: Describes a clear start-to-end progression across exactly 8 seconds. The reader can picture what happens at second 1 vs second 4 vs second 8.
    4. single_subject_focus: One subject performing one simple, observable action. No competing elements, no multi-step processes.
    5. camera_feasibility: Camera angle explicitly stated (e.g. eye-level, overhead, 45-degree low angle). Movement is stationary or slow dolly/pan only.
    6. narration_quality: One sentence, max 18 words. Adds insight or context beyond what is visible—not a description of the visuals.
    7. veo_compatibility: Likely to succeed with Veo 3. Penalize: abstract concepts shown literally, grid/pattern layouts, technical measurement setups, multiple scene transitions, complex hand manipulations.

    REWRITE RULES:
    - Replace every vague adjective with a concrete visual detail (color, material, texture, light).
    - Add explicit temporal beats ("The shot opens with...", "Over the next few seconds...", "The clip ends as...").
    - Specify camera angle and movement in the first sentence of the script.
    - Ensure the narration teaches or contextualizes—never just restates the visuals.
    - Simplify any action that requires precise timing or coordination.
    - If the original is fundamentally incompatible with Veo 3, reimagine the scene using an everyday analogy that conveys the same concept.

    Respond with ONLY valid JSON:
    {{
        "scores": {{
            "policy_compliance": 0-10,
            "visual_specificity": 0-10,
            "temporal_clarity": 0-10,
            "single_subject_focus": 0-10,
            "camera_feasibility": 0-10,
            "narration_quality": 0-10,
            "veo_compatibility": 0-10
        }},
        "issues": ["specific issue 1", "..."],
        "rewritten_script": "The improved 8-second video description with all fixes applied.",
        "rewritten_narration": "Improved narration, max 18 words.",
        "explanation": "What was changed and why."
    }}"""

        r = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": critique_prompt}],
            max_completion_tokens=1200
        )
        
        result_text = r.choices[0].message.content.strip()
        result_text = clean_json_block(result_text)
        return json.loads(result_text)

    QUALITY_THRESHOLD = 7.0
    MAX_CRITIQUE_ROUNDS = 3

async def run_iterative_critique(query: str, script_data: dict) -> dict:
    """Score-and-rewrite loop: critique the script, rewrite weak areas, repeat until good enough."""
    critique_tool = PromptCritiqueTool()
    current_script = str(script_data.get("script", ""))
    current_narration = str(script_data.get("narration", ""))

    all_rounds = []

    for round_num in range(MAX_CRITIQUE_ROUNDS):
        print(f"  Critique round {round_num + 1}/{MAX_CRITIQUE_ROUNDS}...")
        args = PromptCritiqueArgs(
            video_prompt=current_script,
            narration=current_narration,
            topic=query,
        )
        result = await critique_tool.run(args, None)
        all_rounds.append(result)

        scores = result.get("scores", {})
        avg = sum(scores.values()) / len(scores) if scores else 0
        min_score = min(scores.values()) if scores else 0
        issues = result.get("issues", [])

        print(f"    Avg={avg:.1f}  Min={min_score}  Issues={len(issues)}")
        for dim, val in sorted(scores.items()):
            flag = " <--" if val < QUALITY_THRESHOLD else ""
            print(f"      {dim}: {val}{flag}")

        if avg >= QUALITY_THRESHOLD and min_score >= 5 and not issues:
            print(f"    Script passes quality threshold (avg={avg:.1f}, min={min_score})")
            break

        rewritten_script = result.get("rewritten_script", "")
        rewritten_narration = result.get("rewritten_narration", "")
        if rewritten_script:
            current_script = rewritten_script
        if rewritten_narration:
            current_narration = rewritten_narration
        print(f"    Applied rewrite from round {round_num + 1}")

    return {
        "script": current_script,
        "narration": current_narration,
        "rounds": all_rounds,
        "final_scores": all_rounds[-1].get("scores", {}) if all_rounds else {},
    }

class VeoVideoGeneratorArgs(BaseModel):
    query: str
    iid: str
    duration_seconds: int = 8
    frame: Optional[bytes] = None

class VeoVideoGeneratorTool(BaseTool[VeoVideoGeneratorArgs, Dict[str, Any]]):
    def __init__(self):
        super().__init__(
            args_type=VeoVideoGeneratorArgs,
            return_type=Dict[str, Any],
            name="veo_video_generator",
            description="Generates videos using Veo API - fails immediately if generation fails"
        )

    async def run(self, args: VeoVideoGeneratorArgs, context):
        """Generate video with no fallbacks - fail fast approach"""
        try:
            operation_name = generate_video(
                args.query, 
                duration_seconds=args.duration_seconds,
                frame=args.frame
            )
            result = poll_operation(operation_name)
            
            if "error" in result:
                error_message = result["error"].get("message", "Unknown error")
                raise Exception(f"Video generation failed: {error_message}")
            
            videos = result["response"]["videos"]
            paths = []
            output_dir = os.path.join("test", "parts")
            os.makedirs(output_dir, exist_ok=True)
            
            for i, video in enumerate(videos):
                filename = os.path.join(output_dir, f"{args.iid}_{i}.mp4")
                
                if "gcsUri" in video:
                    download_video(video["gcsUri"], filename)
                else:
                    with open(filename, "wb") as f:
                        f.write(base64.b64decode(video["bytesBase64Encoded"]))
                
                paths.append(filename)
            
            # frame = extract_frame_bytes(paths[-1]) if paths else None
            frame = None 
            if frame:
                print("  ✓ Extracted last frame for next clip continuity")
            
            return {"success": True, "video_paths": paths, "frame": frame}
            
        except Exception as e:
            print(f"✗ Video generation failed: {str(e)}")
            raise

class PostProcessingArgs(BaseModel):
    video_paths: List[str]
    narrations: List[str]
    iid: str
    query: str
    crossfade_duration: float = 1.0

class PostProcessingAgent(BaseTool[PostProcessingArgs, str]):
    """Dedicated agent for all post-processing: TTS, transitions, and final assembly"""
    def __init__(self):
        super().__init__(
            args_type=PostProcessingArgs,
            return_type=str,
            name="post_processing_agent",
            description="Handles TTS generation, audio replacement, transitions, and final video assembly"
        )

    def elevenlabs_tts_to_file(self, text, out_path, voice_settings=None):
        """Generate TTS audio file"""
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_turbo_v2_5",
            text=text,
            voice_settings=voice_settings,
        )
        with open(out_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        return out_path

    def replace_audio_with_tts(self, video_path, narration_text, audio_path, voice_settings):
        """Replace video audio with TTS narration"""
        self.elevenlabs_tts_to_file(narration_text, audio_path, voice_settings)

        clip = VideoFileClip(video_path)
        aclip = AudioFileClip(audio_path)

        if aclip.duration > clip.duration:
            final_audio = aclip.with_effects([vfx.MultiplySpeed(final_duration=clip.duration)])
        else:
            delayed = aclip.set_start(0.5)
            final_audio = CompositeAudioClip([delayed]).set_duration(clip.duration)

        return clip.set_audio(final_audio)

    async def run(self, args: PostProcessingArgs, context):
        """Execute complete post-processing pipeline"""
        print("\n" + "="*60)
        print("POST-PROCESSING AGENT STARTING")
        print("="*60)
        
        tmpdir = "test/audios"
        os.makedirs(tmpdir, exist_ok=True)
        output_dir = os.path.join("test", "final")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[1/3] Generating TTS narration for {len(args.video_paths)} clips...")
        consistent_voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
        
        clips = []
        for i, (video_path, narration) in enumerate(zip(args.video_paths, args.narrations)):
            audio_path = os.path.join(tmpdir, f"{args.iid}_{i}.mp3")
            print(f"  • Clip {i+1}/{len(args.video_paths)}: Generating TTS...")
            clip_with_audio = self.replace_audio_with_tts(
                video_path, 
                narration, 
                audio_path, 
                consistent_voice_settings
            )
            clips.append(clip_with_audio)
            print(f"    ✓ TTS added to clip {i+1}")
        
        print(f"\n[2/3] Applying {args.crossfade_duration}s crossfade transitions...")
        xfaded = [clips[0]] + [
            c.with_effects([vfx.CrossFadeIn(args.crossfade_duration)]) 
            for c in clips[1:]
        ]
        print(f"  ✓ Transitions applied to {len(clips)-1} clip boundaries")
        
        print(f"\n[3/3] Assembling and exporting final video...")
        final_clip = concatenate_videoclips(
            xfaded,
            method="compose",
            padding=-args.crossfade_duration
        )
        
        query_safe = re.sub(r'[^A-Za-z0-9_]+', '', args.query.replace(" ", "_"))
        output_path = os.path.join(output_dir, f"{args.iid}.mp4")
        
        final_clip.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            threads=0
        )
        
        print("\n" + "="*60)
        print(f"✓ POST-PROCESSING COMPLETE")
        print(f"✓ Final video: {output_path}")
        print("="*60 + "\n")
        
        return output_path

async def run_pipeline(query: str, iid: int):  
    iid = str(iid)   
    web_retriever = WebKnowledgeRetrieverTool()
    image_retriever = ImageRetrieverTool()
    script_generator = ScriptGeneratorTool()
    simple_critique = PromptCritiqueTool()
    veo_tool = VeoVideoGeneratorTool()
    post_processor = PostProcessingAgent()
    
    print("[STEP 1/5] Retrieving web knowledge...")
    retriever_args = WebKnowledgeRetrieverArgs(query=query, detail_level="comprehensive")
    web_context = await web_retriever.run(retriever_args, None)
    
    out_dir = Path("test") / "0_web"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{iid}_web.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(web_context)

    # print("[STEP 2/5] Retrieving reference images...")
    # reference_images = await image_retriever.run(ImageRetrieverArgs(query=query, iid=iid, pynum_images=1), None)
    # if not reference_images:
    #     raise Exception("No reference images were retrieved")

    # # # TEST IMAGES
    # # reference_images = Path("test/2_unique/unique3__0.jpg")
    # # if isinstance(reference_images, (str, Path)):
    # #     image_path = Path(reference_images)
    # #     reference_images = [{
    # #         "url": str(image_path),
    # #         "bytes": image_path.read_bytes(),
    # #         "index": 0,
    # #         "relevance_score": 10,
    # #     }]

    # frame_image = PILImage.open(io.BytesIO(reference_images[0]["bytes"])).convert("RGB")
    # frame_buffer = io.BytesIO()
    # frame_image.save(frame_buffer, format="JPEG", quality=95)
    # frame = frame_buffer.getvalue()
    # ref_dir = Path("test/2_images")
    # ref_dir.mkdir(parents=True, exist_ok=True)
    # for ref in reference_images:
    #     url = str(ref.get("url", ""))
    #     m = re.search(r"\.(jpg|jpeg|png|gif|webp)(?:$|[?#])", url, flags=re.IGNORECASE)
    #     ext = (m.group(1).lower() if m else "jpg")
    #     img_bytes = ref["bytes"]
    #     idx = ref.get("index", 0)
    #     safe_url = re.sub(r"[^A-Za-z0-9_]+", "", url)[:30]
    #     out_name = f"{iid}.{ext}"
    #     out_path = ref_dir / out_name
    #     with open(out_path, "wb") as imgf:
    #         imgf.write(img_bytes)
    
    print("[STEP 3/5] Generating script...")
    generator_args = ScriptGeneratorArgs(query=query, context_info=web_context)
    raw = await script_generator.run(generator_args, None)
    script = clean_json_block(raw)
    data = json.loads(script)
    print("data", data)

    out_dir = Path("test") / "1_script"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{iid}_script.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(script)

    # # TEST DATA
    # file_path = Path("test") / "1_script" / "1_script.txt"
    # with open(file_path, "r", encoding="utf-8") as f:
    #     script = f.read()
    # data = json.loads(script)

    print("[STEP 4/5] Running iterative critique...")
    critique_result = await run_iterative_critique(query, data)
    data = {
        "script": critique_result["script"],
        "narration": critique_result["narration"],
    }
    out_dir = Path("test") / "2_critique"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{iid}_critique.txt"
    print(f"Critiqued script: {data['script'][:120]}...")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "final": data,
            "scores": critique_result["final_scores"],
            "rounds": critique_result["rounds"],
        }, indent=2))
    
    # print("[STEP 5/5] Generating videos...")
    # generated_paths = []
    # narrations = []

    # if not data:
    #     raise Exception("No validated prompts to generate video")

    # script = str(data["script"]).strip()
    # narration = str(data["narration"]).strip()

    # try:
    #     veo_args = VeoVideoGeneratorArgs(
    #         query=script,
    #         iid=iid,
    #         duration_seconds=8,
    #         frame=frame
    #     )
    #     veo_result = await veo_tool.run(veo_args, None)

    #     generated_paths.extend(veo_result["video_paths"])
    #     narrations.append(narration)

    # except Exception as e:
    #     print(f"  ✗ Video generation failed: {str(e)}")
    #     print(f"  Pipeline halted while generating the video\n")
    #     raise

    # if not generated_paths:
    #     raise Exception("No videos were successfully generated")
        
    # print("[STEP 6] Post-processing...")
    # post_args = PostProcessingArgs(
    #     video_paths=generated_paths,
    #     narrations=narrations,
    #     iid=iid,
    #     query=query,
    #     crossfade_duration=0.0
    # )
    # final_video_path = await post_processor.run(post_args, None)
    
    # return final_video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Video topic query")
    parser.add_argument("iid", type=int, help="Unique identifier for this generation")
    args = parser.parse_args()
    
    try:
        result = asyncio.run(run_pipeline(args.query, args.iid))
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"Final video: {result}")
        print('='*60)
    except Exception as e:
        print(f"\n{'='*60}")
        print("PIPELINE FAILED")
        print(f"Error: {str(e)}")
        print('='*60)
        sys.exit(1)
