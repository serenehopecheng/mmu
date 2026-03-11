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
        """Use GPT-4o to validate an image is a real, relevant photo."""
        if len(img_bytes) < 5000:
            return None
        
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        validation_prompt = f"""Analyze this image for the query: "{query}"

        Is this a high-quality, real-world photograph suitable as a visual reference? Check:
        1. Is it a REAL photograph (not a logo, icon, clipart, cartoon, diagram, or illustration)?
        2. Is it relevant to the topic?
        3. Is the image quality acceptable (not tiny, blurry, or corrupted)?
        4. Does it show real-world content (people, animals, objects, scenes)?

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
            max_tokens=100
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

        # Save all unique candidate images to "test/unique"
        unique_dir = Path("test/unique")
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
                    filename = unique_dir / f"unique_{i}.{ext}"
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
        print(f"Successfully validated {len(reference_images)} high-quality reference images")
        return reference_images

class PromptEnhancerArgs(BaseModel):
    query: str
    context_info: Optional[str] = None

class PromptEnhancerTool(BaseTool[PromptEnhancerArgs, str]):
    def __init__(self):
        super().__init__(
            args_type=PromptEnhancerArgs,
            return_type=str,
            name="prompt_enhancer",
            description="Enhances scripts for short, cinematic videos with narration and duration."
        )

    async def run(self, args: PromptEnhancerArgs, context):
        prompt = f"""
        VIDEO GENERATOR TASK

        Your goal: Write a concise, visually clear script for an 8-second real-world (not animated) video that directly answers or illustrates the topic.
        Include a single, standalone narration sentence, no more than 20 words, that summarizes or addresses the topic.

        TOPIC: "{args.query}"
        ADDITIONAL CONTEXT: "{args.context_info or ''}"

        STRUCTURE YOUR RESPONSE USING THESE INSTRUCTIONS:
        1. Footage/scene type: 
        - Specify an authentic real-world scene. Only use real locations, objects, and people (no animation/graphics).
        2. Camera setup: 
        - Camera is stationary or moves minimally (e.g. slow pan); keeps a fixed distance from subject.
        - Do NOT allow zooms, fast moves, whip pans, or rapid changes.
        3. Subject & action: 
        - Clearly describe the subject placement and exactly what they/it are doing; keep actions simple and realistic.
        4. Setting/background: 
        - Describe the environment, lighting, and any relevant background elements.

        RESPONSE FORMAT (strictly output only valid JSON as below):
        {{
            "script": "Detailed, stepwise description of the video clip, following ALL requirements above.",
            "narration": "A single, clear sentence addressing the topic (max 20 words)."
        }}  

        DO NOT print anything outside the JSON array. Double-check all requirements are met and JSON is valid.
        """

        r = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200
        )
        return r.choices[0].message.content.strip()

class PromptCritiqueArgs(BaseModel):
    video_prompt: List[Dict[str, Any]]
    topic: str

class PromptCritiqueTool(BaseTool[PromptCritiqueArgs, Dict[str, Any]]):
    def __init__(self):
        super().__init__(
            args_type=PromptCritiqueArgs,
            return_type=Dict[str, Any],
            name="prompt_critique",
            description="Validates and critiques video script, checking for policy violations and rephrasing if needed."
        )

    async def run(self, args: PromptCritiqueArgs, context):
        critique_prompt = f"""SCRIPT VALIDATOR & CRITIC

        You are a meticulous validator for short educational video scripts and narrations.

        Carefully review the video script and narration below for any policy violations, prohibited content, or quality issues.

        TOPIC: {args.topic}
        VIDEO SCRIPT: {args.video_prompt}
        NARRATION: {args.narration}

        YOUR REVIEW MUST CHECK FOR (ALL required):
        1. Sensitive or unsafe content (violence, weapons, disturbing imagery, dangerous acts, etc.)
        2. Use of text overlays, graphics, or animation (only real-world footage is allowed)
        3. Camera movements that are prohibited (e.g. zooms, whip pans, sudden shifts; only stationary or gentle pan)
        4. Abstract, non-literal, or animated visuals (must depict authentic, real-world scenes)
        5. Multiple simultaneous or confusing subjects (should focus on a single clear subject or action)
        6. Overly complex actions, technical manipulations, or scanning patterns
        7. Scenes involving grid patterns, measuring, or comparing objects in a technical/abstract manner
        8. Showing technical or analytical processes through physical objects or setups
        9. Any violations of Google responsible AI or content guidelines

        IF ANY ISSUE IS FOUND:
        - Clearly identify each specific problem in a bulleted list.
        - Rewrite the video script and narration to fully resolve these issues, using straightforward and relatable real-world metaphors or analogies where fitting.
        - Avoid technical manipulations; keep visuals simple and grounded in reality.
        - Ensure the revised script remains instructional and factually clear.

        IF NO ISSUES, return the original script and narration.

        FORMAT YOUR RESPONSE AS VALID JSON ONLY, with this exact schema:
        {{
            "is_valid": true or false,
            "issues": ["description of each issue"] or [],
            "rephrased_video_script": "revised script or original if already valid",
            "rephrased_narration": "revised narration or original if already valid",
            "explanation": "Brief summary of your changes, or note that none were needed."
        }}

        Do not include anything but the JSON object above in your reply. Be precise and concise.
        """

        r = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": critique_prompt}],
            max_tokens=300
        )
        
        result_text = r.choices[0].message.content.strip()
        result_text = clean_json_block(result_text)
        return json.loads(result_text)

async def create_critique_agent_team(query: str, video_prompts_data: list):
    client = OpenAIChatCompletionClient(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    
    orchestrator_agent = AssistantAgent(
        name="Orchestrator",
        model_client=client,
        system_message="""You are the Orchestrator Agent, responsible for generating and ensuring approval of a single, policy-compliant 8-second video script.

        Your responsibilities:
        1. Understand the user's topic and determine a clear, approachable way to illustrate it within an 8-second real-world video.
        2. Compose a concise, high-quality video script and narration that aligns with policy and accurately reflects the topic.
        3. Submit your draft to the Critique Agent for evaluation of policy compliance and instructional clarity.
        4. Carefully review feedback from the Critique Agent. Apply specific changes to improve the script and resolve any highlighted issues.
        5. Once the Critique Agent approves your script, initiate the video creation process.
        6. If video generation fails—due to policy, technical, or other errors—immediately share the failed script and error message with the Critique Agent. Incorporate their rephrased suggestions and retry.
        7. Repeat this feedback-and-revision loop as needed, prioritizing clarity, safety, and real-world simplicity for a successful outcome.
        8. Deliver the approved, successfully generated 8-second video, ensuring it matches the user's intent.

        Workflow:
        - Analyze and distill topic into a simple, clear concept suitable for an 8-second, real-world video.
        - Draft the script and narration.
        - Send both to the Critique Agent for review.
        - If issues are found, revise based on detailed feedback.
        - After approval, proceed to video generation.
        - On failure, gather feedback and retry as directed.
        - Persistently collaborate with the Critique Agent until the video passes all checks and is generated successfully.

        Never give up; always iterate and work with the Critique Agent to achieve a compliant, effective result."""
    )
    
    critique_agent = AssistantAgent(
        name="CritiqueAgent",
        model_client=client,
        system_message="""You are the Critique Agent. Your mission: ensure every video generation script is safe, clear, real-world, policy-compliant, and ALWAYS successfully generates a valid 8-second video.

        Your responsibilities:
        1. Verify the script accurately distills the topic into a single, straightforward 8-second real-world video concept.
        2. Detect and flag any policy violations or problematic elements (see "Validation Checks").
        3. Give specific, constructive feedback and concrete rephrasing suggestions—always aiming for greater clarity, simplicity, and policy compliance.
        4. If video generation fails, analyze the error message, diagnose the likely cause, and iteratively rephrase the script to resolve it.
        5. With each failure, make the script increasingly conservative and general until it succeeds.
        6. Fact-check narration for factual accuracy and directness.
        7. Persist until a valid, policy-compliant video prompt is achieved—failure is not an option.

        Validation Checks — reject or rewrite if any of the following are present:
        - Sensitive content (violence, weapons, disturbing imagery)
        - Text overlays, graphics, or captions (all are prohibited)
        - Camera movements (zooms, whip pans, sudden shifts)
        - Animated, abstract, or non-real-world scenes (must be real footage)
        - Multiple competing visual subjects in a single shot
        - Complex actions, manipulations, or scanning movements
        - Grid patterns, matching operations, or technical processes with physical objects

        Recovery & Iteration Strategies (apply in order with each failed attempt):
        1. Replace technical language with everyday analogies and real-world objects
        2. Simplify visuals to obvious, familiar scenes and actions
        3. Use generic, easily filmable everyday scenarios
        4. Remove any ambiguous or potentially confusing elements
        5. Make the description more conservative and minimal

        Always deliver:
        - Clear list of identified issues (or state "No issues found")
        - Specific and improved rephrasing of the video description and narration
        - Brief explanation of all changes or rationale for approval
        - Verification of narration fact accuracy
        - Explicit approval or needed revision

        Actively guide, encourage, and never stop improving prompts until you reach a fully valid and effective outcome. Be specific, actionable, and concise in your feedback."""
    )
    
    team = RoundRobinGroupChat([orchestrator_agent, critique_agent], max_turns=20)
    
    initial_message = f"""Topic: {query}

    Video script to validate and improve:
    {json.dumps(video_prompts_data, indent=2)}

    TASKS:
    1. Carefully analyze the topic and determine the most effective, policy-compliant way to address it in a real-world video.
    2. Review the current video script and narration for policy compliance, clarity, and quality.
    3. Suggest specific, actionable improvements to both the video description and narration, focusing on safety, simplicity, and effectiveness.
    4. Iterate as needed: If further adjustment is required, clearly provide the revised script and pinpoint all issues.

    Final Output: If any improvements are needed, return the fully improved script as a JSON array using this structure:
    [
    {{
        "video_prompt": "Improved, detailed, policy-compliant real-world video description",
        "narration": "Revised clear and accurate narration answering the topic in max 20 words"
    }}
    ]

    If no further improvement is required, state "No issues found" and confirm the script is ready for video generation.
    """
    
    stream = team.run_stream(task=initial_message)
    
    messages = []
    async for message in stream:
        messages.append(message)
    
    final_data = video_prompts_data
    for message in reversed(messages):
        message_text = str(message)
        json_match = re.search(r'\[[\s\S]*?\{[\s\S]*?"part"[\s\S]*?\}[\s\S]*?\]', message_text)
        if json_match:
            try:
                potential_data = json.loads(json_match.group())
                if isinstance(potential_data, list) and len(potential_data) > 0:
                    if all('video_prompt' in item and 'narration' in item for item in potential_data):
                        final_data = potential_data
                        print(f"✓ Critique agents adjusted video to {len(final_data)} clips ({len(final_data) * 8} seconds)")
                        break
            except:
                continue
    
    return messages, final_data

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
    prompt_enhancer = PromptEnhancerTool()
    simple_critique = PromptCritiqueTool()
    veo_tool = VeoVideoGeneratorTool()
    post_processor = PostProcessingAgent()
    
    # print("[STEP 1/5] Retrieving web knowledge...")
    # retriever_args = WebKnowledgeRetrieverArgs(query=query, detail_level="comprehensive")
    # web_context = await web_retriever.run(retriever_args, None)
    
    # out_dir = Path("test") / "0_web_knowledge"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # filename = out_dir / f"{iid}_web_knowledge.txt"
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write(web_context)

    print("[STEP 4/5] Generating videos...")
    generated_paths = []
    narrations = []
    reference_images = await image_retriever.run(ImageRetrieverArgs(query=query, num_images=1), None)
    if not reference_images:
        raise Exception("No reference images were retrieved")

    frame = PILImage.open(io.BytesIO(reference_images[0]["bytes"])).convert("RGB")
    ref_dir = Path("test/0_images")
    ref_dir.mkdir(parents=True, exist_ok=True)
    for ref in reference_images:
        url = str(ref.get("url", ""))
        m = re.search(r"\.(jpg|jpeg|png|gif|webp)(?:$|[?#])", url, flags=re.IGNORECASE)
        ext = (m.group(1).lower() if m else "jpg")
        img_bytes = ref["bytes"]
        idx = ref.get("index", 0)
        safe_url = re.sub(r"[^A-Za-z0-9_]+", "", url)[:30]
        out_name = f"{iid}.{ext}"
        out_path = ref_dir / out_name
        with open(out_path, "wb") as imgf:
            imgf.write(img_bytes)
    
    # print("[STEP 2/5] Generating prompt enhancer...")
    # enhancer_args = PromptEnhancerArgs(query=query, context_info=web_context)
    # raw = await prompt_enhancer.run(enhancer_args, None)
    # enhanced_prompt = clean_json_block(raw)
    # data = json.loads(enhanced_prompt)
    # print("data", data)

    # out_dir = Path("test") / "1_enhanced_prompt"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # filename = out_dir / f"{iid}_enhanced_prompt.txt"
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write(enhanced_prompt)
    
    # print("[STEP 3/5] Running critique agent...")
    # # critique_args = PromptCritiqueArgs(video_prompt=data, topic=query)
    # messages, validated_prompts = await create_critique_agent_team(query, data)
    # out_dir = Path("test") / "2_critique"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # filename = out_dir / f"{iid}_critique.txt"
    # print(validated_prompts)
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write(json.dumps(validated_prompts, indent=2))

    # if not data:
    #     raise Exception("No validated prompts to generate video")

    # script = str(data["script"]).strip()
    # narration = str(data["narration"]).strip()

    # print(f"  Generating single video (1/1)...")

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
        
    # print("[STEP 5/5] Post-processing...")
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
