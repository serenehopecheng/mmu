import os, re, asyncio, requests, nest_asyncio, openai, subprocess, tempfile
import sys, io
import subprocess, time, uuid, base64
import hashlib
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(requests.RequestException)
)
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
        print("    Using reference image for visual grounding")

    response = requests.post(url, headers=headers, json=data)
    print(f"    Veo API response: {response.status_code}")
    
    response_json = response.json()
    
    if "error" in response_json:
        raise Exception(f"Veo API Error: {response_json['error']['message']}")

    return response_json["name"]

def poll_operation(operation_name, max_wait_minutes=10):
    """Poll Veo operation with progress reporting and timeout"""
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:fetchPredictOperation"
    data = {"operationName": operation_name}
    
    start_time = time.time()
    poll_count = 0
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        access_token = get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            result = response.json()
        except requests.RequestException as e:
            print(f"    Poll request failed: {e}, retrying...")
            time.sleep(10)
            continue
        
        poll_count += 1
        elapsed = time.time() - start_time
        
        if result.get("done"):
            if "response" in result and "videos" in result["response"]:
                print(f"    Video generation complete after {elapsed:.0f}s ({poll_count} polls)")
                return result
            elif "error" in result:
                raise Exception(f"Veo generation failed: {result['error']}")
        
        if elapsed > max_wait_seconds:
            raise Exception(f"Veo generation timed out after {max_wait_minutes} minutes")
        
        if poll_count % 2 == 0:
            print(f"    Generating... ({elapsed:.0f}s elapsed)")
        
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

class SemanticCache:
    """Cache for expensive API calls with semantic key hashing"""
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        path = self.cache_dir / f"{self._hash_key(key)}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None
    
    def set(self, key: str, value: Any) -> None:
        path = self.cache_dir / f"{self._hash_key(key)}.json"
        path.write_text(json.dumps(value, default=str))
    
    def get_bytes(self, key: str) -> Optional[bytes]:
        path = self.cache_dir / f"{self._hash_key(key)}.bin"
        if path.exists():
            return path.read_bytes()
        return None
    
    def set_bytes(self, key: str, value: bytes) -> None:
        path = self.cache_dir / f"{self._hash_key(key)}.bin"
        path.write_bytes(value)
semantic_cache = SemanticCache()
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, aiohttp.ClientError))
)
def robust_request(url: str, **kwargs) -> requests.Response:
    """HTTP request with automatic retry and exponential backoff"""
    return requests.get(url, **kwargs)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((openai.APIError, openai.RateLimitError))
)
def robust_openai_call(func, *args, **kwargs):
    """OpenAI API call with automatic retry"""
    return func(*args, **kwargs)

def _strip_metadata_json_block(text: str) -> str:
    """Remove 'The following metadata...' plus the following balanced {...} (nested-safe)."""
    key = "The following metadata was extracted from the webpage:"
    while True:
        i = text.find(key)
        if i < 0:
            break
        j = i + len(key)
        while j < len(text) and text[j] in " \t\n\r":
            j += 1
        if j >= len(text) or text[j] != "{":
            break
        depth = 0
        k = j
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        text = text[:i] + text[k:]
    return text

def _strip_empty_relevant_context_sections(text: str) -> str:
    """Drop ## Relevant Context sections whose body is empty after cleaning."""
    pattern = re.compile(r"^## Relevant Context \d+\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    out = []
    preamble = text[: matches[0].start()].rstrip()
    if preamble:
        out.append(preamble)
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            out.append(text[m.start() : end].rstrip())
    return "\n\n".join(out).strip()

def _sanitize_web_surfer_text(text: str) -> str:
    """Remove Playwright tracebacks, web surfer errors, image reprs, and Bing boilerplate from MultimodalWebSurfer output."""
    if not text:
        return text
    text = _strip_metadata_json_block(text)
    text = re.sub(r"Here is a screenshot of the page\.\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<autogen_core\._image\.Image object at 0x[0-9a-f]+>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<PIL\.(?:Image\.)?Image object at 0x[0-9a-f]+>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Full "Web surfing error" + traceback ending in Playwright error + Call log
    text = re.sub(
        r"Web surfing error:\s*\n+Traceback \(most recent call last\):[\s\S]+?"
        r"playwright\._impl\._errors\.Error:[^\n]+\n(?:Call log:\s*\n[^\n]*\n[^\n]*)?\s*",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"Traceback \(most recent call last\):[\s\S]+?"
        r"playwright\._impl\._errors\.Error:[^\n]+\n(?:Call log:\s*\n[^\n]*\n[^\n]*)?\s*",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"playwright\._impl\._errors\.Error: Page\.goto:[^\n]+\n(?:Call log:\s*\n[^\n]*\n[^\n]*)?\s*",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Truncated / copy-pasted tracebacks without a closing Error line
    text = re.sub(
        r"(?:Web surfing error:\s*\n+)?Traceback \(most recent call last\):[\s\S]*?(?=\n## Relevant Context |\n# Knowledge Summary|\Z)",
        "",
        text,
    )
    text = re.sub(r"^\)\s*$\n\s*\^\s*$\n(?:^\s*File \"[\s\S]*?$)+", "", text, flags=re.MULTILINE)
    bad_substrings = (
        "site-packages/playwright",
        "site-packages/autogen_ext",
        "multimodal_web_surfer.py",
        "_multimodal_web_surfer.py",
        "playwright/async_api/_generated.py",
        "playwright/_impl/_page.py",
        "playwright/_impl/_frame.py",
        "playwright/_impl/_connection.py",
        "rewrite_error",
        "locals_to_params",
        "parsed_st['apiName']",
    )
    bad_line_res = (
        re.compile(r"^\s*raise\s+"),
        re.compile(r"^\s*await self\._"),
        re.compile(r"^\s*return await self\._"),
        re.compile(r"^\s*\^+\s*$"),
        re.compile(r"\.\.\.<\d+ lines>\.\.\."),
    )
    lines = []
    for line in text.splitlines():
        if any(s in line for s in bad_substrings):
            continue
        if any(r.search(line) for r in bad_line_res):
            continue
        if line.strip() in (")", "^") and len(line.strip()) <= 1:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"^\}\s*$", "", text, flags=re.MULTILINE)
    # WebSurfer agent narration about the search box / current URL (not topical content)
    text = re.sub(
        r"I typed '[^']+' into 'Enter your search here[^']*'\.\s*The web browser is open to the page \[[^\]]+\]\([^)]+\)\.?\s*",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_empty_relevant_context_sections(text)
    return text.strip()

def _flatten_surfer_content(content) -> str:
    """Turn multimodal surfer content into plain text without Image repr strings."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, AutogenImage):
                continue
            if isinstance(item, PILImage.Image):
                continue
            if isinstance(item, str):
                parts.append(item)
            else:
                s = str(item)
                if "Image object at 0x" in s:
                    continue
                parts.append(s)
        return " ".join(parts)
    return str(content)

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
            description="Search and browse the web to gather comprehensive knowledge for video generation.",
            start_page="about:blank",
        )

    async def _search_single(self, query: str, context) -> str:
        """Execute a single search query"""
        print(f"  WebSurfer searching: {query}")
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
        
        content = _flatten_surfer_content(content)
        return _sanitize_web_surfer_text(str(content))

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into semantic chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(sentence)
            current_size += len(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _deduplicate_chunks(self, chunks: List[str], similarity_threshold: float = 0.8) -> List[str]:
        """Remove near-duplicate chunks based on word overlap"""
        if not chunks:
            return []
        
        unique_chunks = [chunks[0]]
        
        for chunk in chunks[1:]:
            chunk_words = set(chunk.lower().split())
            is_duplicate = False
            
            for existing in unique_chunks:
                existing_words = set(existing.lower().split())
                if not chunk_words or not existing_words:
                    continue
                overlap = len(chunk_words & existing_words) / max(len(chunk_words), len(existing_words))
                if overlap > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks

    async def _rank_by_relevance(self, chunks: List[str], query: str, top_k: int = 10) -> List[str]:
        """Rank chunks by relevance to query using embeddings"""
        if not chunks:
            return []
        
        try:
            query_response = robust_openai_call(
                openai_client.embeddings.create,
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = np.array(query_response.data[0].embedding)
            
            scored_chunks = []
            for chunk in chunks:
                truncated = chunk[:8000]
                chunk_response = robust_openai_call(
                    openai_client.embeddings.create,
                    model="text-embedding-3-small",
                    input=truncated
                )
                chunk_embedding = np.array(chunk_response.data[0].embedding)
                score = np.dot(query_embedding, chunk_embedding)
                scored_chunks.append((score, chunk))
            
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in scored_chunks[:top_k]]
        except Exception as e:
            print(f"  Warning: Embedding ranking failed ({e}), using original order")
            return chunks[:top_k]

    async def run(self, args: WebKnowledgeRetrieverArgs, context):
        cache_key = f"web:{args.query}:{args.detail_level}"
        cached = semantic_cache.get(cache_key)
        if cached:
            print(f"  Using cached web knowledge for: {args.query}")
            return cached
        
        search_queries = [
            f"{args.query} detailed explanation",
            f"{args.query} tutorial guide",
            f"{args.query} step by step process",
            f"how {args.query} works visual guide"
        ]
        
        print(f"  Running {len(search_queries)} parallel web searches...")
        tasks = [self._search_single(q, context) for q in search_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_knowledge = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  Search {i+1} failed: {result}")
                continue
            all_knowledge.append(result)
        
        combined_text = _sanitize_web_surfer_text("\n\n".join(all_knowledge))
        chunks = self._chunk_text(combined_text)
        unique_chunks = self._deduplicate_chunks(chunks)
        print(f"  Chunked into {len(chunks)} pieces, {len(unique_chunks)} unique")
        
        ranked_chunks = await self._rank_by_relevance(unique_chunks, args.query, top_k=15)
        
        structured_knowledge = self._format_structured_knowledge(ranked_chunks, args.query)
        
        semantic_cache.set(cache_key, structured_knowledge)
        
        print(f"  WebSurfer complete: {len(structured_knowledge)} chars of ranked knowledge")
        return structured_knowledge

    def _format_structured_knowledge(self, chunks: List[str], query: str) -> str:
        """Format ranked chunks into structured knowledge"""
        sections = []
        sections.append(f"# Knowledge Summary: {query}\n")
        n = 1
        for chunk in chunks:
            c = _sanitize_web_surfer_text(chunk).strip()
            if not c:
                continue
            sections.append(f"## Relevant Context {n}\n{c}\n")
            n += 1
        return "\n".join(sections)

class ImageRetrieverArgs(BaseModel):
    query: str
    iid: str
    num_images: int = 1 

class ImageRetrieverTool(BaseTool[ImageRetrieverArgs, list]):
    MAX_URLS_PER_ENGINE = 8
    MAX_DOWNLOAD_CONCURRENT = 20
    ACCEPT_SCORE_THRESHOLD = 6
    EARLY_EXIT_SCORE = 8
    BATCH_VALIDATION_SIZE = 4

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
        resp = robust_request(url, headers=headers, timeout=15)
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

    async def _download_image_async(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
        """Download a single image asynchronously"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    if len(content) >= 5000:
                        return {"url": url, "bytes": content}
        except Exception:
            pass
        return None

    async def _download_all_images(self, urls: List[str]) -> List[Dict]:
        """Download all images concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._download_image_async(session, url) for url in urls[:self.MAX_DOWNLOAD_CONCURRENT]]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

    def _batch_validate_images(self, images: List[Dict], query: str) -> List[Dict]:
        """Validate multiple images in batches to reduce API calls"""
        validated = []
        
        for batch_start in range(0, len(images), self.BATCH_VALIDATION_SIZE):
            batch = images[batch_start:batch_start + self.BATCH_VALIDATION_SIZE]
            
            image_contents = []
            for i, img in enumerate(batch):
                img_b64 = base64.b64encode(img["bytes"]).decode('utf-8')
                image_contents.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            validation_prompt = f"""Analyze these {len(batch)} images for the query: "{query}"

            These images will be used as VISUAL REFERENCES for video generation. We need images that show what the subject LOOKS LIKE.

            ACCEPT images that:
            - Show real-world subjects, objects, people, or scenes 
            - Are photographs or photo-realistic images (not pure illustrations/cartoons)
            - Are relevant to the query topic
            - Have acceptable visual quality

            If there are several watermarks on an image, remove it.

            Respond with JSON array only, one object per image in order:
            [{{"index": 0, "is_valid": true/false, "reason": "brief", "relevance_score": 0-10}}, ...]"""
       

            try:
                response = robust_openai_call(
                    openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": validation_prompt}] + image_contents
                    }],
                    max_completion_tokens=300
                )
                
                result_text = response.choices[0].message.content.strip()
                result_text = clean_json_block(result_text)
                validations = json.loads(result_text)
                
                for v in validations:
                    idx = v.get("index", 0)
                    if idx < len(batch):
                        if v.get("is_valid") and v.get("relevance_score", 0) >= self.ACCEPT_SCORE_THRESHOLD:
                            img_data = batch[idx].copy()
                            img_data["relevance_score"] = v.get("relevance_score", 0)
                            img_data["index"] = len(validated)
                            validated.append(img_data)
                            print(f"  ✓ Validated image {len(validated)} (score: {v.get('relevance_score')}/10)")
                            
                            if v.get("relevance_score", 0) >= self.EARLY_EXIT_SCORE:
                                print(f"  Found strong match, stopping early")
                                return validated
                        else:
                            print(f"  ✗ Rejected: {v.get('reason', 'low score')}")
                            
            except Exception as e:
                print(f"  Batch validation error: {str(e)[:60]}")
                continue
        
        return validated

    async def run(self, args: ImageRetrieverArgs, context):
        cache_key = f"images:{args.query}:{args.num_images}"
        cached = semantic_cache.get(cache_key)
        if cached:
            print(f"  Using cached images for: {args.query}")
            for img in cached:
                if "bytes_b64" in img:
                    img["bytes"] = base64.b64decode(img["bytes_b64"])
            return cached
        
        search_queries = [
            f"{args.query} photo",
            f"{args.query} real",
            f"{args.query} how to",
        ]

        all_urls = []
        for sq in search_queries:
            print(f"  Searching images: {sq}")
            try:
                urls = self._scrape_google_images(sq, num_results=self.MAX_URLS_PER_ENGINE)
                print(f"    Google Images: {len(urls)} URLs")
                all_urls.extend(urls)
            except Exception as e:
                print(f"    Google Images failed: {str(e)[:80]}")
            try:
                urls = self._scrape_duckduckgo_images(sq, num_results=self.MAX_URLS_PER_ENGINE)
                print(f"    DuckDuckGo Images: {len(urls)} URLs")
                all_urls.extend(urls)
            except Exception as e:
                print(f"    DuckDuckGo Images failed: {str(e)[:80]}")

        all_urls = list(dict.fromkeys(all_urls))
        print(f"  Found {len(all_urls)} unique candidate URLs")

        print(f"  Downloading images concurrently...")
        downloaded = await self._download_all_images(all_urls)
        print(f"  Downloaded {len(downloaded)} images (>5KB)")

        unique_dir = Path("test/2_unique")
        unique_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(downloaded):
            try:
                url = img.get("url", "")
                m = re.search(r"\.(jpg|jpeg|png|gif|webp)(?:$|[?#])", url, flags=re.IGNORECASE)
                ext = (m.group(1).lower() if m else "jpg")
                filename = unique_dir / f"unique{args.iid}_{i}.{ext}"
                with open(filename, "wb") as f:
                    f.write(img["bytes"])
            except Exception:
                pass

        print(f"  Batch validating images...")
        reference_images = self._batch_validate_images(downloaded, args.query)
        
        if not reference_images and downloaded:
            print("  No validated images, using first downloaded as fallback")
            reference_images = [{
                "url": downloaded[0]["url"],
                "bytes": downloaded[0]["bytes"],
                "index": 0,
                "relevance_score": 0
            }]
        
        reference_images.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        reference_images = reference_images[:args.num_images]
        
        cache_data = []
        for img in reference_images:
            cache_entry = img.copy()
            cache_entry["bytes_b64"] = base64.b64encode(img["bytes"]).decode()
            del cache_entry["bytes"]
            cache_data.append(cache_entry)
        semantic_cache.set(cache_key, cache_data)
        
        print(f"  Successfully validated {len(reference_images)} reference images")
        return reference_images

class ScriptGeneratorArgs(BaseModel):
    query: str
    context_info: Optional[str] = None
    reference_image_bytes: Optional[bytes] = None

class ScriptGeneratorTool(BaseTool[ScriptGeneratorArgs, str]):
    FEW_SHOT_EXAMPLES = """
    === EXAMPLE 1 ===
    INPUT: "A person unboxing the latest iPhone 16 Pro in Desert Titanium on a wooden table, front view."
    OUTPUT:
    {
        "script": "[00:00-00:02] Top-down close-up of the iPhone 16 Pro resting on a dark wooden table. The Desert Titanium finish catches soft studio light, emphasizing its metallic texture and triple-lens camera bump. [00:02-00:04] Side-angle shot as a hand lifts the phone, highlighting its thin profile. The screen reflects a subtle gradient from the overhead lighting. [00:04-00:08] Slow rotation shot as the phone is turned to show the camera array, with light glinting off the sapphire lens covers."
    }

    === EXAMPLE 2 ===
    INPUT: "A close-up of a Tesla Cybertruck driving through a puddle, low angle."
    OUTPUT:
    {
        "script": "[00:00-00:03] Low-angle ground shot. The sharp, geometric stainless-steel body of a Cybertruck approaches the camera, reflecting the environment on its flat panels.\\n[00:03-00:06] Slow-motion close-up on the front tire as it strikes a large puddle. Water explodes outward in a dramatic arc, showing massive displacement.\\n[00:06-00:08] Tracking shot following the truck as it accelerates smoothly through the resistance, water trailing from the tires."
    }

    === EXAMPLE 3 ===
    INPUT: "A person opening a bottle of Coca-Cola with the new attached tethered cap, side view."
    OUTPUT:
    {
        "script": "[00:00-00:03] Side-view medium shot of a person holding a Coca-Cola bottle. Fingers grip the cap and twist it counterclockwise.\\n[00:03-00:05] Close-up on the cap. It separates from the ring but remains attached via a plastic hinge, bending back to rest against the side of the bottle.\\n[00:05-00:08] The person tilts the bottle to take a drink, showing how the hinged mechanism prevents the cap from detaching or getting lost."
    }
    """

    def __init__(self):
        super().__init__(
            args_type=ScriptGeneratorArgs,
            return_type=str,
            name="script_generator",
            description="Generates scripts for short, cinematic videos with narration."
        )

    async def _describe_reference_image(self, image_bytes: bytes) -> str:
        """Use GPT-4o to describe a reference image for script generation"""
        try:
            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
            response = robust_openai_call(
                openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """Describe this image in detail for video generation. Focus on:
                        - Colors and color palette (specific: "dusty terracotta", not "orange")
                        - Lighting quality and direction (soft diffused, harsh directional, golden hour, etc.)
                        - Textures and materials visible
                        - Composition and spatial arrangement
                        - Mood and atmosphere
                        - Any motion that could be implied

                        Keep description to 100-150 words, highly specific and visual."""},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                max_completion_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Warning: Image description failed: {e}")
            return ""

    async def run(self, args: ScriptGeneratorArgs, context):
        image_description = ""
        if args.reference_image_bytes:
            print("  Analyzing reference image for visual details...")
            image_description = await self._describe_reference_image(args.reference_image_bytes)
            if image_description:
                print(f"  Reference image described: {len(image_description)} chars")

        context_truncated = args.context_info[:4000] if args.context_info else "No additional context."
        ref_block = (
            f"Reference image (ground truth for look/layout): {image_description}\n        "
            if image_description
            else ""
        )

        prompt = f"""You are a script writer tasked with producing a Veo 3-shot video script for the given {args.query}. 
        Few shot examples: {self.FEW_SHOT_EXAMPLES}
        {ref_block}Background information: {context_truncated}
        Output valid JSON: {{"script": "...", "narration": "..."}}.
        Required rules (encode missing patterns from the examples as explicit requirements):
        - Topic fidelity: The three shots must directly illustrate the exact topic described by {args.query}; no tangents or unrelated visuals. The {args.query} wording should anchor the entire sequence.
        - Exactly three shots totaling 8 seconds: Create a three-shot sequence that sums to 8 seconds in total.
        - [00:00-00:02] Shot 1
        - [00:02-00:05] Shot 2
        - [00:05-00:08] Shot 3
        - Script format and cadence: In the "script" value, provide three clearly timestamped lines, each starting with the exact time bracket, followed by the shot label and a concise, camera-focused description. Example structure (adapt to your topic):
        - [00:00-00:02] Shot 1: [scene/context] - Camera: [angle], Framing: [distance], Movement: [type], Subject: [single].
        - [00:02-00:05] Shot 2: [scene/context] - Camera: [angle], Framing: [distance], Movement: [type], Subject: [single].
        - [00:05-00:08] Shot 3: [scene/context] - Camera: [angle], Framing: [distance], Movement: [type], Subject: [single].
        - Camera specificity: Each shot must specify explicit camera angles/distance/movement (e.g., wide shot, eye-level tracking, close-up; distance: wide/medium/close; movement: static/slow tracking). No vague language.
        - Single-subject focus: The sequence must keep a single subject in frame across all shots; no two-shot or cutaways to additional subjects.
        - Narration alignment: The "narration" should be less than 18 words and add context to the video.
        - Style: Use present tense, concrete imagery, and concise phrasing. Avoid extraneous or meta-language.
        - JSON integrity: The output must be valid JSON with exactly the keys "script" and "narration". Do not include any text outside the JSON.
        - Placeholder integrity: Do not replace {args.query} with synonyms or altered phrasing; use the placeholder as the basis for the content."""

        response = robust_openai_call(
            openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()

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

    GOLDEN STANDARD FORMAT:
    Scripts must use time-stamped segments with specific camera angles and concrete details:
    
    Example 1 (Product showcase):
    "[00:00-00:02] Top-down close-up of the iPhone 16 Pro resting on a dark wooden table. The Desert Titanium finish catches soft studio light, emphasizing its metallic texture and triple-lens camera bump.
    [00:02-00:04] Side-angle shot as a hand lifts the phone, highlighting its thin profile. The screen reflects a subtle gradient from the overhead lighting.
    [00:04-00:08] Slow rotation shot as the phone is turned to show the camera array, with light glinting off the sapphire lens covers."
    
    Example 2 (Action shot):
    "[00:00-00:03] Low-angle ground shot. The sharp, geometric stainless-steel body of a Cybertruck approaches the camera, reflecting the environment on its flat panels.
    [00:03-00:06] Slow-motion close-up on the front tire as it strikes a large puddle. Water explodes outward in a dramatic arc, showing massive displacement.
    [00:06-00:08] Tracking shot following the truck as it accelerates smoothly through the resistance, water trailing from the tires."

    SCORING DIMENSIONS:
    1. policy_compliance: No sensitive content, text overlays, graphics, animation, fantasy, or prohibited camera moves (zooms, whip pans, rack focus, scanning).
    2. visual_specificity: Names concrete materials, textures, colors, lighting source/quality, and spatial relationships. Avoids vague words like "beautiful", "amazing", "stunning".
    3. temporal_clarity: Uses time-stamped segments [00:00-00:03], [00:03-00:06], etc. The reader can picture exactly what happens in each segment.
    4. single_subject_focus: One subject performing one simple, observable action. No competing elements, no multi-step processes.
    5. camera_feasibility: Camera angle explicitly stated per segment (e.g. top-down view, tight close-up, side-angle shot, macro shot). Movement is stationary or slow dolly/pan only.
    6. narration_quality: One sentence, max 18 words. Adds insight or context beyond what is visible—not a description of the visuals.
    7. veo_compatibility: Likely to succeed with Veo 3. Penalize: abstract concepts shown literally, grid/pattern layouts, technical measurement setups, multiple scene transitions, complex hand manipulations.

    RULES:
    - The first shot [00:00-00:XX] MUST display the main subject directly. If the topic is "iPhone," the phone must be visible in the first shot—not a box or packaging. If "Cybertruck," the truck must appear immediately.
    - Use time-stamped segments: [00:00-00:03], [00:03-00:06], [00:06-00:08] (or similar 2-3 second intervals).
    - Start each segment with the camera angle/shot type (e.g. "Close-up of...", "Wide shot of...", "Macro shot of...").
    - Replace every vague adjective with a concrete visual detail (color, material, texture, light).
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
        "rewritten_script": "The improved 8-second script using time-stamped segments and golden standard format.",
        "rewritten_narration": "Improved narration, max 18 words.",
        "explanation": "What was changed and why."
    }}"""

        r = robust_openai_call(
            openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": critique_prompt}],
            max_completion_tokens=1200,
            response_format={"type": "json_object"}
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
            voice_id="hpp4J3VqNfWAUOO0d1Us",
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

async def run_pipeline(query: str, iid: int | str):
    iid = str(iid)
    start_time = time.time()
    
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

    print("\n[STEP 2/6] Retrieving reference images (async download + batch validation)...")
    reference_images = await image_retriever.run(
        ImageRetrieverArgs(query=query, iid=iid, num_images=1), 
        None
    )
    if not reference_images:
        raise Exception("No reference images were retrieved")

    frame_image = PILImage.open(io.BytesIO(reference_images[0]["bytes"])).convert("RGB")
    frame_buffer = io.BytesIO()
    frame_image.save(frame_buffer, format="JPEG", quality=95)
    frame = frame_buffer.getvalue()
    
    ref_dir = Path("test/2_images")
    ref_dir.mkdir(parents=True, exist_ok=True)
    for ref in reference_images:
        url = str(ref.get("url", ""))
        m = re.search(r"\.(jpg|jpeg|png|gif|webp)(?:$|[?#])", url, flags=re.IGNORECASE)
        ext = (m.group(1).lower() if m else "jpg")
        img_bytes = ref["bytes"]
        out_name = f"{iid}.{ext}"
        out_path = ref_dir / out_name
        with open(out_path, "wb") as imgf:
            imgf.write(img_bytes)
    print(f"  Best reference image score: {reference_images[0].get('relevance_score', 'N/A')}/10")
    
    print("\n[STEP 3/6] Generating script (GPT-4o + few-shot + reference image analysis)...")
    generator_args = ScriptGeneratorArgs(
        query=query, 
        context_info=web_context,
        reference_image_bytes=frame
    )
    raw = await script_generator.run(generator_args, None)
    script = clean_json_block(raw)
    data = json.loads(script)

    out_dir = Path("test") / "1_script"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{iid}_script.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Script generated: {len(data.get('script', ''))} chars")
    print(f"  Narration: {data.get('narration', '')[:80]}...")

    print("\n[STEP 4/6] Running iterative critique (score + rewrite loop)...")
    critique_result = await run_iterative_critique(query, data)
    data = {
        "script": critique_result["script"],
        "narration": critique_result["narration"],
    }
    
    out_dir = Path("test") / "2_critique"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{iid}_critique.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "final": data,
            "scores": critique_result["final_scores"],
            "rounds": critique_result["rounds"],
        }, indent=2))
    
    final_scores = critique_result.get("final_scores", {})
    avg_score = sum(final_scores.values()) / len(final_scores) if final_scores else 0
    print(f"  Final average score: {avg_score:.1f}/10")
    print(f"  Critiqued script saved")
    
    print("\n[STEP 5/6] Generating video with Veo 3...")
    generated_paths = []
    narrations = []

    if not data:
        raise Exception("No validated prompts to generate video")

    script_text = str(data["script"]).strip()
    narration = str(data["narration"]).strip()

    try:
        veo_args = VeoVideoGeneratorArgs(
            query=script_text,
            iid=iid,
            duration_seconds=8,
            frame=frame
        )
        veo_result = await veo_tool.run(veo_args, None)
        generated_paths.extend(veo_result["video_paths"])
        narrations.append(narration)
        print(f"  Generated {len(veo_result['video_paths'])} video(s)")

    except Exception as e:
        print(f"  Video generation failed: {str(e)}")
        raise

    if not generated_paths:
        raise Exception("No videos were successfully generated")
    # post_processor = PostProcessingAgent()
    # generated_paths = ["test/parts/1_0.mp4"]
    # narrations = ["Experience artistry and technology intertwined in the design of the iPhone 16 Pro."]
        
    print("\n[STEP 6/6] Post-processing (TTS + assembly)...")
    post_args = PostProcessingArgs(
        video_paths=generated_paths,
        narrations=narrations,
        iid=iid,
        query=query,
        crossfade_duration=0.0
    )
    final_video_path = await post_processor.run(post_args, None)
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Final video: {final_video_path}")
    print("="*60)
    
    return final_video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Video topic query")
    parser.add_argument("iid", type=str, help="Unique identifier for this generation")
    args = parser.parse_args()
    
    result = asyncio.run(run_pipeline(args.query, args.iid))
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"Final video: {result}")
    print('='*60)
