import os
import json
import re
import time
import asyncio
import httpx
import hashlib
import regex
from dotenv import load_dotenv

load_dotenv()

# REST Configuration
API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL_NAME = "gemini-3.1-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Environment-aware storage (Vercel uses /tmp)
IS_VERCEL = os.environ.get("VERCEL") == "1"
BASE_DIR = "/tmp" if IS_VERCEL else os.getcwd()

LLM_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "llm_cache")
if not os.path.exists(LLM_CACHE_DIR):
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)

USE_CACHE = True
SUMMARY_CACHE_VERSION = 5
PROMPT_VERSION = "compact-english-v10"
EVENTS_PER_BATCH = 10
MAX_EXCERPT_CHARS = 900
MAX_DENSE_SCRIPT_EXCERPT_CHARS = 650
LLM_CACHE_TTL_SECONDS = 30 * 86400
LLM_REQUESTS_PER_MINUTE = 15
print(f"DEBUG: [Analysis] Cache status: {'Enabled' if USE_CACHE else 'DISABLED'}")

# Concurrency limit for AI inference
llm_semaphore = asyncio.Semaphore(2)


class RequestRateLimiter:
    """Space request starts to stay within the provider RPM limit."""

    def __init__(self, requests_per_minute):
        self.interval = 60 / requests_per_minute
        self.next_allowed = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if delay:
            await asyncio.sleep(delay)


llm_rate_limiter = RequestRateLimiter(LLM_REQUESTS_PER_MINUTE)

print(f"DEBUG: [Analysis] API Key: {'Detected' if API_KEY else 'MISSING'}")


def get_event_hash(event):
    """Hash the exact compact LLM input and prompt version for safe cache reuse."""
    content = "|".join(
        (
            str(SUMMARY_CACHE_VERSION),
            PROMPT_VERSION,
            str(event.get("title") or ""),
            str(event.get("start_date") or ""),
            str(event.get("location") or ""),
            build_event_excerpt(event.get("description", "")),
        )
    )
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _is_dense_script(text):
    """Use a lower character cap where characters tend to tokenize densely."""
    letters = regex.findall(r"\p{L}", text)
    if not letters:
        return False
    dense = sum(
        regex.match(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Hangul}]", char)
        is not None
        for char in letters
    )
    return dense / len(letters) >= 0.3


def _truncate_at_boundary(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary >= limit // 2:
        cut = cut[: boundary + 1]
    return cut.rstrip(" ,;:-") + "…"


def build_event_excerpt(description):
    """Select a small, high-signal multilingual excerpt for one LLM event input."""
    if not description:
        return ""

    units = []
    for block in re.split(r"\n\s*\n", str(description)):
        for line in block.splitlines():
            line = re.sub(r"\s+", " ", line).strip(" -•*\t")
            if len(line) < 12 or re.fullmatch(r"(?:https?://\S+|www\.\S+)", line):
                continue
            units.append(line)

    compact = "\n".join(dict.fromkeys(units)).strip()
    limit = (
        MAX_DENSE_SCRIPT_EXCERPT_CHARS
        if _is_dense_script(compact)
        else MAX_EXCERPT_CHARS
    )
    return _truncate_at_boundary(compact, limit)


def build_local_fallback_summary(description):
    """Keep event cards useful when the configured LLM account is unavailable."""
    text = re.sub(r"\s+", " ", build_event_excerpt(description)).strip()
    words = text.split()
    if len(words) > 40:
        return " ".join(words[:40]).rstrip(" ,;:-") + "..."
    if len(words) <= 40:
        return text or "Event details are available on the listing."
    return " ".join(words[:50]).rstrip(" ,;:-") + "…"


def normalize_tags(tags):
    """Keep concise, non-generic topic tags returned by the model."""
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return []

    cleaned = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) == 3:
            break
    return cleaned


def normalize_ai_data(data):
    """Normalize optional model fields before rendering or caching them."""
    if not isinstance(data, dict):
        return {}
    return {**data, "t": normalize_tags(data.get("t"))}


def get_cached_summary(event_hash):
    if not USE_CACHE:
        return None
    path = os.path.join(LLM_CACHE_DIR, f"{event_hash}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cached_at = data.get("cached_at")
                if cached_at and time.time() - float(cached_at) > LLM_CACHE_TTL_SECONDS:
                    return None
                return data
        except:
            return None
    return None


def set_cached_summary(event_hash, data):
    if not USE_CACHE:
        return
    path = os.path.join(LLM_CACHE_DIR, f"{event_hash}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**data, "cached_at": time.time()}, f)
    except:
        pass


def extract_json(text):
    """Extract first JSON dictionary or array from text."""
    try:
        text = text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"^```(?:json)?", "", text)
                text = re.sub(r"```$", "", text).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text
    except:
        return text


async def process_batch_async(client, batch, batch_index, total_batches):
    """Process a single batch using httpx asynchronously."""
    async with llm_semaphore:
        print(
            f"DEBUG: [Analysis] Processing Batch {batch_index+1}/{total_batches} ({len(batch)} events)..."
        )

        markdown_items = []
        for e in batch:
            eid = str(e.get("id"))
            title = e.get("title", "Untitled")
            date = e.get("start_date")
            location = e.get("location")
            excerpt = build_event_excerpt(e.get("description", "")) or "No description provided."
            fields = [f"ID: {eid}", f"Title: {title}"]
            if date:
                fields.append(f"Date: {date}")
            if location:
                fields.append(f"Location: {location}")
            fields.append(f"Excerpt: {excerpt}")
            markdown_items.append("\n".join(fields))

        input_markdown = "\n\n---\n\n".join(markdown_items)

        system_message = "Concise factual event summarizer. Output only valid JSON."
        prompt = f"""Summarize {len(batch)} events into ONE JSON object.
Format:
{{
  "ID": {{
    "s": "One factual English paragraph of at most 40 words.",
    "r": "One factual English Why Attend sentence of at most 30 words.",
    "t": ["lowercase tag", "lowercase tag", "lowercase tag"]
  }}
}}
Rules:
1. Write 's' as one concise English summary paragraph, translating internally when needed.
2. State the event overview, intended audience, and material activities or format.
3. Include named organizers, hosts, sponsors, and confirmed speakers only when
   explicitly supplied. Do not state that speakers are absent when none are named.
4. Write 'r' separately as one concise sentence explaining why someone should attend,
   based only on explicit value. Do not prefix it with "Why attend:".
5. Use only supplied information. Ignore URLs, image captions, contacts, payment or
   registration instructions, and repeated boilerplate.
6. 's' is at most 40 words; 'r' is at most 30 words; 't' has one to three tags.
7. Tags are canonical filter keywords, not a list of every word found in the text.
   Choose the smallest useful set of broad, user-understandable categories that
   cover the event. Merge activities with the same audience, purpose, or domain
   into one umbrella keyword: for example, yoga, fitness, running, and wellness
   should become one appropriate canonical keyword, not four tags. Apply this
   same semantic consolidation to every domain; do not use a hardcoded category
   list, synonyms, parent-child pairs, or near-duplicates. Prefer one umbrella
   tag whenever multiple tags would mean nearly the same thing.
INPUT:
{input_markdown}
"""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        max_retries = 1 if IS_VERCEL else 5
        request_timeout = 20 if IS_VERCEL else 90
        retry_delay = 10  # Start with longer delay for 429
        for attempt in range(max_retries):
            try:
                await llm_rate_limiter.wait()
                resp = await client.post(
                    API_URL, json=payload, headers=headers, timeout=request_timeout
                )
                if resp.status_code == 429:
                    print(
                        f"DEBUG: [Analysis] 429 Rate Limit. Retry in {retry_delay}s... ({attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue
                if 400 <= resp.status_code < 500:
                    print(
                        f"DEBUG: [Analysis] Gemini rejected Batch {batch_index+1} "
                        f"with HTTP {resp.status_code}; returning raw events without AI fields."
                    )
                    return {}
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"DEBUG: [Analysis] Error: {e}. Retrying...")
                await asyncio.sleep(retry_delay)
        else:
            raise RuntimeError("Gemini rate limit retries exhausted")

        try:
            result = resp.json()
            content = result["choices"][0]["message"]["content"].strip()
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            print(f"DEBUG: [Analysis] Invalid response payload for Batch {batch_index+1}")
            return {}
        json_str = extract_json(content)

        try:
            # Use strict=False to handle control characters like raw newlines in strings
            data = json.loads(json_str, strict=False)
            return {str(k): v for k, v in data.items()}
        except Exception as e:
            print(f"DEBUG: [Analysis] JSON Error Batch {batch_index+1}: {e}")
            # Try once more with aggressive whitespace cleaning if it failed
            try:
                # Remove literal newlines/tabs inside quotes (very common LLM mistake)
                cleaned_str = re.sub(r"\n", " ", json_str)
                data = json.loads(cleaned_str, strict=False)
                return {str(k): v for k, v in data.items()}
            except:
                return {}


async def summarize_events(events):
    """
    Summarize events with caching and async batching.
    Only processes events not found in the local cache.
    """
    if not events:
        return events
    if not API_KEY:
        print("DEBUG: [Analysis] API key missing; using local summary fallback.")
        for event in events:
            event["ai_summary"] = build_local_fallback_summary(
                event.get("description", "")
            )
            event["top_reasons"] = []
            event["tags"] = []
        return events

    total_start = time.time()

    # 1. Identify which events need AI analysis
    to_process = []
    id_to_hash = {}

    for event in events:
        ehash = get_event_hash(event)
        eid = str(event.get("id"))
        id_to_hash[eid] = ehash

        cached = get_cached_summary(ehash)
        if cached:
            cached = normalize_ai_data(cached)
            event["ai_summary"] = cached.get("s")
            event["top_reasons"] = cached.get("r", [])
            event["tags"] = cached.get("t", [])
        else:
            to_process.append(event)

    if not to_process:
        print("DEBUG: [Analysis] All events found in cache. Skipping AI calls.")
        return events

    # 2. Parallel batching for high-speed AI inference
    batches = [
        to_process[i : i + EVENTS_PER_BATCH]
        for i in range(0, len(to_process), EVENTS_PER_BATCH)
    ]
    total_batches = len(batches)

    print(
        f"DEBUG: [Analysis] Analyzing {len(to_process)} events in {total_batches} batches..."
    )

    # Optimized client config for Windows socket stability
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=15)
    async with httpx.AsyncClient(timeout=100, limits=limits, trust_env=False) as client:
        # Re-enable parallel processing for maximum speed
        tasks = [
            process_batch_async(client, b, i, total_batches)
            for i, b in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Merge results and update cache
    master_map = {}
    for index, res in enumerate(batch_results):
        if isinstance(res, Exception):
            print(f"DEBUG: [Analysis] Batch {index + 1} failed: {type(res).__name__}: {res}")
            continue
        if isinstance(res, dict):
            master_map.update(res)

    for event in to_process:
        eid = str(event.get("id"))
        ai_data = master_map.get(eid)
        if ai_data:
            ai_data = normalize_ai_data(ai_data)
            event["ai_summary"] = ai_data.get("s")
            event["top_reasons"] = ai_data.get("r", [])
            event["tags"] = ai_data.get("t", [])
            # Save to cache
            ehash = id_to_hash.get(eid)
            if ehash:
                set_cached_summary(ehash, ai_data)
        else:
            event["ai_summary"] = build_local_fallback_summary(
                event.get("description", "")
            )
            event["top_reasons"] = []
            event["tags"] = []

    duration = time.time() - total_start
    print(f"DEBUG: [Analysis] Process complete. Total time: {duration:.2f}s")
    return events
