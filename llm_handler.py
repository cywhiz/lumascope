import os
import json
import re
import time
import asyncio
import httpx
import hashlib
from dotenv import load_dotenv

load_dotenv()

# REST Configuration
API_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"
API_KEY = os.getenv("CEREBRAS_API_KEY")

# Environment-aware storage (Vercel uses /tmp)
IS_VERCEL = os.environ.get("VERCEL") == "1"
BASE_DIR = "/tmp" if IS_VERCEL else os.getcwd()

LLM_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "llm_cache")
if not os.path.exists(LLM_CACHE_DIR):
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)

USE_CACHE = True
print(f"DEBUG: [Analysis] Cache status: {'Enabled' if USE_CACHE else 'DISABLED'}")

# Concurrency limit for AI inference
llm_semaphore = asyncio.Semaphore(3)

print(f"DEBUG: [Analysis] API Key: {'Detected' if API_KEY else 'MISSING'}")


def get_event_hash(event):
    """Generate a hash based on title and description to identify unique event content."""
    content = f"{event.get('title', '')}{event.get('description', '')}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def get_cached_summary(event_hash):
    if not USE_CACHE:
        return None
    path = os.path.join(LLM_CACHE_DIR, f"{event_hash}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None


def set_cached_summary(event_hash, data):
    if not USE_CACHE:
        return
    path = os.path.join(LLM_CACHE_DIR, f"{event_hash}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
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
            desc = e.get("description", "No description provided.")
            markdown_items.append(f"### ID: {eid}\n**T:** {title}\n**D:** {desc}")

        input_markdown = "\n\n---\n\n".join(markdown_items)

        system_message = "Expert Web3 event summarizer. Output ONLY clean JSON."
        prompt = f"""Summarize {len(batch)} events into ONE JSON object.
Format:
{{
  "ID": {{
    "s": ["Summary: ...", "Topics: ...", "Speakers: ..."],
    "r": ["Reason1", "Reason2", "Reason3"],
    "t": ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5"]
  }}
}}
Rules:
1. 's': 3 bullets (Detailed Event Summary, Key Topics and Technologies, Key Speakers).
2. 'r': 3 strategic reasons to attend.
3. 't': Distinct single words. NO compound words or camelCase.
4. Maximize density with professional tech/finance terms.
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
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        max_retries = 5
        retry_delay = 10  # Start with longer delay for 429
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    API_URL, json=payload, headers=headers, timeout=90
                )
                if resp.status_code == 429:
                    print(
                        f"DEBUG: [Analysis] 429 Rate Limit. Retry in {retry_delay}s... ({attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"DEBUG: [Analysis] Error: {e}. Retrying...")
                await asyncio.sleep(retry_delay)

        result = resp.json()
        content = result["choices"][0]["message"]["content"].strip()
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
            event["ai_summary"] = cached.get("s")
            event["top_reasons"] = cached.get("r")
            event["tags"] = cached.get("t")
        else:
            to_process.append(event)

    if not to_process:
        print("DEBUG: [Analysis] All events found in cache. Skipping AI calls.")
        return events

    # 2. Parallel batching for high-speed AI inference
    batch_size = 15
    batches = [
        to_process[i : i + batch_size] for i in range(0, len(to_process), batch_size)
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
        batch_results = await asyncio.gather(*tasks)

    # 3. Merge results and update cache
    master_map = {}
    for res in batch_results:
        master_map.update(res)

    for event in to_process:
        eid = str(event.get("id"))
        ai_data = master_map.get(eid)
        if ai_data:
            event["ai_summary"] = ai_data.get("s")
            event["top_reasons"] = ai_data.get("r")
            event["tags"] = ai_data.get("t")
            # Save to cache
            ehash = id_to_hash.get(eid)
            if ehash:
                set_cached_summary(ehash, ai_data)
        else:
            event["ai_summary"] = "Synthesis incomplete."
            event["top_reasons"] = ["N/A"]
            event["tags"] = []

    duration = time.time() - total_start
    print(f"DEBUG: [Analysis] Process complete. Total time: {duration:.2f}s")
    return events
