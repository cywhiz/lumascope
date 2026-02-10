import httpx
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import asyncio
import re
import regex
import hashlib
import os

# Environment-aware storage (Vercel uses /tmp)
IS_VERCEL = os.environ.get("VERCEL") == "1"
BASE_DIR = "/tmp" if IS_VERCEL else os.getcwd()

CACHE_DIR = os.path.join(BASE_DIR, "outputs", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

USE_CACHE = True
print(f"DEBUG: [Scraper] Cache status: {'Enabled' if USE_CACHE else 'DISABLED'}")

# Prevent socket exhaustion (ConnectError) on Windows
scraper_semaphore = asyncio.Semaphore(10)


def get_cache_path(url):
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.json")


def get_cached_data(url):
    if not USE_CACHE:
        return None
    path = get_cache_path(url)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None


def set_cached_data(url, data):
    if not USE_CACHE:
        return
    path = get_cache_path(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except:
        pass


def clean_description(text):
    if not text:
        return ""

    # 1. Truncate after keywords
    keywords = ["About", "Previous Events", "Contact"]
    for kw in keywords:
        idx = text.lower().find(kw.lower())
        if idx != -1:
            text = text[:idx]

    # 2. fast_clean logic
    if regex.search(r"\p{Latin}", text):
        text = regex.sub(r"\p{Han}+", "", text)

    text = regex.sub(r"\p{S}+|[^\x00-\x7f]+", " ", text)

    # 3. Truncate to 1500 characters
    if len(text) > 1500:
        text = text[:1500] + "..."

    # 4. Final whitespace cleanup
    text = regex.sub(r"\s+", " ", text).strip()
    return text


async def fetch_event_details(client, url):
    """
    Go through each url in the events field and get the full json,
    parse the description and organizer name fields.
    Using cache and httpx for speed.
    """
    async with scraper_semaphore:
        cached = get_cached_data(url)
        if cached:
            return cached.get("description")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://lu.ma/",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await client.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    ld_script = soup.find("script", type="application/ld+json")
                    description = None

                    if ld_script:
                        ld_data = json.loads(ld_script.string)
                        event_data = None
                        if isinstance(ld_data, list):
                            for item in ld_data:
                                if isinstance(item, dict) and (
                                    item.get("@type") == "Event"
                                    or "description" in item
                                ):
                                    event_data = item
                                    break
                        elif isinstance(ld_data, dict):
                            event_data = ld_data

                        if event_data:
                            description = clean_description(
                                event_data.get("description")
                            )

                            # Save to cache
                            set_cached_data(url, {"description": description})

                            return description

                    # If no JSON-LD, maybe it's a redirection or different structure
                    return None

                elif resp.status_code == 429:
                    wait = (attempt + 1) * 2
                    print(
                        f"DEBUG: [Scraper] 429 Rate Limit on {url}. Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    print(f"DEBUG: [Scraper] HTTP {resp.status_code} for {url}")
                    return None

            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"DEBUG: [Scraper] Final error for {url}: {type(e).__name__} {e}"
                    )
                else:
                    await asyncio.sleep(1)

    return None


async def fetch_calendar_api_events(client, calendar_id, target_date):
    """
    Fetches events for a specific calendar using Luma's v2 internal API.
    """
    try:
        after = f"{target_date}T00:00:00.000+08:00"
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        before = f"{next_day}T00:00:00.000+08:00"

        url = "https://api2.luma.com/calendar/get-items"
        params = {
            "after": after,
            "before": before,
            "calendar_api_id": calendar_id,
            "period": "specific",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://lu.ma/",
        }

        print(f"DEBUG: [Scraper] Calling Calendar API: {url} with params {params}")
        resp = await client.get(url, params=params, headers=headers, timeout=12)
        print(f"DEBUG: [Scraper] API2 Response Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            items = (
                data.get("items")
                or data.get("entries")
                or data.get("data", {}).get("entries")
                or data.get("data", {}).get("items")
                or []
            )
            print(f"DEBUG: [Scraper] API2 found {len(items)} raw items.")
            return items
        else:
            print(
                f"DEBUG: [Scraper] API2 Error: {resp.status_code} - {resp.text[:200]}"
            )
        return []
    except Exception as e:
        print(f"DEBUG: [Scraper] API2 Fatal Fetch Error: {type(e).__name__}: {e}")
        return []


async def get_optimized_events(url, target_date):
    """
    Main entry point for scraping Luma events for a target date.
    Uses httpx for high-concurrency async fetching.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://lu.ma/",
    }

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=20)
    async with httpx.AsyncClient(
        headers=headers, follow_redirects=True, limits=limits, trust_env=False
    ) as client:
        try:
            print(f"DEBUG: [Scraper] Analyzing landing page: {url}")
            response = await client.get(url)

            if response.status_code != 200:
                print(
                    f"DEBUG: [Scraper] Failed to fetch landing page: {response.status_code}"
                )
                return []

            soup = BeautifulSoup(response.text, "html.parser")

            cal_id = None
            meta_app = soup.find("meta", attrs={"name": "apple-itunes-app"})
            if meta_app and meta_app.get("content"):
                content = meta_app.get("content")
                match = re.search(r"luma://calendar/(cal-[a-zA-Z0-9]+)", content)
                if match:
                    cal_id = match.group(1)

            events_list = []
            if cal_id:
                print(f"DEBUG: [Scraper] Calendar ID detected: {cal_id}")
                events_list = await fetch_calendar_api_events(
                    client, cal_id, target_date
                )
            else:
                print(
                    "DEBUG: [Scraper] No Calendar ID found. Falling back to __NEXT_DATA__."
                )
                next_data_script = soup.find("script", id="__NEXT_DATA__")
                if next_data_script:
                    data = json.loads(next_data_script.string)
                    initial_data = (
                        data.get("props", {})
                        .get("pageProps", {})
                        .get("initialData", {})
                    )
                    events_list = initial_data.get("events") or initial_data.get(
                        "data", {}
                    ).get("events", [])

            print(
                f"DEBUG: [Scraper] Pre-filtering items: found {len(events_list)} raw entries."
            )

            def prepare_event(item):
                if not isinstance(item, dict):
                    return None
                if cal_id:
                    # For calendar API, all items are usually relevant if they come back from the specific period
                    # but we check platform anyway
                    platform = item.get("platform", "").lower()
                    if platform != "luma":
                        return None

                ev = item.get("event") or item.get("api_event") or item
                if not isinstance(ev, dict):
                    return None

                start_at_str = ev.get("start_at") or ev.get("start_date") or ""

                # If NOT calendar API, we MUST check the date in the JSON
                if not cal_id:
                    if target_date not in start_at_str:
                        return None

                eid = ev.get("api_id") or ev.get("id") or f"evt-{hash(start_at_str)}"
                name = ev.get("name") or ev.get("title") or "Untitled Event"
                path = ev.get("url") or ev.get("url_path") or ""
                full_url = f"https://lu.ma/{path}" if path else url

                return {
                    "eid": eid,
                    "name": name,
                    "full_url": full_url,
                    "ev": ev,
                    "item": item,
                }

            prepped_events = [
                p for p in (prepare_event(item) for item in events_list) if p
            ]

            print(
                f"DEBUG: [Scraper] Starting async deep fetch for {len(prepped_events)} prepped items."
            )

            async def fetch_with_results(event_info):
                desc = await fetch_event_details(client, event_info["full_url"])

                ev = event_info["ev"]
                item = event_info["item"]
                geo = ev.get("geo_address_info") or {}
                loc = geo.get("full_address") or geo.get("address") or "Hong Kong"

                return {
                    "id": str(event_info["eid"]),
                    "title": event_info["name"],
                    "url": event_info["full_url"],
                    "start_date": ev.get("start_at"),
                    "end_date": ev.get("end_at"),
                    "location": loc,
                    "description": desc or clean_description(ev.get("description", "")),
                    "guest_count": item.get("guest_count")
                    or ev.get("guest_count")
                    or ev.get("ticket_count")
                    or item.get("num_tickets_sold")
                    or 0,
                }

            results = await asyncio.gather(
                *(fetch_with_results(p) for p in prepped_events)
            )
            return results

        except Exception as e:
            print(f"DEBUG: [Scraper] Fatal error in get_optimized_events: {e}")
            import traceback

            traceback.print_exc()
            return []
