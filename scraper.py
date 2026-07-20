import httpx
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import asyncio
import re
import regex
import hashlib
import os
import unicodedata
import time
from urllib.parse import urljoin
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Environment-aware storage (Vercel uses /tmp)
IS_VERCEL = os.environ.get("VERCEL") == "1"
BASE_DIR = "/tmp" if IS_VERCEL else os.getcwd()

CACHE_DIR = os.path.join(BASE_DIR, "outputs", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

USE_CACHE = True
SCRAPER_CACHE_VERSION = 5
CACHE_TTL_SECONDS = 3 * 86400
print(f"DEBUG: [Scraper] Cache status: {'Enabled' if USE_CACHE else 'DISABLED'}")

# Prevent socket exhaustion (ConnectError) on Windows
scraper_semaphore = asyncio.Semaphore(10)


def is_allowed_luma_url(value):
    """Allow only HTTPS URLs hosted by Luma."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(str(value).strip())
        host = (parsed.hostname or "").lower().rstrip(".")
        return (
            parsed.scheme == "https"
            and not parsed.username
            and not parsed.password
            and (
                host in {"lu.ma", "luma.com"}
                or host.endswith(".lu.ma")
                or host.endswith(".luma.com")
            )
        )
    except (TypeError, ValueError):
        return False

ENGLISH_WORD_RE = regex.compile(r"\b[A-Za-z]+(?:['’-][A-Za-z]+)?\b")
BOILERPLATE_HEADING_RE = regex.compile(
    r"(?im)^\s*(?:about(?:\s+(?:the\s+)?(?:host|organizer|venue))?|"
    r"previous\s+events?|contact|registration|refund\s+policy|disclaimer|"
    r"terms(?:\s+and\s+conditions)?|privacy(?:\s+policy)?)\s*:?[ \t]*$"
)


def get_cache_path(url):
    url_hash = hashlib.md5(
        f"{SCRAPER_CACHE_VERSION}:{url}".encode("utf-8")
    ).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.json")


def get_cached_data(url):
    if not USE_CACHE:
        return None
    path = get_cache_path(url)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cached_at = data.get("cached_at")
                if cached_at and time.time() - float(cached_at) > CACHE_TTL_SECONDS:
                    return None
                return data
        except:
            return None
    return None


def set_cached_data(url, data):
    if not USE_CACHE:
        return
    path = get_cache_path(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**data, "cached_at": time.time()}, f)
    except:
        pass


def clean_description(text):
    """Preserve multilingual event prose while removing deterministic noise."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n")
    heading = BOILERPLATE_HEADING_RE.search(text)
    if heading:
        text = text[: heading.start()]

    # Keep paragraph breaks for the compact LLM selector, but remove text-only noise.
    text = regex.sub(r"https?://\S+|www\.\S+|\b\S+@\S+\.\S+\b", " ", text)
    lines = []
    seen = set()
    for line in text.splitlines():
        line = regex.sub(r"(?i)^\s*(?:image|uploaded image|image description)\s*:\s*.*$", "", line)
        line = regex.sub(r"[ \t]+", " ", line).strip(" ,;:-")
        key = line.casefold()
        if line and key not in seen:
            lines.append(line)
            seen.add(key)
    return "\n".join(lines).strip()


def get_event_timezone(event, item=None):
    """Return the first valid IANA timezone supplied by Luma event metadata."""
    item = item or {}
    geo = event.get("geo_address_info") or {}
    candidates = [
        event.get("timezone"),
        event.get("timezone_id"),
        event.get("tz"),
        geo.get("timezone"),
        geo.get("timezone_id"),
        item.get("timezone"),
        item.get("timezone_id"),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("id") or candidate.get("name")
        if not isinstance(candidate, str) or not candidate:
            continue
        try:
            ZoneInfo(candidate)
            return candidate
        except (ValueError, ZoneInfoNotFoundError):
            continue
    return None


def get_event_location(event, item=None):
    """Return only a location supplied by Luma; never assume a calendar city."""
    item = item or {}
    sources = [
        event.get("geo_address_info"),
        item.get("geo_address_info"),
        event,
        item,
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("full_address", "address", "location_name", "location"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        city = source.get("city") or source.get("locality")
        country = source.get("country") or source.get("country_name")
        if isinstance(city, str) and city.strip():
            return ", ".join(
                part.strip()
                for part in (city, country)
                if isinstance(part, str) and part.strip()
            )
    return None


def event_overlaps_date_range(
    event, item, start_date, end_date, fallback_timezone=None
):
    """Check whether an event overlaps an inclusive local-date range."""
    try:
        start_day = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_day = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_day < start_day:
            return False
        timezone_name = get_event_timezone(event, item) or fallback_timezone
        timezone = ZoneInfo(timezone_name) if timezone_name else None

        def parse_timestamp(value):
            if not value:
                return None
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone or ZoneInfo("UTC"))
            return parsed.astimezone(timezone) if timezone else parsed

        starts_at = parse_timestamp(event.get("start_at") or event.get("start_date"))
        ends_at = parse_timestamp(event.get("end_at") or event.get("end_date"))
        if not starts_at:
            return False

        range_start = datetime.combine(
            start_day, datetime.min.time(), tzinfo=starts_at.tzinfo
        )
        range_end = datetime.combine(
            end_day + timedelta(days=1), datetime.min.time(), tzinfo=starts_at.tzinfo
        )
        return starts_at < range_end and (ends_at or starts_at) >= range_start
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        return False


def event_overlaps_date(event, item, target_date, fallback_timezone=None):
    """Check whether an event overlaps one local calendar date."""
    return event_overlaps_date_range(
        event, item, target_date, target_date, fallback_timezone
    )


def find_timezone(data):
    """Find a valid IANA timezone in landing-page JSON without assuming a city."""
    if isinstance(data, dict):
        for key in ("timezone", "timezone_id", "tz"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    ZoneInfo(value)
                    return value
                except (ValueError, ZoneInfoNotFoundError):
                    pass
        for value in data.values():
            found = find_timezone(value)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_timezone(value)
            if found:
                return found
    return None


async def fetch_event_details(client, url):
    """
    Go through each url in the events field and get the full json,
    parse the event description field.
    Using cache and httpx for speed.
    """
    async with scraper_semaphore:
        if not is_allowed_luma_url(url):
            return None
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
                        raw_ld = ld_script.string or ld_script.get_text()
                        if not raw_ld:
                            return None
                        try:
                            ld_data = json.loads(raw_ld)
                        except (TypeError, json.JSONDecodeError):
                            return None
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


async def fetch_calendar_api_events(
    client, calendar_id, start_date, end_date, timezone_name=None
):
    """
    Fetches events for a specific calendar using Luma's v2 internal API.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        if timezone_name:
            timezone = ZoneInfo(timezone_name)
            after_dt = start_dt.replace(tzinfo=timezone)
            before_dt = end_dt.replace(tzinfo=timezone)
        else:
            # Cover every civil timezone when the landing page has no timezone.
            after_dt = start_dt.replace(tzinfo=ZoneInfo("UTC")) - timedelta(hours=14)
            before_dt = end_dt.replace(tzinfo=ZoneInfo("UTC")) + timedelta(hours=12)
        after = after_dt.isoformat(timespec="milliseconds")
        before = before_dt.isoformat(timespec="milliseconds")

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
            if not isinstance(data, dict):
                return []
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
    except httpx.RequestError as e:
        print(f"DEBUG: [Scraper] API2 network error: {type(e).__name__}: {e}")
        raise RuntimeError("Unable to reach the Luma calendar API.") from e
    except Exception as e:
        print(f"DEBUG: [Scraper] API2 Fatal Fetch Error: {type(e).__name__}: {e}")
        return []


async def get_optimized_events(url, start_date, end_date=None):
    """
    Main entry point for scraping Luma events for an inclusive date range.
    Uses httpx for high-concurrency async fetching.
    """
    end_date = end_date or start_date
    if not is_allowed_luma_url(url):
        print(f"DEBUG: [Scraper] Rejected non-Luma URL: {url}")
        return []
    try:
        if datetime.strptime(end_date, "%Y-%m-%d") < datetime.strptime(start_date, "%Y-%m-%d"):
            return []
    except (TypeError, ValueError):
        return []
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
                raise RuntimeError(
                    f"Luma returned HTTP {response.status_code} for the calendar page."
                )

            soup = BeautifulSoup(response.text, "html.parser")

            next_data = None
            calendar_timezone = None
            next_data_script = soup.find("script", id="__NEXT_DATA__")
            if next_data_script and next_data_script.string:
                next_data = json.loads(next_data_script.string)
                calendar_timezone = find_timezone(next_data)

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
                    client, cal_id, start_date, end_date, calendar_timezone
                )
            else:
                print(
                    "DEBUG: [Scraper] No Calendar ID found. Falling back to __NEXT_DATA__."
                )
                if next_data:
                    initial_data = (
                        next_data.get("props", {})
                        .get("pageProps", {})
                        .get("initialData", {})
                    )
                    events_list = initial_data.get("events") or initial_data.get(
                        "data", {}
                    ).get("events", [])
            if not isinstance(events_list, list):
                events_list = []

            print(
                f"DEBUG: [Scraper] Pre-filtering items: found {len(events_list)} raw entries."
            )

            def prepare_event(item):
                if not isinstance(item, dict):
                    return None
                if cal_id:
                    # For calendar API, all items are usually relevant if they come back from the specific period
                    # but we check platform anyway
                    platform = item.get("platform")
                    if platform and str(platform).lower() != "luma":
                        return None

                ev = item.get("event") or item.get("api_event") or item
                if not isinstance(ev, dict):
                    return None

                start_at_str = ev.get("start_at") or ev.get("start_date") or ""

                if not event_overlaps_date_range(
                    ev, item, start_date, end_date, calendar_timezone
                ):
                    return None

                name = ev.get("name") or ev.get("title") or "Untitled Event"
                path = ev.get("url") or ev.get("url_path") or ""
                full_url = urljoin("https://lu.ma/", path) if path else url
                if not is_allowed_luma_url(full_url):
                    return None
                eid = ev.get("api_id") or ev.get("id") or (
                    "evt-" + hashlib.sha256(
                        f"{full_url}|{name}|{start_at_str}".encode("utf-8")
                    ).hexdigest()[:16]
                )

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
                loc = get_event_location(ev, item)
                description = desc or clean_description(ev.get("description", ""))
                if not description:
                    return None

                return {
                    "id": str(event_info["eid"]),
                    "title": event_info["name"],
                    "url": event_info["full_url"],
                    "start_date": ev.get("start_at"),
                    "end_date": ev.get("end_at"),
                    "location": loc,
                    "description": description,
                    "timezone": get_event_timezone(ev, item) or calendar_timezone,
                    "guest_count": item.get("guest_count")
                    or ev.get("guest_count")
                    or ev.get("ticket_count")
                    or item.get("num_tickets_sold")
                    or 0,
                }

            results = await asyncio.gather(
                *(fetch_with_results(p) for p in prepped_events)
            )
            return [result for result in results if result]

        except httpx.RequestError as e:
            print(f"DEBUG: [Scraper] Network error: {type(e).__name__}: {e}")
            raise RuntimeError("Unable to reach Luma. Check the server network connection.") from e
        except Exception as e:
            print(f"DEBUG: [Scraper] Fatal error in get_optimized_events: {e}")
            import traceback

            traceback.print_exc()
            return []
