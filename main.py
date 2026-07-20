import os
import json
import time
import hashlib
import tempfile
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scraper import get_optimized_events, is_allowed_luma_url
from llm_handler import summarize_events

app = FastAPI()

# Environment-aware storage
IS_VERCEL = os.environ.get("VERCEL") == "1"
BASE_DIR = "/tmp" if IS_VERCEL else os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure outputs directory exists
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def cleanup_old_files(max_age_days=3):
    """Delete old output and nested cache files while keeping directories."""
    now = time.time()
    count = 0
    try:
        cutoff = now - (max_age_days * 86400)
        for root, _dirs, files in os.walk(OUTPUTS_DIR):
            for filename in files:
                fpath = os.path.join(root, filename)
                if os.stat(fpath).st_mtime < cutoff:
                    os.remove(fpath)
                    count += 1
        if count > 0:
            print(f"DEBUG: [Janitor] Cleaned up {count} old files.")
    except Exception as e:
        print(f"DEBUG: [Janitor] Error during cleanup: {e}")


def write_json_atomic(path, data):
    """Write a JSON artifact without exposing a partially written file."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False
        ) as handle:
            temp_path = handle.name
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def save_diagnostic_artifact(filename, data):
    """Persist local diagnostics without making a request depend on disk I/O."""
    if IS_VERCEL:
        return
    try:
        write_json_atomic(os.path.join(OUTPUTS_DIR, filename), data)
    except OSError as error:
        print(f"DEBUG: [Main] Could not save diagnostic artifact {filename}: {error}")


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/scrape")
async def handle_scrape(
    request: Request,
    url: str = Form(...),
    date_from: str = Form(...),
    date_to: str = Form(...),
):
    """
    url: https://lu.ma/hongkong
    date_from/date_to: YYYY-MM-DD (inclusive)
    """
    try:
        url = url.strip()
        if not is_allowed_luma_url(url):
            return JSONResponse(
                {"status": "error", "message": "Only HTTPS Luma calendar URLs are supported."},
                status_code=422,
            )
        date_from = date_from.strip()
        date_to = date_to.strip()
        start_day = datetime.strptime(date_from, "%Y-%m-%d").date()
        end_day = datetime.strptime(date_to, "%Y-%m-%d").date()
        if end_day < start_day:
            return JSONResponse(
                {"status": "error", "message": "End date must be on or after the start date."},
                status_code=422,
            )
        date_label = date_from if date_from == date_to else f"{date_from}_to_{date_to}"
        request_id = uuid.uuid4().hex[:12]
        url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        artifact_stem = f"{date_label}_{url_key}_{request_id}"
        print(
            f"DEBUG: [Main] Received request: url={url}, "
            f"date_from={date_from}, date_to={date_to}"
        )

        # 1. & 2. Get all details (including multi-step URL fetching)
        all_event_details = await get_optimized_events(url, date_from, date_to)

        if not all_event_details:
            cleanup_old_files()
            print(f"DEBUG: [Main] No events found for {date_label} at {url}")
            return {
                "status": "success",
                "data": [],
                "message": f"No events found for {date_label}.",
            }

        print(
            f"DEBUG: [Main] Found {len(all_event_details)} events. Proceeding with analysis..."
        )

        # 5. Save to raw JSON file (Full extracted data)
        save_diagnostic_artifact(f"raw_{artifact_stem}.json", all_event_details)

        # 6. Create cleaned_json for diagnostics; the LLM also receives compact metadata.
        cleaned_data = [
            {"id": e["id"], "desc": e["description"]} for e in all_event_details
        ]
        save_diagnostic_artifact(f"cleaned_{artifact_stem}.json", cleaned_data)

        # 7. Summarize with LLM (feeding only id and description data internally)
        print(
            f"DEBUG: [Analysis] Starting AI analysis for {len(all_event_details)} events..."
        )
        start_ai = datetime.now()
        enriched_events = await summarize_events(all_event_details)
        duration = (datetime.now() - start_ai).total_seconds()
        print(f"DEBUG: [Analysis] AI analysis completed in {duration:.2f}s")

        # Save the final results with summaries
        save_diagnostic_artifact(f"final_{artifact_stem}.json", enriched_events)

        # Run Janitor
        if not IS_VERCEL:
            cleanup_old_files()

        return {"status": "success", "data": enriched_events}

    except Exception as e:
        print(f"DEBUG: [Main] Error: {e}")
        import traceback

        traceback.print_exc()
        message = str(e) or "Unable to load events."
        status_code = 502 if isinstance(e, RuntimeError) else 500
        return JSONResponse({"status": "error", "message": message}, status_code=status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
