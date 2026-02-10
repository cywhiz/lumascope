import os
import json
import time
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scraper import get_optimized_events
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
    """Delete files in the outputs directory older than max_age_days."""
    now = time.time()
    count = 0
    try:
        for f in os.listdir(OUTPUTS_DIR):
            fpath = os.path.join(OUTPUTS_DIR, f)
            if os.path.isfile(fpath):
                if os.stat(fpath).st_mtime < (now - (max_age_days * 86400)):
                    os.remove(fpath)
                    count += 1
        if count > 0:
            print(f"DEBUG: [Janitor] Cleaned up {count} old files.")
    except Exception as e:
        print(f"DEBUG: [Janitor] Error during cleanup: {e}")


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/scrape")
async def handle_scrape(request: Request, url: str = Form(...), date: str = Form(...)):
    """
    url: https://lu.ma/hongkong
    date: YYYY-MM-DD
    """
    try:
        url = url.strip()
        date = date.strip()
        print(f"DEBUG: [Main] Received request: url={url}, date={date}")

        # 1. & 2. Get all details (including multi-step URL fetching)
        all_event_details = await get_optimized_events(url, date)

        if not all_event_details:
            print(f"DEBUG: [Main] No events found for {date} at {url}")
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"No events found for {date} at {url}.",
                },
                status_code=400,
            )

        print(
            f"DEBUG: [Main] Found {len(all_event_details)} events. Proceeding with analysis..."
        )

        # 5. Save to raw JSON file (Full extracted data)
        raw_output_path = os.path.join(OUTPUTS_DIR, f"raw_{date}.json")
        with open(raw_output_path, "w", encoding="utf-8") as f:
            json.dump(all_event_details, f, indent=2)

        # 6. Create cleaned_json (ID and cleaned description for LLM)
        cleaned_data = [
            {"id": e["id"], "desc": e["description"]} for e in all_event_details
        ]
        cleaned_output_path = os.path.join(OUTPUTS_DIR, f"cleaned_{date}.json")
        with open(cleaned_output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2)

        # 7. Summarize with LLM (feeding only id and description data internally)
        print(
            f"DEBUG: [Analysis] Starting AI analysis for {len(all_event_details)} events..."
        )
        start_ai = datetime.now()
        enriched_events = await summarize_events(all_event_details)
        duration = (datetime.now() - start_ai).total_seconds()
        print(f"DEBUG: [Analysis] AI analysis completed in {duration:.2f}s")

        # Save the final results with summaries
        final_output_path = os.path.join(OUTPUTS_DIR, f"final_{date}.json")
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_events, f, indent=2)

        # Run Janitor
        cleanup_old_files()

        return {"status": "success", "data": enriched_events}

    except Exception as e:
        print(f"DEBUG: [Main] Error: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    # Disabled reload for stability
    uvicorn.run(app, host="0.0.0.0", port=8000)
