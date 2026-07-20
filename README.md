# 🔭 LumaScope

LumaScope is an event intelligence tool designed to scrape, filter, and analyze events from Lu.ma calendars. The application retrieves event details for a target date range, processes them in parallel batches using the Google Gemini API to extract key topics and highlights, and presents these structured insights through an interactive web dashboard.

## 🔑 Key Features

- **Asynchronous Scraping Pipeline**: Programmatically parses Lu.ma landing pages, identifies calendar IDs, and extracts full event descriptions using concurrent async requests.
- **LLM-Powered Summarization**: Groups extracted events into parallel batches and utilizes the Gemini API to generate concise summaries, topic tags, and reasons to attend.
- **Two-Tier Cache System**: Persists raw scraper pages and LLM API response hashes locally, preventing rate-limiting issues and minimizing token consumption.
- **Automatic Storage Management**: Implements an active janitor system that automatically cleans up old cached and processed JSON data files.

## 🧱 Architecture

LumaScope is structured as a modular FastAPI web service with separate backend layers for page parsing and AI analysis. The frontend dashboard collects user input for a Lu.ma calendar URL and date range, sending it to the FastAPI backend. The scraper layer concurrently extracts the raw event data, while the LLM handler batches descriptions to request structured JSON summaries from the Gemini API. Output files and caches are maintained inside a dedicated outputs folder, and responses are rendered directly onto a responsive user interface.

## 📥 Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   cd lumascope
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🚀 How to Use

- **Start the Web Server**: Run the FastAPI application locally:
  ```bash
  python main.py
  ```
  Access the web dashboard at `http://127.0.0.1:8000`.
