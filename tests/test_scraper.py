import unittest
from unittest.mock import patch

from llm_handler import (
    MAX_DENSE_SCRIPT_EXCERPT_CHARS,
    MAX_EXCERPT_CHARS,
    build_event_excerpt,
    build_local_fallback_summary,
    get_event_hash,
    normalize_tags,
)
from scraper import (
    clean_description,
    event_overlaps_date,
    event_overlaps_date_range,
    get_event_location,
    get_event_timezone,
    get_optimized_events,
    is_allowed_luma_url,
)


class DescriptionCleaningTests(unittest.TestCase):
    def test_keeps_about_inside_english_sentence(self):
        text = "Learn about AI and finance with founders from across the region."
        self.assertEqual(clean_description(text), text)

    def test_preserves_mixed_language_post(self):
        text = "Join builders for AI demos. 活動內容包括創業分享和交流。"
        self.assertEqual(clean_description(text), text)

    def test_preserves_non_english_post(self):
        text = "純中文活動介紹，歡迎大家參加。"
        self.assertEqual(clean_description(text), text.replace("\uff0c", ","))

    def test_removes_urls_images_duplicates_and_trailing_boilerplate(self):
        text = (
            "Join founders for AI demos.\n"
            "Image: User Uploaded Image\n"
            "https://example.com/register\n"
            "Join founders for AI demos.\n\n"
            "Contact\nhello@example.com"
        )
        self.assertEqual(clean_description(text), "Join founders for AI demos.")


class LLMExcerptTests(unittest.TestCase):
    def test_latin_excerpt_is_bounded_and_keeps_opening_content(self):
        text = "First useful event detail. " + ("More useful detail about speakers and agenda. " * 100)
        excerpt = build_event_excerpt(text)
        self.assertTrue(excerpt.startswith("First useful event detail."))
        self.assertLessEqual(len(excerpt), MAX_EXCERPT_CHARS + 1)

    def test_dense_script_excerpt_uses_lower_character_budget(self):
        excerpt = build_event_excerpt("歡迎參加創業者社群活動。" * 100)
        self.assertLessEqual(len(excerpt), MAX_DENSE_SCRIPT_EXCERPT_CHARS + 1)

    def test_local_fallback_summary_stays_within_fifty_words(self):
        summary = build_local_fallback_summary("Useful event detail. " * 100)
        self.assertLessEqual(len(summary.rstrip("…").split()), 50)

    def test_summary_cache_hash_changes_when_compact_input_metadata_changes(self):
        base = {
            "title": "AI Meetup",
            "start_date": "2026-07-18",
            "location": "Bangkok",
            "description": "English and 中文內容",
        }
        self.assertNotEqual(get_event_hash(base), get_event_hash({**base, "location": "New York"}))

    def test_summary_cache_hash_allows_missing_optional_location(self):
        event = {
            "title": "Online Meetup",
            "start_date": "2026-07-18T18:00:00Z",
            "location": None,
            "description": "A practical AI session for builders.",
        }
        self.assertTrue(get_event_hash(event))

    def test_normalizes_exact_duplicate_tags(self):
        tags = normalize_tags(["Party", "developer tools", "Developer Tools", "club", "Fintech"])
        self.assertEqual(tags, ["party", "developer-tools", "club"])


class EventTimezoneTests(unittest.TestCase):
    def test_accepts_only_https_luma_hosts(self):
        self.assertTrue(is_allowed_luma_url("https://luma.com/calendar"))
        self.assertTrue(is_allowed_luma_url("https://events.lu.ma/calendar"))
        self.assertFalse(is_allowed_luma_url("http://luma.com/calendar"))
        self.assertFalse(is_allowed_luma_url("https://example.com/calendar"))

    def test_uses_luma_location_instead_of_a_hard_coded_city(self):
        event = {"geo_address_info": {"city": "New York", "country": "United States"}}
        self.assertEqual(get_event_location(event), "New York, United States")

    def test_uses_event_iana_timezone(self):
        self.assertEqual(get_event_timezone({"timezone": "Asia/Bangkok"}), "Asia/Bangkok")

    def test_bangkok_event_is_compared_in_bangkok_date(self):
        event = {
            "timezone": "Asia/Bangkok",
            "start_at": "2026-07-15T18:30:00Z",
            "end_at": "2026-07-15T20:00:00Z",
        }
        self.assertFalse(event_overlaps_date(event, {}, "2026-07-15"))
        self.assertTrue(event_overlaps_date(event, {}, "2026-07-16"))

    def test_new_york_event_is_compared_in_new_york_date(self):
        event = {
            "timezone": "America/New_York",
            "start_at": "2026-07-16T02:00:00Z",
            "end_at": "2026-07-16T03:00:00Z",
        }
        self.assertTrue(event_overlaps_date(event, {}, "2026-07-15"))
        self.assertFalse(event_overlaps_date(event, {}, "2026-07-16"))

    def test_event_is_included_when_it_overlaps_a_multi_day_range(self):
        event = {
            "timezone": "Asia/Bangkok",
            "start_at": "2026-07-16T18:00:00+07:00",
            "end_at": "2026-07-16T20:00:00+07:00",
        }
        self.assertTrue(event_overlaps_date_range(event, {}, "2026-07-15", "2026-07-17"))
        self.assertFalse(event_overlaps_date_range(event, {}, "2026-07-17", "2026-07-18"))


class EventNormalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_cleaned_event_with_local_timezone(self):
        class Response:
            def __init__(self, status_code=200, text="", data=None):
                self.status_code = status_code
                self.text = text
                self._data = data

            def json(self):
                return self._data

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def get(self, url, **_kwargs):
                if url == "https://lu.ma/bangkok":
                    return Response(
                        text=(
                            '<meta name="apple-itunes-app" '
                            'content="app-argument=luma://calendar/cal-test">'
                            '<script id="__NEXT_DATA__" type="application/json">'
                            '{"timezone":"Asia/Bangkok"}</script>'
                        )
                    )
                if url == "https://api2.luma.com/calendar/get-items":
                    return Response(
                        data={
                            "items": [
                                {
                                    "platform": "luma",
                                    "guest_count": 42,
                                    "event": {
                                        "api_id": "evt-test",
                                        "name": "Bangkok Builder Meetup",
                                        "url_path": "builder-meetup",
                                        "start_at": "2026-07-15T03:00:00Z",
                                        "end_at": "2026-07-15T05:00:00Z",
                                        "timezone": "Asia/Bangkok",
                                    },
                                }
                            ]
                        }
                    )
                if url == "https://lu.ma/builder-meetup":
                    return Response(
                        text=(
                            '<script type="application/ld+json">'
                            '{"@type":"Event","description":'
                            '"Join local builders for practical AI demos and founder networking."}'
                            '</script>'
                        )
                    )
                raise AssertionError(f"Unexpected URL: {url}")

        with patch("scraper.httpx.AsyncClient", return_value=Client()), patch("scraper.set_cached_data"):
            events = await get_optimized_events("https://lu.ma/bangkok", "2026-07-15")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["id"], "evt-test")
        self.assertEqual(events[0]["timezone"], "Asia/Bangkok")
        self.assertEqual(events[0]["guest_count"], 42)
        self.assertEqual(events[0]["description"], "Join local builders for practical AI demos and founder networking.")


if __name__ == "__main__":
    unittest.main()
