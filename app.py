import os
import gradio as gr
import requests
import inspect
import pandas as pd
import re
import time

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    Tool,
)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# =============================================================
# Key rotation: read up to 5 Gemini keys from secrets.
# When one hits rate limit, rotate to the next.
# =============================================================

def _load_gemini_keys():
    keys = []
    # Primary
    primary = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if primary:
        keys.append(primary)
    # Numbered fallbacks: GEMINI_API_KEY_2 ... GEMINI_API_KEY_6
    for i in range(2, 7):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if k:
            keys.append(k)
    return keys


# Module-level state so YouTube/Vision/Audio tools can rotate keys too
_GEMINI_KEYS = _load_gemini_keys()
_GEMINI_KEY_IDX = 0


def _get_next_gemini_key():
    """Returns the current Gemini key and advances the rotation index by one."""
    global _GEMINI_KEY_IDX
    if not _GEMINI_KEYS:
        return None
    key = _GEMINI_KEYS[_GEMINI_KEY_IDX]
    _GEMINI_KEY_IDX = (_GEMINI_KEY_IDX + 1) % len(_GEMINI_KEYS)
    return key


def _call_gemini_with_rotation(content_parts, model_name="gemini-2.5-flash"):
    """Calls Gemini, rotating through ALL available keys on rate-limit.
    Used by YouTube, Vision, and Audio tools. Returns response text or raises."""
    import google.generativeai as genai
    last_err = None
    for attempt in range(len(_GEMINI_KEYS) + 1):
        key = _get_next_gemini_key()
        if not key:
            raise RuntimeError("No Gemini keys available")
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(content_parts)
            return resp.text
        except Exception as e:
            err_str = str(e).lower()
            last_err = e
            if any(s in err_str for s in ["rate", "429", "quota", "resource_exhausted"]):
                # Rotate to next key, no wait needed
                continue
            raise
    raise last_err


class RotatingLiteLLMModel(LiteLLMModel):
    """
    LiteLLMModel that rotates through multiple Gemini API keys when one
    hits rate limit. Overrides BOTH generate() (the canonical smolagents
    entrypoint) AND __call__() (older API) for maximum compatibility.

    Just swaps the api_key attribute (no full re-init). After cycling
    through every key once without success, performs a real backoff using
    the server-suggested retryDelay (or a 45s fallback) before the second
    cycle.
    """
    # Maximum time a single model call may take before we abort and rotate.
    # Prevents indefinite hangs inside LiteLLM's internal retry/sleep paths.
    CALL_TIMEOUT_SECONDS = 120

    def __init__(self, model_id, api_keys, temperature=0.1, **kwargs):
        self._all_keys = api_keys
        self._key_index = 0
        self._model_id = model_id
        self._temperature = temperature
        self._kwargs = kwargs
        self._call_count = 0  # for diagnostic logging
        self._dead_keys = set()  # indices of keys we've given up on this session

        # Try to suppress LiteLLM's noisy "Give Feedback / Get Help" prints
        # so the log is readable. Best-effort - quietly ignore failures.
        try:
            import litellm
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
        except Exception:
            pass

        # num_retries=0 disables LiteLLM's internal retry loop so OUR
        # rotation takes over immediately on rate-limit.
        # timeout caps individual call duration so we can never hang forever.
        super().__init__(
            model_id=model_id,
            api_key=api_keys[0],
            temperature=temperature,
            num_retries=0,
            timeout=self.CALL_TIMEOUT_SECONDS,
            request_timeout=self.CALL_TIMEOUT_SECONDS,
            **kwargs,
        )
        print(f"RotatingLiteLLMModel: loaded {len(api_keys)} key(s)")

    def _alive_count(self) -> int:
        return len(self._all_keys) - len(self._dead_keys)

    def _mark_current_dead(self, reason: str):
        idx = self._key_index
        if idx not in self._dead_keys:
            self._dead_keys.add(idx)
            print(f"[KEY-DEAD] key #{idx + 1}/{len(self._all_keys)} marked permanently dead this session ({reason}). Alive: {self._alive_count()}/{len(self._all_keys)}")

    def _switch_to_next_key(self):
        """Advance to the next ALIVE key. Skips keys marked dead this session."""
        if self._alive_count() == 0:
            raise RuntimeError("All API keys are dead this session. Check your Gemini projects/accounts.")
        for _ in range(len(self._all_keys)):
            self._key_index = (self._key_index + 1) % len(self._all_keys)
            if self._key_index not in self._dead_keys:
                self.api_key = self._all_keys[self._key_index]
                print(f"[ROTATE] {time.strftime('%H:%M:%S')} -> key #{self._key_index + 1}/{len(self._all_keys)}")
                return
        raise RuntimeError("No alive keys found during rotation")

    def _classify_error(self, err_str: str) -> str:
        """Return one of: 'dead' (permanent per-key), 'rotate' (temporary, try
        another key), 'raise' (not key-related, propagate)."""
        s = err_str.lower()
        # Permanent per-key failures - this key/project will not recover today
        if any(t in s for t in [
            "permission_denied",
            "permission denied",
            "denied access",
            "api_key_invalid",
            "api key not valid",
            "api_key_service_blocked",
            "403",
        ]):
            return "dead"
        # Temporary signals - rate limit, quota, transient network/timeout
        if any(t in s for t in [
            "rate", "429", "quota", "resource_exhausted", "ratelimit",
            "timeout", "timed out", "deadline exceeded", "read timed out",
            "500", "502", "503", "504", "service unavailable",
        ]):
            return "rotate"
        return "raise"

    def _rotating_call(self, parent_call, *args, **kwargs):
        """Shared rotation loop used by both generate() and __call__()."""
        self._call_count += 1
        call_id = self._call_count
        print(f"[GENERATE] call #{call_id} on key #{self._key_index + 1}/{len(self._all_keys)} (alive: {self._alive_count()})")
        last_err = None
        last_retry_delay = None
        keys_tried_this_round = 0
        # Allow up to two full cycles through the ALIVE keys
        max_attempts = max(2, self._alive_count() * 2)
        for cycle in range(max_attempts):
            # Bail out early if we've killed every key
            if self._alive_count() == 0:
                print(f"[FATAL] all {len(self._all_keys)} keys are dead this session")
                raise last_err or RuntimeError("All API keys are dead")

            t_start = time.time()
            try:
                result = parent_call(*args, **kwargs)
                duration = time.time() - t_start
                print(f"[GEN-DONE] call #{call_id} succeeded on key #{self._key_index + 1} in {duration:.1f}s")
                return result
            except Exception as e:
                duration = time.time() - t_start
                err_str = str(e)
                last_err = e
                verdict = self._classify_error(err_str)

                if verdict == "raise":
                    print(f"[NO-ROTATE] call #{call_id} non-rotatable error after {duration:.1f}s: {err_str[:200]}")
                    raise

                if verdict == "dead":
                    self._mark_current_dead(reason=f"403/permission-denied in {duration:.1f}s")
                    if self._alive_count() == 0:
                        print(f"[FATAL] last alive key just died")
                        raise
                    # Rotate to a still-alive key and try again
                    self._switch_to_next_key()
                    time.sleep(1)
                    continue

                # verdict == "rotate" → temporary issue, swap keys & retry
                print(f"[ROTATE-TRIGGER] call #{call_id} on key #{self._key_index + 1} after {duration:.1f}s (cycle {cycle + 1}/{max_attempts})")

                m = re.search(r'"retryDelay":\s*"([\d.]+)s"', err_str)
                if not m:
                    m = re.search(r"retry in ([\d.]+)\s*s", err_str)
                if m:
                    last_retry_delay = int(float(m.group(1))) + 2

                keys_tried_this_round += 1

                if keys_tried_this_round >= self._alive_count():
                    wait = last_retry_delay or 45
                    print(f"[BACKOFF] all {self._alive_count()} alive keys failed this round, waiting {wait}s")
                    time.sleep(wait)
                    keys_tried_this_round = 0
                    last_retry_delay = None

                if self._alive_count() > 1:
                    self._switch_to_next_key()
                    time.sleep(1)
                else:
                    if keys_tried_this_round == 0:
                        wait = last_retry_delay or 45
                        print(f"[BACKOFF] single alive key, waiting {wait}s")
                        time.sleep(wait)
        raise last_err

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        """Canonical smolagents Model entrypoint. This is what CodeAgent calls."""
        return self._rotating_call(
            super().generate,
            messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )

    def __call__(self, messages, **kwargs):
        """Backward-compat for older smolagents versions or direct calls."""
        return self._rotating_call(super().__call__, messages, **kwargs)


# =============================================================
# Tool classes
# =============================================================

class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Fetches a webpage and returns its readable text content. Wikipedia pages get a higher length cap; other sites are truncated to ~6000 chars to keep token costs low."
    inputs = {
        "url": {"type": "string", "description": "The URL of the webpage to fetch."}
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            try:
                from markdownify import markdownify as md
            except ImportError:
                md = None
            r = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )
            r.raise_for_status()
            text = md(r.text) if md else r.text
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            cap = 15000 if "wikipedia.org" in url else 6000
            if len(text) > cap:
                text = text[:cap] + "\n\n...[truncated - try the wikipedia_get_article tool for full Wikipedia content, or refine your question]"
            return text
        except Exception as e:
            return f"Error fetching {url}: {e}"


class WikipediaSearchTool(Tool):
    name = "wikipedia_search"
    description = "Searches Wikipedia and returns up to 3 article summaries with URLs. First step for any factual question about people, places, events, books, films."
    inputs = {
        "query": {"type": "string", "description": "The search query."}
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            s = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 3,
                },
                timeout=15,
            ).json()
            hits = s.get("query", {}).get("search", [])
            if not hits:
                return "No Wikipedia results found."
            out = []
            for h in hits:
                title = h["title"]
                try:
                    summary = requests.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                        timeout=15,
                    ).json()
                    extract = summary.get("extract", "")
                    url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
                    out.append(f"== {title} ==\n{extract}\nURL: {url}")
                except Exception:
                    continue
            return "\n\n".join(out) if out else "No usable summaries."
        except Exception as e:
            return f"Wikipedia error: {e}"


class WikipediaGetArticleTool(Tool):
    name = "wikipedia_get_article"
    description = "Gets the FULL plain-text content of a Wikipedia article by exact title. Use this for long Wikipedia pages (list pages, discographies, filmographies) where visit_webpage truncates too much. Much faster than scraping HTML."
    inputs = {
        "title": {"type": "string", "description": "Exact Wikipedia article title, e.g. 'Mercedes Sosa' or 'Wikipedia:Featured articles promoted in 2016'."}
    }
    output_type = "string"

    def forward(self, title: str) -> str:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                },
                headers={"User-Agent": "GAIAAgent/1.0 (educational use)"},
                timeout=20,
            )
            # Wikipedia sometimes returns 200 with HTML on errors - check content
            if not r.text.strip().startswith("{"):
                return f"Wikipedia API returned non-JSON (status {r.status_code}). Try wikipedia_search with the same query first to find the exact title."
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                if "extract" in page and page["extract"]:
                    text = page["extract"]
                    if len(text) > 25000:
                        text = text[:25000] + "\n...[truncated; article is very long]"
                    return f"== {page.get('title', title)} ==\n\n{text}"
            return f"Article '{title}' not found. Try wikipedia_search first to confirm the exact title."
        except Exception as e:
            return f"Wikipedia API error: {e}. Try wikipedia_search instead."


class DownloadTaskFileTool(Tool):
    name = "download_task_file"
    description = "Downloads the file attached to a GAIA task. Only call when the question EXPLICITLY mentions an attached file. Returns 'NO_FILE: ...' if there is no file."
    inputs = {
        "task_id": {"type": "string", "description": "The task_id of the current GAIA question."}
    }
    output_type = "string"

    def forward(self, task_id: str) -> str:
        try:
            url = f"{DEFAULT_API_URL}/files/{task_id}"
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return "NO_FILE: This task has no attached file. Answer the question using text reasoning and other tools."
            r.raise_for_status()
            cd = r.headers.get("content-disposition", "")
            m = re.search(r'filename="?([^";]+)"?', cd)
            fname = m.group(1) if m else f"{task_id}.bin"
            path = f"/tmp/{fname}"
            with open(path, "wb") as f:
                f.write(r.content)
            return f"Downloaded to: {path} (size: {len(r.content)} bytes)"
        except Exception as e:
            return f"Download error: {e}"


class ReadTextFileTool(Tool):
    name = "read_text_file"
    description = "Reads a text or code file (.txt, .py, .json, .md) as a string. Truncated to ~8000 chars."
    inputs = {
        "path": {"type": "string", "description": "Local filesystem path."}
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 8000:
                content = content[:8000] + "\n\n...[truncated]"
            return content
        except Exception as e:
            return f"Read error: {e}"


class ReadExcelFileTool(Tool):
    name = "read_excel_file"
    description = "Reads an Excel (.xlsx, .xls) or CSV file as a text table."
    inputs = {
        "path": {"type": "string", "description": "Local filesystem path."}
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            text = df.to_string(max_rows=200, max_cols=30)
            if len(text) > 8000:
                text = text[:8000] + "\n...[truncated]"
            return f"Shape: {df.shape}\nColumns: {list(df.columns)}\n\n{text}"
        except Exception as e:
            return f"Excel/CSV read error: {e}"


class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript"
    description = "Fetches the transcript/captions of a YouTube video. Use for any question that references a YouTube URL and asks about what is said or shown."
    inputs = {
        "url": {"type": "string", "description": "Full YouTube URL or 11-character video ID."}
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        # Extract video id
        m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url)
        video_id = m.group(1) if m else url.strip()

        # Strategy 1: new youtube-transcript-api API (v1.x)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id)
            if hasattr(fetched, "to_raw_data"):
                segments = fetched.to_raw_data()
            else:
                segments = list(fetched)
            text = " ".join(
                (seg["text"] if isinstance(seg, dict) else seg.text)
                for seg in segments
            )
            if len(text) > 10000:
                text = text[:10000] + "\n...[truncated]"
            return f"YouTube transcript ({video_id}):\n\n{text}"
        except Exception as e1:
            err1 = str(e1)

        # Strategy 2: old API (0.6.x)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            segments = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(seg["text"] for seg in segments)
            if len(text) > 10000:
                text = text[:10000] + "\n...[truncated]"
            return f"YouTube transcript ({video_id}):\n\n{text}"
        except Exception as e2:
            err2 = str(e2)

        # Strategy 3: Gemini Files API (ask Gemini directly about the video) - WITH KEY ROTATION
        try:
            content = [
                {
                    "file_data": {
                        "file_uri": f"https://www.youtube.com/watch?v={video_id}",
                        "mime_type": "video/*",
                    }
                },
                "Transcribe this YouTube video. Provide the full spoken content and a brief description of what is visually shown.",
            ]
            text = _call_gemini_with_rotation(content)
            return f"YouTube content (via Gemini, {video_id}):\n\n{text[:10000]}"
        except Exception as e3:
            return (
                f"All YouTube methods failed.\n"
                f"transcript-api new: {err1[:200]}\n"
                f"transcript-api old: {err2[:200]}\n"
                f"gemini-video: {str(e3)[:200]}\n"
                f"Try visit_webpage on the URL as last resort."
            )


class GeminiVisionTool(Tool):
    name = "analyze_image"
    description = "Analyzes an image using Gemini Vision. For chess positions, screenshots, diagrams, photos, charts. Download the file first with download_task_file."
    inputs = {
        "image_path": {"type": "string", "description": "Local filesystem path to the image."},
        "question": {"type": "string", "description": "What you want to know about the image."},
    }
    output_type = "string"

    def forward(self, image_path: str, question: str) -> str:
        try:
            import google.generativeai as genai
            # Upload file using the first available key (uploads don't usually rate-limit)
            key = _GEMINI_KEYS[0] if _GEMINI_KEYS else None
            if not key:
                return "Vision error: no API key"
            genai.configure(api_key=key)
            f = genai.upload_file(image_path)
            # Use rotation for the actual generation call which can rate-limit
            return _call_gemini_with_rotation([question, f])
        except Exception as e:
            return f"Vision error: {e}"


class GeminiAudioTool(Tool):
    name = "analyze_audio"
    description = "Transcribes or analyzes an audio file (.mp3, .wav, .m4a) using Gemini multimodal. Download the file first with download_task_file."
    inputs = {
        "audio_path": {"type": "string", "description": "Local filesystem path to the audio."},
        "question": {"type": "string", "description": "What to extract from the audio (e.g., 'transcribe', 'list all ingredients mentioned')."},
    }
    output_type = "string"

    def forward(self, audio_path: str, question: str) -> str:
        try:
            import google.generativeai as genai
            key = _GEMINI_KEYS[0] if _GEMINI_KEYS else None
            if not key:
                return "Audio error: no API key"
            genai.configure(api_key=key)
            f = genai.upload_file(audio_path)
            return _call_gemini_with_rotation([question, f])
        except Exception as e:
            return f"Audio error: {e}"


# =============================================================
# System prompt - tighter, with speed-conscious guidance
# =============================================================

GAIA_INSTRUCTIONS = """You are a GAIA benchmark assistant. SPEED MATTERS - finish each question in 4 steps or fewer.

ANSWER FORMAT (grader does EXACT STRING MATCH):
- Numbers: digits only, no commas, no units unless asked.
- Strings: no leading articles, exact spelling, no trailing punctuation.
- Lists: comma + space separated.
- Yes/No: just "Yes" or "No".
- Names: full name as in source.

EFFICIENT WORKFLOW:
1. Read question. Pick ONE primary tool. Don't re-fetch the same content.
2. ONLY call download_task_file(task_id) when the question EXPLICITLY mentions an attached file. If you get "NO_FILE", continue without it.
3. For Wikipedia FACTS: wikipedia_search first.
4. For Wikipedia LIST/COUNT questions ("how many", "which were promoted in...", filmography, discography): use wikipedia_get_article(title) to get the full plain text in ONE call. DO NOT use visit_webpage on Wikipedia list pages - it's slow and truncates badly.
5. For YouTube URLs: youtube_transcript.
6. For attached images: analyze_image (after download).
7. For attached audio: analyze_audio (after download).
8. For Excel/CSV: read_excel_file (after download).
9. For math/parsing/counting: python interpreter with requests if needed.
10. Pass ONLY the answer to final_answer(). No preamble, no quotes.

CRITICAL: If a tool returns truncated data or errors, switch to a DIFFERENT tool. Don't retry the same one.

WHEN A SITE BLOCKS YOU (403, "Forbidden", "Error fetching"):
- Discogs/Genius/Spotify/Apple Music: blocked by their bot protection. DON'T retry. Search Wikipedia or use duckduckgo instead.
- For discography/album questions: try wikipedia_get_article on the artist's main Wikipedia page - it usually has a Discography section.
- For song lyrics: NEVER scrape Genius. Try Wikipedia or rephrased web search.
- If ALL searches fail, make a best-effort answer from what you have and call final_answer. NEVER return None.

YOU MUST ALWAYS CALL final_answer() WITH SOMETHING, even if uncertain. An incorrect guess is better than no answer (which is auto-wrong).
"""


def _is_reversed_text(text: str) -> bool:
    if not text:
        return False
    return text.startswith(".") or ".rewsna eht sa" in text


# =============================================================
# Basic Agent Definition
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
# =============================================================

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        keys = _load_gemini_keys()
        if not keys:
            raise ValueError(
                "No Gemini API keys found. Add GEMINI_API_KEY (and optionally "
                "GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4, GEMINI_API_KEY_5) in "
                "Space Settings -> Variables and Secrets."
            )

        # Main agent loop uses Flash-Lite (4x daily quota, 50% more RPM
        # vs Flash). Multimodal tools below stay on Flash so their quota
        # pool is separate from the main reasoning loop.
        model = RotatingLiteLLMModel(
            model_id="gemini/gemini-2.5-flash-lite",
            api_keys=keys,
            temperature=0.1,
        )

        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                WikipediaSearchTool(),
                WikipediaGetArticleTool(),
                DownloadTaskFileTool(),
                ReadTextFileTool(),
                ReadExcelFileTool(),
                YouTubeTranscriptTool(),
                GeminiVisionTool(),
                GeminiAudioTool(),
                PythonInterpreterTool(),
            ],
            model=model,
            max_steps=4,
            additional_authorized_imports=[
                "pandas", "numpy", "requests", "bs4", "json", "re",
                "math", "datetime", "itertools", "collections", "os",
                "statistics", "string",
            ],
        )

    def __call__(self, question: str, task_id: str = "") -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # GAIA reversed-text trick
        reversed_note = ""
        if _is_reversed_text(question):
            decoded = question[::-1]
            reversed_note = (
                "NOTE: This question is written in REVERSED text.\n"
                f"Decoded normal form:\n{decoded}\n\n"
                "Answer based on the decoded form.\n\n"
            )

        prompt = GAIA_INSTRUCTIONS + "\n\n" + reversed_note
        if task_id:
            prompt += f"Task ID (for download_task_file if needed): {task_id}\n\n"
        prompt += f"Question: {question}"

        for attempt in range(2):
            try:
                raw = self.agent.run(prompt)
                if raw is None or str(raw).strip() in ("", "None"):
                    # Agent exhausted steps without calling final_answer.
                    # Try one quick LLM-only fallback to produce a guess.
                    answer = self._fallback_guess(question)
                else:
                    answer = self._clean(str(raw))
                print(f"Agent returning answer: {answer}")
                return answer
            except Exception as e:
                print(f"Agent error attempt {attempt+1}: {e}")
                if attempt < 1:
                    time.sleep(5)
                    continue
                return f"ERROR: {e}"
        return "ERROR: max retries exceeded"

    def _fallback_guess(self, question: str) -> str:
        """When the agent hits max_steps without answering, ask the model
        directly for a best-effort short answer. Better than submitting None."""
        try:
            fb_prompt = (
                "You are a GAIA assistant. The research agent could not complete this question. "
                "Provide your single best-guess answer in EXACT-MATCH format "
                "(digits only for numbers, no articles, no punctuation, no preamble). "
                f"Question: {question}\n\nAnswer:"
            )
            result = self.agent.model([{"role": "user", "content": fb_prompt}])
            text = result.content if hasattr(result, "content") else str(result)
            return self._clean(str(text))
        except Exception:
            return "unknown"

    @staticmethod
    def _clean(ans: str) -> str:
        ans = ans.strip()
        prefixes = [
            "FINAL ANSWER:", "Final Answer:", "Answer:",
            "The answer is:", "The answer is",
            "The final answer is:", "The final answer is",
        ]
        for p in prefixes:
            if ans.lower().startswith(p.lower()):
                ans = ans[len(p):].strip(" :.\"'")
        if len(ans) > 1 and ans[0] == ans[-1] and ans[0] in ('"', "'"):
            ans = ans[1:-1]
        if ans.endswith(".") and not ans.endswith("..") and len(ans.split()) <= 8:
            ans = ans.rstrip(".")
        return ans.strip()


def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    # Show how many Gemini keys were detected at startup
    print(f"🔑 Detected Gemini API keys: {len(_GEMINI_KEYS)}")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)