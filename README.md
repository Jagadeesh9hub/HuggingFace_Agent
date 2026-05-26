# GAIA Agent — HuggingFace Agents Course Unit 4

A multi-tool AI agent built for the [HuggingFace Agents Course](https://huggingface.co/learn/agents-course) Unit 4 final assignment. It answers questions from the GAIA benchmark by reasoning step-by-step and calling specialized tools (Wikipedia, web fetch, YouTube transcripts, Gemini Vision/Audio, Excel/CSV parsing).

**Result:** passed the certification with a score above the 30% threshold required for the course completion certificate.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture overview](#architecture-overview)
- [Technical walkthrough](#technical-walkthrough)
- [Problems encountered & how they were solved](#problems-encountered--how-they-were-solved)
- [Tech stack](#tech-stack)
- [Running locally](#running-locally)
- [File structure](#file-structure)
- [Lessons learned](#lessons-learned)

---

## What it does

GAIA is a benchmark of 20 real-world questions that require an agent to combine reasoning, tool use, and multimodal understanding. Example types of questions:

- Factual lookups requiring Wikipedia traversal ("How many studio albums did Mercedes Sosa release between 2000–2009?")
- Multi-step web research ("What is the nationality of the youngest winner of...?")
- YouTube video understanding ("What does the narrator say at 1:30 in this video?")
- Image and audio analysis (chess board positions, ingredient lists in cooking videos)
- Numeric reasoning over uploaded Excel/CSV files
- Reversed-text trick questions

The agent decides which tool to use for each question, fetches the relevant information, reasons through the answer, and returns a single exact-match string. Everything runs from a Gradio UI that fetches the question set, runs the agent on each one, and POSTs the answers to the official scoring endpoint.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│ Gradio UI (gr.Blocks)                                    │
│  └─ run_and_submit_all()                                 │
│      • fetches 20 questions from scoring API             │
│      • runs agent per question                           │
│      • POSTs combined answers, displays score            │
│                                                          │
│      └─ BasicAgent.__call__()                            │
│          • assembles system prompt + question            │
│          • handles reversed-text trick                   │
│          • outer retry loop (defense-in-depth)           │
│          • routes incomplete reasoning to fallback       │
│                                                          │
│          └─ smolagents.CodeAgent                         │
│              • the actual think-execute-repeat loop      │
│              • max_steps=4                               │
│                                                          │
│              ├─ RotatingLiteLLMModel                     │
│              │   • subclass of LiteLLMModel              │
│              │   • multi-key rotation                    │
│              │   • dead-key tracking                     │
│              │   • three-way error classification        │
│              │                                           │
│              └─ Tool[] (11 tools)                        │
│                  Wikipedia, web, YouTube, vision,        │
│                  audio, Excel, CSV, DuckDuckGo,          │
│                  Python interpreter, file download...    │
└──────────────────────────────────────────────────────────┘
```

---

## Technical walkthrough

A section-by-section description of what `app.py` actually does.

### 1. Key loading and global state

```python
def _load_gemini_keys() -> list[str]: ...
_GEMINI_KEYS = _load_gemini_keys()
_GEMINI_KEY_IDX = 0
```

Reads Gemini API keys from environment variables (`GEMINI_API_KEY`, plus optional `GEMINI_API_KEY_2..6`). The list and rotation index are module-level globals so that the multimodal tools (which call Gemini independently of the main LLM) share the same rotation state.

### 2. `RotatingLiteLLMModel` — the rotating LLM client

This is the most substantial piece of original engineering in the file. It subclasses `smolagents.LiteLLMModel` and adds:

- **Multi-key state.** A list of API keys plus an index pointing at the current one. Switching keys is a single attribute assignment (`self.api_key = ...`) rather than re-running the parent constructor, which avoids triggering side effects in LiteLLM's initialization.

- **Dead-key tracking.** A `set` of indices for keys that have permanently failed this session (e.g. `403 PERMISSION_DENIED`). Future rotations skip these, so a banned key never costs more than its initial 0.2-second failure.

- **Three-way error classification.** Every exception is categorized:
  - `"dead"` — permanent failures (`403`, `permission_denied`, `api_key_invalid`). Mark key dead, never retry it.
  - `"rotate"` — transient (`429`, `quota`, `resource_exhausted`, `timeout`, `5xx`). Switch keys and retry.
  - `"raise"` — unrelated to keys (bad request format, model not found). Propagate immediately.

- **Backoff after full cycle.** If every alive key has been tried in one rotation cycle without success, the loop sleeps for the server-suggested `retryDelay` (parsed from the JSON error body via regex) or a 45s fallback. This prevents a tight loop from burning through all keys instantly.

- **Override of both `generate()` and `__call__()`.** smolagents calls `model.generate(...)`, not `model(...)`, so overriding only `__call__` was a silent no-op in the first version. Both are now wrapped through a shared `_rotating_call` helper.

- **Diagnostic logging.** Every model call prints structured events (`[GENERATE]`, `[GEN-DONE]`, `[ROTATE]`, `[KEY-DEAD]`, `[ROTATE-TRIGGER]`, `[BACKOFF]`, `[FATAL]`) so rotation behavior is observable in production logs.

### 3. Tool classes

Each tool inherits from `smolagents.Tool` and exposes a `name`, `description`, `inputs` schema, `output_type`, and a `forward()` method. These metadata fields get serialized into the agent's system prompt so the LLM knows the tools exist and how to call them.

- `VisitWebpageTool` — generic HTTP fetch with markdownification; truncates to 6KB for normal sites and 15KB for Wikipedia.
- `WikipediaSearchTool` — wraps `en.wikipedia.org/w/api.php` for top-3 article summaries with URLs.
- `WikipediaGetArticleTool` — fetches the full plain-text extract of a Wikipedia article by exact title. Used for list/count questions where summaries aren't enough.
- `DownloadTaskFileTool` — pulls GAIA attachments from the scoring API's `/files/{task_id}` endpoint to `/tmp/`.
- `ReadTextFileTool` / `ReadExcelFileTool` — local file readers with truncation.
- `YouTubeTranscriptTool` — three fallback strategies: new `youtube-transcript-api` (v1.x), older API (0.6.x), then Gemini multimodal video understanding.
- `GeminiVisionTool` / `GeminiAudioTool` — upload image/audio to Gemini Files API, then ask Gemini with key rotation.
- `DuckDuckGoSearchTool` / `PythonInterpreterTool` — provided by smolagents.

### 4. System prompt (`GAIA_INSTRUCTIONS`)

A tuned prompt that tells the model the exact-match grading rules ("digits only for numbers, no articles, no trailing punctuation"), which tool to pick for which question type, how to behave when a site blocks the agent (try Wikipedia instead, don't retry 403s), and the hard rule: **always call `final_answer()` with something** — even a wrong guess beats `None`.

### 5. `BasicAgent` — the wrapper

A thin layer around `smolagents.CodeAgent`:

- Detects the GAIA reversed-text trick and prepends a decoded version to the prompt.
- Wraps `agent.run()` in an outer retry loop (2 attempts) as defense-in-depth against rotation-loop failures.
- Detects when the agent returned mid-reasoning text instead of a clean final answer (i.e. hit `max_steps` mid-thought), and routes those cases to `_fallback_guess()`.
- `_fallback_guess()` asks the LLM directly for a single-shot best guess.
- `_clean()` strips common verbiage like "FINAL ANSWER:", surrounding quotes, and trailing periods so the exact-match grader doesn't mark right-but-formatted-wrong answers.

### 6. `run_and_submit_all` — the orchestrator

Plain procedural code: get logged-in HF username, instantiate the agent, GET 20 questions, loop and collect answers, POST to `/submit`, format the result for the Gradio UI.

### 7. Gradio UI

Single-button interface (`gr.LoginButton` + run button + status textbox + results table). The login button auto-injects an `OAuthProfile` argument into the orchestrator function, which is how the user's HF username gets passed without explicit wiring.

---

## Problems encountered & how they were solved

This is the part most useful to other developers building agents on free-tier APIs. Each problem below cost real debugging time.

### Problem 1 — Rotation was a silent no-op

**Symptom.** First version of `RotatingLiteLLMModel` overrode `__call__`. Rate-limit errors hit the agent's outer error handler instead of triggering rotation. No `[ROTATE]` logs ever appeared.

**Diagnosis.** smolagents' `CodeAgent` doesn't call `model(...)`. It calls `model.generate(...)`. The canonical entrypoint in smolagents' `Model` base class is `generate()`, not `__call__()`.

**Fix.** Override `generate()` with the correct smolagents signature (`messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs`). Kept `__call__` overridden too for backward compatibility with the `_fallback_guess` path.

### Problem 2 — LiteLLM's internal retry loop was blocking rotation

**Symptom.** Even after the `generate()` fix, single model calls were hanging for 12–15 minutes before raising. Rotation worked but only after these massive delays. With multiple rate-limit-prone keys, this made single questions take an hour.

**Diagnosis.** When a Gemini call returns a 429, the response includes a `retryDelay` (typically 45s). LiteLLM respects that hint by sleeping internally, and will attempt multiple internal retries before raising the exception up. The default `num_retries` is non-zero.

**Fix.** Pass `num_retries=0` to the parent constructor, along with `timeout=120` and `request_timeout=120` as additional safety nets. (The timeout parameters were partially ignored by LiteLLM in practice — see Problem 5.)

### Problem 3 — Permanent key failures were treated as temporary

**Symptom.** A specific Gemini project returned `403 PERMISSION_DENIED`. The rotation loop kept cycling back to it, wasting 0.2s each time. Worse, when the original code re-raised on non-rate-limit errors, the first 403 killed the whole question.

**Diagnosis.** Two-way classification (rate-limit vs. everything-else) wasn't enough. Some errors are permanent and key-specific (the key is invalid/banned), but those shouldn't propagate up — they should just remove the bad key from rotation.

**Fix.** Three-way classification (`_classify_error` returns `"dead"`, `"rotate"`, or `"raise"`). Added `_dead_keys: set` to track permanently failed keys, and `_alive_count()` so the rotation logic knows how many usable keys remain. Updated `_switch_to_next_key()` to skip dead-marked indices.

### Problem 4 — Agent returned reasoning text as the final answer

**Symptom.** Some questions came back with the answer field containing things like:
> "The `visit_webpage` tool returned truncated content for the everydaypie.com link. This means I cannot reliably extract the ingredients from this page. I need to try another source..."

This was the agent's internal thinking, not a final answer. Happens when the agent hits `max_steps=4` mid-reasoning before calling `final_answer()`. The grader marked all these wrong even though the agent had often done useful work.

**Diagnosis.** `agent.run()` returns whatever the last action was when `max_steps` is hit. If the last action was a Python code block or a thought, that gets returned as the "answer."

**Fix.** Added `_looks_like_thinking()` heuristic that detects mid-reasoning output (length > 250 chars, contains ` ``` `, "I need to", "I will", or any tool name with a parenthesis). When detected, route to `_fallback_guess()` which asks the model directly for a clean one-shot answer.

### Problem 5 — LiteLLM ignored `timeout` parameter

**Symptom.** Despite `timeout=120` and `request_timeout=120` in the parent constructor, some calls still hung for 700+ seconds.

**Workaround.** Couldn't fully solve from inside our subclass. Mitigated by reducing `max_steps` from 6 to 4 to cap total quota burn per question, and switching the main agent to `gemini-2.5-flash-lite` (4× the free-tier daily quota of Flash) to reduce rate-limit pressure in the first place. The remaining occasional long calls were tolerable because dead-key tracking prevented compounding delays.

### Problem 6 — Grading server connection drops

**Symptom.** First successful end-to-end run failed at the submission step with HTTP 500 and "Remote end closed connection without response." All 20 answers were lost (they were only in memory).

**Resolution.** Server-side flakiness on the grading endpoint — unrelated to the agent code. Retried the run later and submission succeeded.

### Problem 7 — Agent narration leaked into the answer string

**Symptom.** Even on questions the agent solved correctly, the answer would sometimes be `"FINAL ANSWER: Mercedes Sosa"` or `"The answer is 42."` — wrong format for an exact-match grader.

**Fix.** `_clean()` strips common prefixes (`"Final Answer:"`, `"Answer:"`, `"The answer is"`, etc.), surrounding quote characters, and trailing periods (but only if the answer is short enough that the period clearly isn't part of the content).

---

## Tech stack

- **[smolagents](https://github.com/huggingface/smolagents)** — agent loop (think → write Python → execute → repeat)
- **[LiteLLM](https://github.com/BerriAI/litellm)** — model provider abstraction
- **Gemini 2.5 Flash-Lite** — main reasoning model (chosen for higher free-tier daily quota)
- **Gemini 2.5 Flash** — multimodal tools (Vision, Audio, YouTube video understanding)
- **Gradio** — web UI
- **HuggingFace Spaces** — deployment target

---

## Running locally

Requires Python 3.10+ and at least one Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/jagadeeshnalluri99/HuggingFace_Agent.git
cd HuggingFace_Agent
pip install -r requirements.txt

export GEMINI_API_KEY=your_key_here
# Optional: additional keys for rotation
# export GEMINI_API_KEY_2=...

python app.py
```

Then open the Gradio URL printed in the terminal.

---

## File structure

| File | Purpose |
|------|---------|
| `app.py` | Everything: tools, rotating LLM client, agent class, Gradio UI |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Standard Python gitignore |
| `README.md` | This file |

---

## Lessons learned

A few takeaways worth carrying into other LLM-agent projects:

- **Override the canonical method, not the convenience method.** Frameworks usually have one method that everything else routes through. Find it. Overriding the wrong one is a silent no-op.

- **Wrapping libraries have their own retry loops.** LiteLLM, the OpenAI SDK, and most other LLM wrapper libraries try to be helpful by retrying internally. If you're implementing your own retry/rotation logic, disable theirs first or you'll get compounding delays.

- **Classify errors more granularly than "succeeded" vs "failed."** At minimum, separate permanent failures (don't retry) from transient ones (do retry) from unrelated errors (propagate). Lumping them all together leads to either infinite retry loops or premature give-ups.

- **Make the rotation observable.** Without structured logs at every state transition, debugging "is rotation even happening?" requires guesswork. A few well-placed print statements cost almost nothing and save hours when things go wrong.

- **The system prompt is the highest-leverage piece.** Tightening the GAIA-specific instructions moved more accuracy than any code change in this project.

- **Always submit something.** Agents hit edge cases. Building a fallback path that produces a best-guess answer when the main path fails is the difference between "0 correct" and "maybe 1 correct" — and over 20 questions, that matters.

---

## Author

Built by [Jagadeesh Nalluri](https://github.com/jagadeeshnalluri99) as part of the HuggingFace AI Agents certification program.
