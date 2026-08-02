# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeepAnalyze is a Chinese-language local data analysis assistant: a Flask backend (`app.py` + `file_processing.py`) serving a single-page vanilla-JS frontend (`static/index.html`). Users upload Excel/CSV/PDF files, ask a question in natural language, and get an AI-generated financial/business analysis report with matplotlib charts. There is no build step, no package manifest (no `requirements.txt`/`package.json`), and no test suite — dependencies are installed manually with pip.

## Running

```bash
python app.py        # serves http://localhost:5000 (or start.bat / start.ps1 on Windows)
```

Startup behavior depends on mode (see Environment variables):

- **Normal mode**: requires a model in `models/`. Lazy-loads torch/transformers at startup (first load takes 10–30s). Exits with an error if no model is found.
- **Debug mode** (`DEEPANALYZE_DEBUG=true`): skips local models entirely and calls the DeepSeek API — this is the fastest way to develop/test without GPU or local model weights. Always sets `DEEPSEEK_API_KEY` in this mode.

Useful deps: `flask pandas matplotlib numpy` (required); `pdfplumber` (PDF tables, optional); `python-docx` (Word export, optional); `torch transformers` (HF models, lazy-imported); `llama-cpp-python` (embedded GGUF, lazy-imported).

## Environment variables

| Variable | Effect |
|---|---|
| `DEEPANALYZE_DEBUG` | `1/true` → use DeepSeek API instead of local models |
| `DEEPSEEK_API_KEY` | Required for DeepSeek API and debug mode |
| `DEEPSEEK_API_URL` | OpenAI-compatible endpoint override (default `https://api.deepseek.com/chat/completions`) — also points at the local llama-server when GGUF is used |
| `DEEPSEEK_THINKING` | `1/true` → use `deepseek-reasoner` (streams `reasoning_content` as "think" events) |
| `DEEPANALYZE_MODEL` | Force-select a model by name from the `models/` scan |
| `LLAMA_SERVER_PATH` | Path to llama-server binary (default auto-search, incl. `C:/Users/tc191/llama-cpp/llama-server.exe`) |

## Architecture

### Inference backends (3 paths, all behind `get_model_and_tokenizer()`)

1. **DeepSeek API** (debug mode): `_call_deepseek_api` / `_call_deepseek_api_stream` — plain `urllib` POST to an OpenAI-compatible `/chat/completions` endpoint. The stream variant yields `("think", ...)` and `("text", ...)` tuples.
2. **GGUF via external llama-server**: spawns `llama-server -m <model> --port N -ngl 99 -c 65536` as a subprocess (port auto-increments from 8080 if busy), waits on `/health`, then sets `DEEPSEEK_API_URL` to the local server so **the same API call path is reused**. Killed via `atexit`.
3. **HuggingFace via transformers**: lazily imported inside the load function (torch/transformers are not imported at module load). Streams via `TextIteratorStreamer` + thread.

### Model discovery (`_scan_models`, `_select_model`)

Scans `models/` for: HF directories (signed by `config.json`/`safetensors`/`tokenizer.json` etc.) and `.gguf` files (root or inside dirs). Selection priority: `DEEPANALYZE_MODEL` env var → auto-select if exactly one → interactive numbered prompt.

### Analysis data pipeline (`/analyze` and `/analyze/stream`)

All file parsing lives in **`file_processing.py`** (no Flask dependency; pure pandas). The first two steps below live there, the last in `app.py`:

1. `read_file` — Excel: all sheets merged into one DataFrame with a `_sheet` column; PDF: pdfplumber tables merged with `_pdf_page`/`_pdf_table` columns; CSV: plain read.
2. `df_summary` — builds a text summary: column dtype/null/uniqueness overview, per-sheet head/tail samples, text-column value distributions with row ranges, per-sheet numeric stats, missing values. This summary (not raw data) is what the model sees.
3. `extract_hard_numbers_core` (also in `file_processing.py`) — finance-specific: re-reads Excel bytes to extract 合计/总计-style rows as "hard numbers" with normalized period labels (`2026H1`, `2026Q3`, 期末/年初 etc., see `_norm_period`). When present, the prompt marks them as authoritative ("禁止修改，必须原样引用") and **omits raw head/tail samples** so the model can't invent figures. This is an anti-hallucination mechanism — preserve it when editing prompt logic.
4. `_prepare_analysis_input_impl` (in `app.py`) — assembles the giant Chinese system prompt: hard numbers first, then per-file summaries, then the user question, then a strict analysis protocol (per-section deep-dive, requirement checklist table, "no vague numbers" rules) and a requirement for a `chartjson` block at the end. File buffers are re-read from saved bytes because `FileStorage` streams are single-use.

### Charts

- AI-directed: `_generate_charts_from_json` extracts a ```chartjson``` code block from the model output (`[{"title", "type": bar|pie|bar_h|line, "data": {label: value}}]`) and renders each spec to base64 PNG.
- Fallback `_generate_charts`: automatic heuristics — detects 对比 pairs (期末/年初, 预算/实际, 本期/同期…), scores numeric columns (coefficient of variation × coverage), draws histograms/rankings/category frequencies.
- Chinese font handling (`_ensure_chinese_font`): project `fonts/` dir → known Windows/Linux/macOS paths → `fc-list` → auto-download Noto Sans SC into `fonts/`. If no font is found, chart text renders as boxes. Matplotlib must use `Agg` backend.

### Routes

| Route | Purpose |
|---|---|
| `GET /` | Serve `index.html` (no-cache headers) |
| `POST /analyze` | Multipart `files` + `question` → `{result, images}` (non-streaming) |
| `POST /analyze/stream` | Same input, SSE stream of `{type: text|think|charts|error|done}` events |
| `POST /export/docx` | Receives HTML report JSON, spawns `export_docx.py` as a subprocess, returns the .docx |

### Frontend (`static/index.html`)

Single file, zero external libraries — CSS variables for dark/light themes, hand-rolled `renderMarkdown`, fetch + `ReadableStream` parsing of the SSE stream (note: not `EventSource`). Sessions persist to `localStorage` (last 50, key `SESSIONS_KEY`), with live think-block rendering, abort controller, and throttled (150ms) markdown re-render during streaming. `index.html.bak` is a stale backup, not served.

### Word export (`export_docx.py`)

Standalone CLI: `python export_docx.py input.html output.docx [--title X]`; converts a subset of HTML (h1–h4, tables, p/li, hr) to a styled .docx via python-docx. Invoked by Flask through `subprocess` with temp files; also runnable directly from stdin.

## Gotchas

- Comments, prompts, and UI strings are all in Chinese — keep new prompts/reports in Chinese.
- torch/transformers/pdfplumber/llama-cpp are optional and must stay lazily imported (module import must not fail without them).
- `DEBUG_MODE` (not Flask's `debug`) gates model loading; Flask's own `debug=False` in `__main__`.
- The giant prompt in `_prepare_analysis_input_impl` is the core product logic — model behavior is shaped almost entirely there, not in code.
- Upload limit is 200MB (`MAX_CONTENT_LENGTH`); allowed extensions are xlsx/xls/csv/pdf.
