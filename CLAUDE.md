# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

QuantView is a Chinese-language local data analysis assistant: a Flask backend (`app.py` + `file_processing.py`) serving a single-page vanilla-JS frontend (`static/index.html`). Users upload Excel/CSV/PDF files, ask a question in natural language, and get an AI-generated financial/business analysis report with matplotlib charts. There is no build step, no package manifest (no `requirements.txt`/`package.json`), and no test suite — dependencies are installed manually with pip.

## Running

```bash
python app.py        # serves http://localhost:5000 (or start.bat / start.ps1 on Windows)
./start.sh           # macOS / Linux：./start.sh | master | node [port]
```

Startup behavior depends on mode (see Environment variables):

- **Normal mode**: requires a GGUF model in `models/`. Spawns the external `llama-server` at startup (first load takes 10–30s). Exits with an error if no model is found.
- **Debug mode** (`DEEPANALYZE_DEBUG=true`): skips local models entirely and calls the DeepSeek API — this is the fastest way to develop/test without GPU or local model weights. Always sets `DEEPSEEK_API_KEY` in this mode.

Useful deps: `flask pandas matplotlib numpy` (required); `openpyxl` (required — pandas reads .xlsx with it), `xlrd` (.xls); `pdfplumber` (PDF tables, optional); `python-docx` (Word export, optional). No torch/transformers — local inference is GGUF-only via the external `llama-server` binary.

## Environment variables

| Variable | Effect |
|---|---|
| `DEEPANALYZE_DEBUG` | `1/true` → use DeepSeek API instead of local models |
| `DEEPSEEK_API_KEY` | Required for DeepSeek API and debug mode |
| `DEEPSEEK_API_URL` | OpenAI-compatible endpoint override (default `https://api.deepseek.com/chat/completions`) — also points at the local llama-server when GGUF is used |
| `DEEPSEEK_THINKING` | `1/true` → use `deepseek-reasoner` (streams `reasoning_content` as "think" events) |
| `DEEPANALYZE_MODEL` | Force-select a model by name from the `models/` scan |
| `LLAMA_SERVER_PATH` | Path to llama-server binary. Default auto-search: `<项目目录>/llama-server/llama-server` (or `.exe`), project root, PATH. The binary is placed manually in the `llama-server/` folder (Windows: llama-server.exe, macOS/Linux: llama-server, macOS Metal build recommended) |
| `DEEPANALYZE_NODES` | `name=url,name=url` 加速节点列表 — 设置了即为**主节点（分布式）模式**，工作表任务轮流分发，主节点只生成总览；不设置则为单节点模式 |
| `DEEPANALYZE_NODE_TIMEOUT` | 单次节点调用超时秒数（默认 600） |
| `DEEPANALYZE_PORT` | 服务端口（默认 5000；同机跑主/加速双实例时用 `start_node.bat` 的 5001） |
| `DEEPANALYZE_TEMPERATURE` | 采样温度（默认 0.5；报告为确定性任务，低温度降低单次跑飞/对话式输出概率） |
| `DEEPANALYZE_HEADLESS` | `1/true` → 无头加速节点模式：不提供 Web 界面（`GET /` 返回 404 提示），仅保留 `/analyze/sheet` 任务接口，控制台逐任务打印收到/完成/耗时 |
| `DEEPANALYZE_CONTEXT` | llama-server 上下文长度（默认 65536）。同机多实例时加速节点建议 32768 省内存（32B 模型 64K KV 缓存约 16GB） |
| `DEEPANALYZE_NGL` | llama-server GPU 显存层数（默认 99 全量卸载）。显存不足报 `ErrorOutOfDeviceMemory`/KV 缓存分配失败时调低（如 60/30），0 = 纯 CPU |

## Architecture

### Inference backends (2 paths, all behind `_run_inference`)

1. **DeepSeek API** (debug mode): `_call_deepseek_api` / `_call_deepseek_api_stream` — plain `urllib` POST to an OpenAI-compatible `/chat/completions` endpoint. The stream variant yields `("think", ...)` and `("text", ...)` tuples.
2. **GGUF via external llama-server**: spawns `llama-server -m <model> --port N -ngl 99 -c <ctx>` as a subprocess (port auto-increments from 8080 if busy), waits on `/health`, then sets `DEEPSEEK_API_URL` to the local server so **the same API call path is reused**. Killed via `atexit`. HF/transformers route removed — GGUF only.

Both are wrapped by **`_run_inference(prompt, max_tokens, stream)`** — the single inference entry point: `stream=False` returns text, `stream=True` returns a `("think"|"text", chunk)` generator. New code paths (e.g. `/analyze/sheet`) call it instead of reaching into backends directly.

### Model discovery (`_scan_models`, `_select_model`)

Scans `models/` for `.gguf` files (root or inside dirs). HF-only directories are skipped with a hint. Selection priority: `DEEPANALYZE_MODEL` env var → auto-select if exactly one → interactive numbered prompt.

### Analysis data pipeline (`/analyze` and `/analyze/stream`)

All file parsing lives in **`file_processing.py`** (no Flask dependency; pure pandas). The first two steps below live there, the last in `app.py`:

1. `read_file` — Excel: all sheets merged into one DataFrame with a `_sheet` column; PDF: pdfplumber tables merged with `_pdf_page`/`_pdf_table` columns; CSV: plain read.
2. `df_summary` — builds a text summary: column dtype/null/uniqueness overview, per-sheet head/tail samples, text-column value distributions with row ranges, per-sheet numeric stats, missing values. This summary (not raw data) is what the model sees.
3. `extract_hard_numbers_core` (also in `file_processing.py`) — finance-specific: re-reads Excel bytes to extract 合计/总计-style rows as "hard numbers" with normalized period labels (`2026H1`, `2026Q3`, 期末/年初 etc., see `_norm_period`). Line format is `指标名(期别=数值)` — the closing paren and the "负号只在数值本身、括号内是期别" warning in the prompt are deliberate anti-misreading protections; keep them when editing prompt logic. When present, the prompt marks hard numbers as authoritative ("禁止修改，必须原样引用") and **omits raw head/tail samples** so the model can't invent figures. This is an anti-hallucination mechanism — preserve it when editing prompt logic.
4. `_prepare_distributed_input_impl` (in `app.py`) — splits every file into **per-sheet tasks** (CSV/PDF = whole file); this is the ONLY analysis path now (single-node = the master's own model processes sheets one by one). The old single giant-prompt path was removed.

### Per-sheet pipeline (single-node and distributed)

All analysis goes through the same per-sheet pipeline; `DEEPANALYZE_NODES` adds accelerator nodes. Flow (streaming and non-streaming paths both support it):

1. `_prepare_distributed_input_impl` — per-sheet split: each sheet of each Excel file becomes one task (CSV/PDF = whole file, one task). Per-sheet summary from `df_summary` on the filtered DataFrame; per-sheet hard numbers from `extract_hard_numbers_by_sheet` (parses `extract_hard_numbers_core` output by `【工作表名】` headers).
2. `_build_sheet_prompt` / `_build_overview_prompt` — compact prompts preserving the anti-hallucination rules (hard numbers must be quoted verbatim, no fabricated periods, `chartjson` required at the end).
3. `_distributed_analysis_events` (SSE) / `_perform_analysis_distributed` (non-streaming) — round-robin across **all nodes including the master itself** (`_work_nodes`/`_submit_node_task`). The streaming path is **parallel-generation, ordered-delivery**: each task runs in a `threading.Thread` (`_stream_section_worker`) that pumps text chunks into a **per-task `queue.Queue`** — local tasks iterate `_run_inference(stream=True)`, accelerator nodes read `/analyze/sheet/stream` SSE — and the master generator consumes task queues **in task order**, so each section's chunks stream contiguously (tables never get split by other sections' chunks) while all nodes still generate in parallel. The first chunk of each section carries the `### 工作表「x」分析（节点：y）` heading. Failed nodes degrade to a ⚠️ failure section instead of aborting the report.
4. After all sections: the master's own model runs the overview prompt (`_run_inference`), producing 总览/综合结论. The final report = per-sheet sections + overview, and all `chartjson` blocks from all parts are rendered (multiple blocks supported via `finditer`).
5. Without `DEEPANALYZE_NODES`, `_work_nodes()` is just the master itself — sheets are processed one by one by the local model, with the same per-sheet quality benefits (short focused prompts instead of one giant prompt).

Each node is just another `app.py` instance — it needs its own model (or `DEEPANALYZE_DEBUG` + API key) and exposes `/analyze/sheet` automatically. Workers must NOT set `DEEPANALYZE_NODES` (avoid recursive distribution).

### Charts

- AI-directed: `_generate_charts_from_json` extracts **all** ```chartjson``` code blocks from the model output (`[{"title", "type": bar|pie|bar_h|line, "data": {label: value}}]`) and renders each spec to base64 PNG (multiple blocks — one per distributed section/overview — are merged; bad blocks are skipped).
- Fallback `_generate_charts`: automatic heuristics — detects 对比 pairs (期末/年初, 预算/实际, 本期/同期…), scores numeric columns (coefficient of variation × coverage), draws histograms/rankings/category frequencies.
- Chinese font handling (`_ensure_chinese_font`): project `fonts/` dir → known Windows/Linux/macOS paths → `fc-list` → auto-download Noto Sans SC into `fonts/`. If no font is found, chart text renders as boxes. Matplotlib must use `Agg` backend.

### Routes

| Route | Purpose |
|---|---|
| `GET /` | Serve `index.html` (no-cache headers) |
| `POST /analyze` | Multipart `files` + `question` → `{result, images, mode, nodes}` (non-streaming; single-node and distributed) |
| `POST /analyze/stream` | Same input, SSE stream of `{type: meta|progress|text|think|charts|error|done}` events (single-node and distributed). `meta` carries a `session` id for follow-up chat |
| `POST /analyze/followup` | **二次追问**: JSON `{session, question}` → SSE `{type: text|think|error|done}`. Answers with the session's data context (hard numbers + file summaries) + conversation history via `_build_followup_prompt`; sessions kept in-memory (`_SESSIONS`, last 20) |
| `POST /analyze/sheet` | **Accelerator-node job endpoint**: JSON `{prompt, max_tokens}` → `{result}`. Calls `_run_inference` non-streaming |
| `POST /analyze/sheet/stream` | **Accelerator-node streaming endpoint**: same input, SSE `{type: text|think|error|done}` events; master forwards chunks to the browser so 中栏 cards fill in real time |
| `POST /export/docx` | Receives HTML report JSON, spawns `export_docx.py` as a subprocess, returns the .docx |

### Frontend (`static/index.html`)

Single file, zero external libraries — CSS variables for dark/light themes, hand-rolled `renderMarkdown`, fetch + `ReadableStream` parsing of the SSE stream (note: not `EventSource`). No browser-side session history (the left-panel history list was removed as bug-prone); the report lives only in the current page. Live think-block rendering, abort controller, and throttled (150ms) markdown re-render during streaming. A mode badge (`.analysis-meta`) renders the SSE `meta` event (单节点/分布式 + node names) and live `progress` events (分项分析 N/M). The main area is a three-pane grid: left = input (upload/question/loading), middle = 节点工作内容 (per-node section cards built from the `### 工作表「x」分析（节点：y）` headings in distributed text events), right = final report + charts + 追问 chat panel. `index.html.bak` is a stale backup, not served.

### Word export (`export_docx.py`)

Standalone CLI: `python export_docx.py input.html output.docx [--title X]`; converts a subset of HTML (h1–h4, tables, p/li, hr) to a styled .docx via python-docx. Invoked by Flask through `subprocess` with temp files; also runnable directly from stdin.

## Gotchas

- Comments, prompts, and UI strings are all in Chinese — keep new prompts/reports in Chinese.
- torch/transformers/pdfplumber/llama-cpp are optional and must stay lazily imported (module import must not fail without them).
- `DEBUG_MODE` (not Flask's `debug`) gates model loading; Flask's own `debug=False` in `__main__`.
- Model behavior is shaped by the per-sheet prompts (`_build_sheet_prompt`) and the overview prompt (`_build_overview_prompt`) — the 铁律 (iron rules) in these prompts are the core product logic.
- Upload limit is 200MB (`MAX_CONTENT_LENGTH`); allowed extensions are xlsx/xls/csv/pdf.
