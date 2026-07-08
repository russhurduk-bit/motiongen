# MotionGen — Telegram Motion-Transfer Bot

A Telegram bot that turns **a reference photo + a reference video** into a
motion-transfer clip: the movement (pose, face, gesture) from the video is
transferred onto the character in the photo, and the finished clip is returned
as a Telegram document.

The bot is backend-agnostic. It ships with two interchangeable generation
backends behind a small abstraction layer:

- **Vast.ai + ComfyUI** running **Wan 2.2 Animate** — rents a GPU on demand,
  provisions ComfyUI, submits a workflow, streams progress over WebSocket, and
  destroys the instance when done. Cheap, self-hosted, fully scriptable.
- **Browser automation fallback** — drives a hosted motion-control web UI
  (Higgsfield / Kling) through Playwright when a managed API is preferred.

## Features

- **Conversational flow** built on `aiogram` FSM: `/start` → pick backend →
  upload photo → upload video → cost preview → confirm → result.
- **Cost preview** before every run, derived from the reference-video duration.
- **Per-user daily credit cap** with a small SQLite usage ledger (`/stats`).
- **Global generation lock** — only one generation runs at a time (`asyncio.Lock`),
  so a single GPU is never double-booked.
- **Access control** — an allow-list of Telegram user IDs from the environment.
- **Automatic post-processing** in `engine.py`:
  - **Aspect alignment** — the result is scaled/padded to match the source
    photo's aspect ratio (no crop), computed from `ffprobe`.
  - **Audio mux** — the original soundtrack from the reference video is muxed
    back onto the silent generated clip via `ffmpeg`.
  - **Document delivery** — the file is sent as a Telegram *document* so it is
    not re-compressed.

## Architecture

```
bot.py                 aiogram FSM, access control, credit cap, cost preview
engine.py              ffprobe/ffmpeg post-processing + backend dispatch
db.py                  SQLite job log
backends/
  vast_helper.py       Vast.ai REST: search offers, create/destroy, wait-running
  comfy_client.py      ComfyUI REST/WS client: upload, submit, progress, collect
  vast_comfyui.py      Orchestrates a full Vast + ComfyUI generation
  higgsfield_browser.py  Playwright browser backend (fallback)
workflows/
  workflow_parser.py   Auto-detects node roles and patches a ComfyUI API graph
  example_wan22.json   Example Wan 2.2 workflow (drop your own default.json here)
build_custom_v1..v3.py Programmatic builders for custom Wan Animate workflows
```

### Workflow auto-patching

`workflows/workflow_parser.py` takes any ComfyUI *API-format* workflow and
locates the image loader, video loader, positive/negative text encoders and
samplers by `class_type` and node title heuristics — then injects the uploaded
image/video, prompts and seed. This lets the same code drive many different
Wan Animate graphs without hand-editing node IDs. It also validates a workflow's
`class_type`s against a live instance's `/object_info` so missing custom nodes
fail fast with a clear message.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install aiogram python-dotenv playwright requests websockets
playwright install chromium   # only needed for the browser backend

cp .env.example .env          # then fill in the values
python bot.py
```

Drop a ComfyUI API-format workflow at `workflows/default.json` to use the
Vast backend (`example_wan22.json` shows the expected shape).

## Configuration

All secrets and machine-specific paths come from the environment — see
`.env.example`. Nothing sensitive is committed.

---

*Portfolio note: this repository is a sanitized extract of a private project.
Large binary assets (model workflows, media, browser profiles) are excluded.*
