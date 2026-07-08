"""SQLite job log for MotionGen agent."""
import sqlite3, datetime, json, os
from pathlib import Path

DB_PATH = os.environ.get("MOTIONGEN_DB", str(Path(__file__).resolve().parent / "logs.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    status        TEXT NOT NULL,           -- queued|running|done|failed
    backend       TEXT NOT NULL,           -- higgsfield_browser|vast_comfyui|higgsfield_api
    preset        TEXT,
    prompt        TEXT,
    image_input   TEXT,
    output_path   TEXT,
    error         TEXT,
    duration_s    REAL,
    cost_usd      REAL,
    extra_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_backend ON jobs(backend);
"""

def conn():
    c = sqlite3.connect(DB_PATH)
    c.executescript(SCHEMA)
    return c

def now():
    return datetime.datetime.now(datetime.UTC).isoformat()

def add(backend, preset, prompt, image_input, extra=None):
    c = conn()
    cur = c.execute(
        "INSERT INTO jobs(created_at,updated_at,status,backend,preset,prompt,image_input,extra_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (now(), now(), "queued", backend, preset, prompt, image_input,
         json.dumps(extra or {}, ensure_ascii=False)))
    c.commit(); jid = cur.lastrowid; c.close(); return jid

def update(jid, **fields):
    if not fields: return
    c = conn()
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), jid))
    c.commit(); c.close()

def get(jid):
    c = conn()
    row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    cols = [d[0] for d in c.description]
    c.close()
    return dict(zip(cols, row)) if row else None
