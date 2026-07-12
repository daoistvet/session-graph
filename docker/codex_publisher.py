#!/usr/bin/env python3
"""Publish new/updated Codex sessions to RabbitMQ for near-real-time ingestion.

Watches /codex-sessions/**/*.jsonl (mounted from ~/.codex/sessions) by periodic scan.
For each new or changed file, publishes a job with transcript_path + session_id
using the same queue contract as hooks/stop_hook.sh.
"""

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pika

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://devkg:devkg@rabbitmq:5672/")
QUEUE = os.environ.get("RABBITMQ_QUEUE", "devkg_jobs")
SESSIONS_GLOB = os.environ.get("CODEX_SESSIONS_GLOB", "/codex-sessions/**/*.jsonl")
POLL_INTERVAL_SECONDS = int(os.environ.get("CODEX_POLL_INTERVAL_SECONDS", "30"))
STATE_FILE = Path(os.environ.get("CODEX_WATCHER_STATE", "/app/output/codex/watcher_state.json"))


def log(level: str, msg: str):
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


def connect_with_retry(url: str, max_retries: int = 20) -> pika.BlockingConnection:
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            params = pika.URLParameters(url)
            params.heartbeat = 600
            params.blocked_connection_timeout = 300
            conn = pika.BlockingConnection(params)
            log("INFO", f"Connected to RabbitMQ (attempt {attempt})")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            if attempt == max_retries:
                raise
            log("WARN", f"RabbitMQ connect failed ({attempt}/{max_retries}): {e}")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("Unreachable")


def load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_session_id(path: str) -> str:
    # Prefer Codex session_meta.payload.id when available.
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "session_meta":
                    payload = obj.get("payload", {}) if isinstance(obj.get("payload"), dict) else {}
                    sid = payload.get("id")
                    if sid:
                        return sid
                    break
    except Exception:
        pass
    return Path(path).stem


def publish(channel, transcript_path: str, session_id: str):
    payload = json.dumps({
        "transcript_path": transcript_path,
        "session_id": session_id,
    })
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE,
        body=payload,
        properties=pika.BasicProperties(delivery_mode=2),
    )


def scan_once(channel, state: dict[str, str]) -> int:
    files = sorted(glob.glob(SESSIONS_GLOB, recursive=True))
    queued = 0

    for fp in files:
        try:
            current_hash = file_hash(fp)
        except FileNotFoundError:
            continue

        prev = state.get(fp)
        if prev == current_hash:
            continue

        sid = extract_session_id(fp)
        publish(channel, fp, sid)
        state[fp] = current_hash
        queued += 1
        log("QUEUE", f"{sid} -> {fp}")

    return queued


def main():
    state = load_state()

    conn = connect_with_retry(RABBITMQ_URL)
    channel = conn.channel()

    log("READY", f"Watching {SESSIONS_GLOB} every {POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            queued = scan_once(channel, state)
            if queued:
                save_state(state)
                log("INFO", f"Queued {queued} Codex session(s)")
        except Exception as e:
            log("ERROR", f"scan failed: {e}")
            # Attempt reconnect on broker problems.
            try:
                conn.close()
            except Exception:
                pass
            conn = connect_with_retry(RABBITMQ_URL)
            channel = conn.channel()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
