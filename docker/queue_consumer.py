#!/usr/bin/env python3
"""RabbitMQ consumer for DevKG pipeline jobs.

Listens on the `devkg_jobs` queue, processes Claude Code session transcripts
into RDF Turtle, and uploads them to Fuseki.
"""

import base64
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pika


def _init_vertex_credentials():
    """Decode GOOGLE_APPLICATION_CREDENTIALS_BASE64 to a temp file if present."""
    b64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if not b64 or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    decoded = base64.b64decode(b64).decode("utf-8")
    fd, path = tempfile.mkstemp(suffix=".json", prefix="gcp-creds-")
    os.write(fd, decoded.encode())
    os.close(fd)
    os.chmod(path, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


_init_vertex_credentials()


RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://devkg:devkg@localhost:5672/")
FUSEKI_URL = os.environ.get("FUSEKI_URL", "http://localhost:3030")
FUSEKI_DATASET = os.environ.get("FUSEKI_DATASET", "devkg")
FUSEKI_USER = os.environ.get("FUSEKI_USER", "admin")
FUSEKI_PASS = os.environ.get("FUSEKI_PASS", "admin")
FUSEKI_AUTH = (FUSEKI_USER, FUSEKI_PASS)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/output/claude"))

QUEUE = "devkg_jobs"
DLX = "devkg_jobs_dlx"
DLQ = "devkg_jobs_failed"
WATERMARK_FILE = OUTPUT_DIR / "watermarks.json"


def log(level: str, msg: str):
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


def write_turtle_atomic(graph, output_file: Path) -> None:
    """Serialize Turtle to a temp file and replace atomically on success."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{output_file.stem}.", suffix=".ttl.tmp", dir=output_file.parent)
    os.close(fd)
    try:
        graph.serialize(destination=tmp_path, format="turtle")
        os.replace(tmp_path, output_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def file_hash(path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_watermarks() -> dict:
    """Load watermark file mapping session_id -> content hash."""
    if not WATERMARK_FILE.exists():
        return {}
    with open(WATERMARK_FILE) as f:
        return json.load(f)


def save_watermarks(watermarks: dict) -> None:
    """Save watermark state."""
    WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WATERMARK_FILE, "w") as f:
        json.dump(watermarks, f, indent=2)


def connect_with_retry(url: str, max_retries: int = 10) -> pika.BlockingConnection:
    """Connect to RabbitMQ with exponential backoff."""
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
            log("WARN", f"Connection failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def setup_queues(channel):
    """Declare the main queue with dead-letter exchange."""
    # Dead-letter exchange + queue
    channel.exchange_declare(exchange=DLX, exchange_type="fanout", durable=True)
    channel.queue_declare(queue=DLQ, durable=True)
    channel.queue_bind(queue=DLQ, exchange=DLX)

    # Main queue with DLX
    channel.queue_declare(
        queue=QUEUE,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX},
    )
    channel.basic_qos(prefetch_count=1)


def translate_path(host_path: str) -> str:
    """Translate host path to container path.

    Host: ~/.claude/projects/{slug}/{session}.jsonl
    Container: /claude-sessions/{slug}/{session}.jsonl

    Host: ~/.pi/agent/sessions/{slug}/{session}.jsonl
    Container: /pi-sessions/{slug}/{session}.jsonl

    Host: ~/.codex/sessions/{yyyy}/{mm}/{dd}/{session}.jsonl
    Container: /codex-sessions/{yyyy}/{mm}/{dd}/{session}.jsonl
    """
    marker_claude = "/projects/"
    idx = host_path.find(marker_claude)
    if idx != -1:
        return "/claude-sessions" + host_path[idx + len("/projects") :]

    marker_pi = "/sessions/"
    idx = host_path.find(marker_pi)
    if idx != -1 and ".pi" in host_path:
        return "/pi-sessions" + host_path[idx + len("/sessions") :]

    idx = host_path.find("/.codex/sessions/")
    if idx != -1:
        return "/codex-sessions" + host_path[idx + len("/.codex/sessions") :]

    return host_path  # can't translate, return as-is


CAG_SERVER_URL = os.environ.get("CAG_SERVER_URL", "http://localhost:4000")


def post_cag_triples(session_id: str, triples: list[dict], replace: bool = False) -> None:
    """POST extracted CAG triples to the compounding-agents server. Non-fatal.

    Args:
        session_id: The session identifier.
        triples: List of triple dicts to insert.
        replace: If True, the server deletes existing session triples before inserting.
    """
    if not triples:
        return
    import requests
    url = f"{CAG_SERVER_URL}/hooks/cag-triples"
    try:
        resp = requests.post(url, json={
            "session_id": session_id,
            "triples": triples,
            "replace": replace,
        }, timeout=5)
        if resp.ok:
            result = resp.json()
            log("INFO", f"  CAG triples posted: {result.get('inserted', 0)} (replace={replace})")
        else:
            log("WARN", f"  CAG POST failed: {resp.status_code}")
    except Exception as e:
        log("WARN", f"  CAG server unreachable ({url}): {e}")


def extract_last_assistant_text(transcript_path: str) -> str:
    """Extract only the last assistant message text from a transcript.

    Supports Claude JSONL, pi JSONL, and Codex JSONL formats.
    """
    last_text = ""
    is_pi_session = "/pi-sessions/" in transcript_path or ".pi/agent/sessions" in transcript_path
    is_codex_session = "/codex-sessions/" in transcript_path or ".codex/sessions" in transcript_path

    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if is_pi_session:
                if entry_type != "message":
                    continue
                msg = entry.get("message", {})
                role = msg.get("role")
                if role != "assistant":
                    continue
                content = msg.get("content", "")
            elif is_codex_session:
                if entry_type != "response_item":
                    continue
                payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else entry
                if payload.get("type") != "message" or payload.get("role") != "assistant":
                    continue
                content = payload.get("content", "")
            else:
                if entry_type != "assistant":
                    continue
                content = entry.get("message", {}).get("content", "")

            parts = []
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    btype = block.get("type") if isinstance(block, dict) else None
                    if btype in ("text", "output_text"):
                        parts.append(block.get("text", ""))
            text = "\n\n".join(p for p in parts if p.strip())
            if text.strip():
                last_text = text
    return last_text


def process_message(body: bytes) -> None:
    """Process a single pipeline job."""
    msg = json.loads(body)
    transcript_path = msg.get("transcript_path", "")
    session_id = msg.get("session_id", "")

    if not transcript_path:
        log("WARN", "Message missing transcript_path, skipping")
        return

    # Skip subagent sessions
    if "/subagents/" in transcript_path:
        log("INFO", f"Skipping subagent session: {session_id}")
        return

    # Translate host path to container path
    container_path = translate_path(transcript_path)

    if not os.path.exists(container_path):
        raise FileNotFoundError(f"Transcript not found: {container_path} (host: {transcript_path})")

    # Derive output filename
    basename = session_id or Path(container_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{basename}.ttl"

    # SHA256 watermark: skip if file content hasn't changed since last processing
    watermarks = load_watermarks()
    current_hash = file_hash(container_path)
    if watermarks.get(basename) == current_hash:
        log("SKIP", f"{basename} (unchanged since last processing)")
        return

    log("INFO", f"Processing: {basename}")

    # Phase 6e: Query compounding-agents server for session correlation context
    orch_session_id = None
    try:
        import requests
        sid = session_id or basename
        resp = requests.get(f"{CAG_SERVER_URL}/api/session-context/{sid}", timeout=3)
        if resp.ok:
            ctx = resp.json()
            orch_session_id = ctx.get("orchSessionId")
            if orch_session_id:
                log("INFO", f"  Correlated with orchestrator session: {orch_session_id}")
    except Exception:
        pass  # Graceful degradation — proceed without correlation

    # Import pipeline modules (deferred to avoid import errors during setup)
    from pipeline.llm_providers import get_provider

    model = get_provider()

    # Run devkg and CAG extraction in parallel
    devkg_triple_count = 0
    cag_triple_count = 0

    def _run_devkg():
        nonlocal devkg_triple_count
        
        if "/pi-sessions/" in container_path or ".pi/agent/sessions" in container_path:
            from pipeline.pi_to_rdf import build_graph
        elif "/codex-sessions/" in container_path or ".codex/sessions" in container_path:
            from pipeline.codex_to_rdf import build_graph
        else:
            from pipeline.jsonl_to_rdf import build_graph
            
        from pipeline.load_fuseki import ensure_dataset, upload_turtle

        graph = build_graph(container_path, skip_extraction=False, model=model)

        # Phase 6e: Tag session node with correlation ID if available
        if orch_session_id:
            from pipeline.common import DATA, DEVKG, slug as uri_slug
            from rdflib import Literal
            session_uri = DATA[f"session/{uri_slug(session_id or basename)}"]
            graph.add((session_uri, DEVKG.hasCorrelationId, Literal(orch_session_id)))

        if os.environ.get("DEVKG_SKIP_LINKING", "").lower() not in ("1", "true", "yes"):
            try:
                from pipeline.link_entities import link_entities_into_graph
                link_stats = link_entities_into_graph(graph)
                log(
                    "INFO",
                    "  wikidata: "
                    f"{link_stats['linked']} linked "
                    f"({link_stats['cache_hits']} cache, "
                    f"{link_stats['agentic_calls']} agentic, "
                    f"{link_stats['negative_cache_hits']} negative, "
                    f"{link_stats['skipped']} skipped)",
                )
            except Exception as e:
                log("WARN", f"  wikidata linking failed (continuing): {e}")

        write_turtle_atomic(graph, output_file)
        devkg_triple_count = len(graph)

        ensure_dataset(FUSEKI_URL, FUSEKI_DATASET, auth=FUSEKI_AUTH)
        upload_turtle(FUSEKI_URL, FUSEKI_DATASET, str(output_file), auth=FUSEKI_AUTH)
        log("DONE", f"  devkg: {devkg_triple_count} triples -> {output_file}")

    def _run_cag():
        nonlocal cag_triple_count
        from pipeline.cag_extraction import extract_cag_triples
        from pipeline.cag_cache import get_cached_cag, cache_cag

        sid = session_id or basename
        last_msg = extract_last_assistant_text(container_path)
        if not last_msg or len(last_msg.strip()) <= 50:
            return
        prior_triples = get_cached_cag(sid)
        cag_triples = extract_cag_triples(model, last_msg, prior_triples=prior_triples)
        if cag_triples:
            cag_triple_count = len(cag_triples)
            log("INFO", f"  CAG extracted: {cag_triple_count} triples (prior={len(prior_triples or [])})")
            post_cag_triples(sid, cag_triples, replace=True)
            cache_cag(sid, cag_triples)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_devkg): "devkg",
            executor.submit(_run_cag): "cag",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                log("WARN" if name == "cag" else "ERROR", f"  {name} extraction failed: {e}")
                if name == "devkg":
                    raise  # devkg failure is fatal for the job

    # Notify pipeline completion (non-fatal)
    try:
        import requests
        requests.post(
            f"{CAG_SERVER_URL}/hooks/pipeline-complete",
            json={
                "session_id": session_id or basename,
                "devkg_triples": devkg_triple_count,
                "cag_triples": cag_triple_count,
            },
            timeout=5,
        )
    except Exception:
        pass

    # Update watermark after successful processing
    watermarks[basename] = current_hash
    save_watermarks(watermarks)

    log("DONE", f"{basename} ({devkg_triple_count} devkg + {cag_triple_count} cag triples)")


def on_message(channel, method, properties, body):
    """RabbitMQ message callback."""
    try:
        process_message(body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        try:
            msg = json.loads(body) if body else {}
            session_id = msg.get("session_id", "unknown")
        except Exception:
            session_id = body[:80].decode(errors="replace") if body else "unknown"
        log("ERROR", f"{session_id}: {e}")
        traceback.print_exc(file=sys.stderr)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    connection = connect_with_retry(RABBITMQ_URL)
    channel = connection.channel()
    setup_queues(channel)

    channel.basic_consume(queue=QUEUE, on_message_callback=on_message)

    log("READY", f"Waiting for jobs on {QUEUE}")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        log("INFO", "Shutting down")
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
