#!/usr/bin/env python3
"""Real E2E: process session 019f4ecb through queue_consumer.process_message."""

import json
import os
import sys
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF, OWL

# Must run with PYTHONPATH=/app inside pipeline-runner
from docker.queue_consumer import process_message, OUTPUT_DIR, WATERMARK_FILE
from pipeline.common import DEVKG

SESSION_ID = "019f4ecb-c0f7-79c1-ac67-1a1cdf577159"
HOST_PATH = (
    "/Users/robertoshimizu/.codex/sessions/2026/07/10/"
    "rollout-2026-07-10T22-30-03-019f4ecb-c0f7-79c1-ac67-1a1cdf577159.jsonl"
)
CONTAINER_PATH = (
    "/codex-sessions/2026/07/10/"
    "rollout-2026-07-10T22-30-03-019f4ecb-c0f7-79c1-ac67-1a1cdf577159.jsonl"
)


def main() -> int:
    print("=== REAL E2E ===")
    print(f"session_id: {SESSION_ID}")
    print(f"transcript exists: {os.path.exists(CONTAINER_PATH)}")
    if not os.path.exists(CONTAINER_PATH):
        print("FATAL: transcript not found in container mount", file=sys.stderr)
        return 1

    # Clear watermark so consumer reprocesses
    if WATERMARK_FILE.exists():
        wm = json.loads(WATERMARK_FILE.read_text())
        removed = wm.pop(SESSION_ID, None)
        WATERMARK_FILE.write_text(json.dumps(wm, indent=2))
        print(f"cleared watermark: {bool(removed)}")
    else:
        print("no watermark file")

    out = OUTPUT_DIR / f"{SESSION_ID}.ttl"
    before_sameas = 0
    before_kt = 0
    if out.exists():
        g0 = Graph()
        g0.parse(str(out), format="turtle")
        before_sameas = sum(1 for _ in g0.triples((None, OWL.sameAs, None)))
        before_kt = sum(1 for _ in g0.triples((None, RDF.type, DEVKG.KnowledgeTriple)))
    print(f"BEFORE ttl sameAs={before_sameas} knowledge_triples={before_kt}")

    body = json.dumps({
        "transcript_path": HOST_PATH,
        "session_id": SESSION_ID,
    }).encode()

    print("\n=== RUNNING process_message (extract + wikidata link + fuseki) ===")
    process_message(body)
    print("=== process_message returned ===\n")

    if not out.exists():
        print("FATAL: output TTL missing after process", file=sys.stderr)
        return 1

    g = Graph()
    g.parse(str(out), format="turtle")
    kt = sum(1 for _ in g.triples((None, RDF.type, DEVKG.KnowledgeTriple)))
    ents = sum(1 for _ in g.triples((None, RDF.type, DEVKG.Entity)))
    sameas = sum(1 for _ in g.triples((None, OWL.sameAs, None)))
    platform = next(
        (str(o) for s, _, _ in g.triples((None, RDF.type, DEVKG.Session))
         for _, _, o in g.triples((s, DEVKG.hasSourcePlatform, None))),
        "?",
    )

    print("=== RESULT ===")
    print(f"session_id: {SESSION_ID}")
    print(f"platform: {platform}")
    print(f"output: {out}")
    print(f"knowledge_triples: {kt}")
    print(f"entities: {ents}")
    print(f"owl:sameAs in TTL: {sameas} (was {before_sameas})")
    print(f"ttl size bytes: {out.stat().st_size}")

    if sameas <= before_sameas:
        print("FAIL: no new Wikidata links written to TTL", file=sys.stderr)
        return 1
    print("PASS: real session processed with Wikidata links in TTL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
