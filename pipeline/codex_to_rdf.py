#!/usr/bin/env python3
"""Convert Codex session JSONL logs to RDF Turtle using the devkg ontology.

Usage:
    # Structure only (no LLM calls)
    python -m pipeline.codex_to_rdf <input.jsonl> <output.ttl> --skip-extraction

    # Real-time extraction
    python -m pipeline.codex_to_rdf <input.jsonl> <output.ttl>
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pipeline.common import (
    DEVKG,
    PROV,
    add_triples_to_graph,
    create_developer_node,
    create_graph,
    create_message_node,
    create_model_node,
    create_project_node,
    create_session_node,
)
from pipeline.triple_extraction import cache_triples, extract_triples_gemini, get_cached_triples


def _extract_text_from_content(content) -> str:
    """Extract plain text from Codex message content blocks."""
    if isinstance(content, str):
        return content

    parts = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("text", "input_text", "output_text"):
                text = block.get("text", "")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def build_graph(jsonl_path: str, skip_extraction: bool = False, model=None, developer: str = "developer"):
    """Parse a Codex JSONL file and build an RDF graph."""
    g = create_graph()

    path = Path(jsonl_path).resolve()
    session_id = path.stem
    session_timestamp = None
    model_provider = None
    cwd = None

    developer_uri = create_developer_node(g, developer)

    entries = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] Skipping malformed JSON at line {line_num}", file=sys.stderr)
                continue

            outer_type = raw.get("type")
            payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
            ts = raw.get("timestamp") or payload.get("timestamp")

            if outer_type == "session_meta":
                session_id = payload.get("id", session_id)
                session_timestamp = payload.get("timestamp", session_timestamp)
                model_provider = payload.get("model_provider", model_provider)
                cwd = payload.get("cwd", cwd)
                continue

            if outer_type != "response_item":
                continue

            item_type = payload.get("type")
            if item_type == "message":
                entries.append({"timestamp": ts, "item": payload})

    if not entries:
        print("No Codex response items found.", file=sys.stderr)
        return g

    session_uri = create_session_node(
        g,
        session_id,
        "codex",
        created=session_timestamp,
        modified=entries[-1]["timestamp"] if entries and entries[-1].get("timestamp") else None,
        source_file=str(path),
    )
    g.add((session_uri, PROV.wasAssociatedWith, developer_uri))

    if model_provider:
        model_uri = create_model_node(g, f"codex-{model_provider}")
        g.add((session_uri, PROV.wasAssociatedWith, model_uri))

    if cwd:
        proj_uri = create_project_node(g, cwd, label=Path(cwd).name)
        g.add((session_uri, DEVKG.belongsToProject, proj_uri))

    user_count = 0
    assistant_count = 0
    triple_count = 0
    cache_hits = 0
    message_index = 0

    for i, row in enumerate(entries):
        item = row["item"]
        timestamp = row.get("timestamp")
        item_type = item.get("type")

        # Skip tool call / tool output items — not knowledge-graph material
        if item_type != "message":
            continue

        role = item.get("role")
        if role not in ("user", "assistant"):
            continue

        msg_id = item.get("id") or f"{session_id}-msg-{message_index}"
        message_index += 1
        full_text = _extract_text_from_content(item.get("content"))

        msg_uri = create_message_node(
            g,
            msg_id,
            role,
            session_uri,
            creator_uri=developer_uri if role == "user" else None,
            timestamp=timestamp,
            content=full_text if full_text.strip() else None,
            parent_uri=None,
        )

        if role == "user":
            user_count += 1
        else:
            assistant_count += 1

        if not skip_extraction and model is not None and role == "assistant" and full_text.strip():
            cached = get_cached_triples(msg_id)
            if cached is not None:
                triples = cached
                cache_hits += 1
            else:
                triples = extract_triples_gemini(
                    model,
                    full_text,
                    trace_metadata={
                        "source_platform": "codex",
                        "session_id": session_id,
                        "message_id": msg_id,
                        "source_file": str(path),
                        "project": Path(cwd).name if cwd else "",
                    },
                )
                cache_triples(msg_id, triples, full_text)
                time.sleep(0.5)

            add_triples_to_graph(g, msg_uri, triples, session_uri)
            triple_count += len(triples)

            if triples:
                label = "cached" if cached is not None else "extracted"
                print(f"  [{i+1}/{len(entries)}] {len(triples)} triples {label}", file=sys.stderr)

    cache_msg = f", {cache_hits} cache hits" if cache_hits else ""
    print(
        f"  Processed: {user_count} user messages, {assistant_count} assistant messages, "
        f"{triple_count} knowledge triples{cache_msg}",
        file=sys.stderr,
    )

    return g


def main():
    parser = argparse.ArgumentParser(description="Convert Codex JSONL to RDF Turtle")
    parser.add_argument("input", help="Path to Codex JSONL file")
    parser.add_argument("output", help="Path to output Turtle file")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip LLM triple extraction")
    parser.add_argument(
        "--provider",
        help="LLM provider: gemini, openai, anthropic, fireworks, ollama (auto-detect if omitted)",
    )
    parser.add_argument("--model", help="LLM model name override")
    args = parser.parse_args()

    model = None
    if not args.skip_extraction:
        from pipeline.llm_providers import get_extraction_model

        model = get_extraction_model(provider_name=args.provider, model_name=args.model)

    graph = build_graph(args.input, skip_extraction=args.skip_extraction, model=model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(out_path), format="turtle")
    print(f"Wrote {len(graph)} triples to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
