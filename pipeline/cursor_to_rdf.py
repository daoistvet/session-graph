#!/usr/bin/env python3
"""Convert Cursor agent-transcript JSONL logs to RDF Turtle using the devkg ontology.

Cursor transcripts live at:
  ~/.cursor/projects/{project-slug}/agent-transcripts/{uuid}/{uuid}.jsonl

Format (per line):
  {"role": "user"|"assistant", "message": {"content": [{"type": "text", "text": "..."}, ...]}}
  {"type": "turn_ended", "status": "..."}   # skipped

Usage:
    python -m pipeline.cursor_to_rdf <input.jsonl> <output.ttl> --skip-extraction
    python -m pipeline.cursor_to_rdf <input.jsonl> <output.ttl>
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


def detect_project(jsonl_path: str) -> str | None:
    """Detect project slug from a Cursor agent-transcript path.

    ~/.cursor/projects/{project-slug}/agent-transcripts/{uuid}/{uuid}.jsonl
    """
    parts = Path(jsonl_path).resolve().parts
    try:
        idx = parts.index("projects")
        # Prefer slug immediately under .cursor/projects
        if idx >= 1 and parts[idx - 1] == ".cursor" and idx + 1 < len(parts):
            return parts[idx + 1]
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1]
    except ValueError:
        pass
    return None


def _extract_text(content) -> str:
    """Return plain text from Cursor message content (ignore tool_use blocks)."""
    if isinstance(content, str):
        return content

    text_parts = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
    return "\n".join(text_parts)


def build_graph(jsonl_path: str, skip_extraction: bool = False, model=None, developer: str = "developer"):
    """Parse a Cursor agent-transcript JSONL file and build an RDF graph."""
    g = create_graph()
    path = Path(jsonl_path).resolve()
    session_id = path.stem
    # Prefer parent directory name when it matches the UUID layout
    if path.parent.name and path.parent.name != "agent-transcripts":
        session_id = path.parent.name

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

            role = raw.get("role")
            if role not in ("user", "assistant"):
                continue

            msg = raw.get("message") if isinstance(raw.get("message"), dict) else {}
            content = msg.get("content", "")
            full_text = _extract_text(content)
            msg_id = raw.get("id") or msg.get("id") or f"{session_id}-L{line_num}"

            entries.append({
                "line_num": line_num,
                "role": role,
                "id": msg_id,
                "text": full_text,
                "model": msg.get("model") or raw.get("model"),
                "timestamp": raw.get("timestamp") or msg.get("timestamp"),
            })

    if not entries:
        print("No user/assistant entries found.", file=sys.stderr)
        return g

    timestamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    session_uri = create_session_node(
        g,
        session_id,
        "cursor",
        created=timestamps[0] if timestamps else None,
        modified=timestamps[-1] if len(timestamps) > 1 else None,
        source_file=str(path),
    )
    g.add((session_uri, PROV.wasAssociatedWith, developer_uri))

    project_slug = detect_project(str(path))
    if project_slug:
        proj_uri = create_project_node(g, project_slug)
        g.add((session_uri, DEVKG.belongsToProject, proj_uri))

    models_seen = set()
    user_count = 0
    assistant_count = 0
    triple_count = 0
    cache_hits = 0
    prev_uri = None

    for i, entry in enumerate(entries):
        role = entry["role"]
        full_text = entry["text"]
        msg_uri = create_message_node(
            g,
            entry["id"],
            role,
            session_uri,
            creator_uri=developer_uri if role == "user" else None,
            timestamp=entry.get("timestamp"),
            content=full_text if full_text.strip() else None,
            parent_uri=prev_uri,
        )
        prev_uri = msg_uri

        if role == "user":
            user_count += 1
        else:
            assistant_count += 1
            model_id = entry.get("model")
            if model_id and model_id not in models_seen:
                models_seen.add(model_id)
                model_uri = create_model_node(g, model_id)
                g.add((session_uri, PROV.wasAssociatedWith, model_uri))

        if (
            not skip_extraction
            and model is not None
            and full_text.strip()
            and role == "assistant"
        ):
            cached = get_cached_triples(entry["id"])
            if cached is not None:
                triples = cached
                cache_hits += 1
            else:
                triples = extract_triples_gemini(
                    model,
                    full_text,
                    trace_metadata={
                        "source_platform": "cursor",
                        "session_id": session_id,
                        "message_id": entry["id"],
                        "source_file": str(path),
                        "project": project_slug or "",
                    },
                )
                cache_triples(entry["id"], triples, full_text)
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
    parser = argparse.ArgumentParser(description="Convert Cursor agent-transcript JSONL to RDF Turtle")
    parser.add_argument("input", help="Path to JSONL file")
    parser.add_argument("output", help="Path to output Turtle file")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip LLM triple extraction")
    parser.add_argument("--provider", help="LLM provider: gemini, openai, anthropic, fireworks, ollama")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--developer", default="developer", help="Developer name for provenance")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.skip_extraction:
        from pipeline.llm_providers import get_extraction_model
        model = get_extraction_model(provider_name=args.provider, model_name=args.model)

    print(f"Processing: {input_path}", file=sys.stderr)
    g = build_graph(str(input_path), skip_extraction=args.skip_extraction, model=model, developer=args.developer)
    print(f"  Total RDF triples: {len(g)}", file=sys.stderr)

    from pipeline.triple_extraction import get_truncation_count
    tc = get_truncation_count()
    if tc > 0:
        print(f"  Truncated responses: {tc} (salvaged where possible)", file=sys.stderr)

    print(f"  Writing to: {output_path}", file=sys.stderr)
    g.serialize(destination=str(output_path), format="turtle")
    print("  Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
