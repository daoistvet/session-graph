#!/usr/bin/env python3
"""Convert ChatGPT/OpenAI conversation exports to RDF Turtle using the devkg ontology.

ChatGPT exports contain multiple conversation files (conversations-000.json through
conversations-NNN.json), each holding up to 100 conversations. This parser uses a
global index across all files for consistency.

Usage:
    # List all conversations (across all files in directory)
    python -m pipeline.chatgpt_to_rdf <export_dir> <output.ttl>

    # Process a specific conversation by global index
    python -m pipeline.chatgpt_to_rdf <export_dir> <output.ttl> --conversation 42

    # Skip triple extraction (structure only)
    python -m pipeline.chatgpt_to_rdf <export_dir> <output.ttl> --conversation 42 --skip-extraction

    # Custom model
    python -m pipeline.chatgpt_to_rdf <export_dir> <output.ttl> --conversation 42 --model gemini-2.5-pro

    # Filter by date (only conversations from 2025 onwards)
    python -m pipeline.chatgpt_to_rdf <export_dir> <output.ttl> --date-from 2025-01-01
"""

import json
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Literal
from rdflib.namespace import RDF, RDFS, DCTERMS, XSD

from pipeline.common import (
    PROV, SIOC, DEVKG, DATA,
    slug, create_graph, create_session_node, create_developer_node,
    create_model_node, create_message_node, add_triples_to_graph,
)
from pipeline.triple_extraction import (
    extract_triples_gemini, get_cached_triples, cache_triples,
    get_truncation_count,
)


# =============================================================================
# Data Loading
# =============================================================================

def find_conversation_files(export_dir: str) -> list[Path]:
    """Find all conversations-NNN.json files in the export directory, sorted."""
    export_path = Path(export_dir)
    files = sorted(export_path.glob("conversations-*.json"))
    return files


def load_all_conversations(export_dir: str) -> list[dict]:
    """Load all conversations from all conversation files, preserving order.

    Returns a flat list of conversation dicts across all files.
    """
    files = find_conversation_files(export_dir)
    if not files:
        print(f"Error: No conversations-*.json files found in {export_dir}", file=sys.stderr)
        sys.exit(1)

    all_convs = []
    for f in files:
        with open(f) as fh:
            convs = json.load(fh)
        all_convs.extend(convs)

    return all_convs


def filter_conversations_by_date(
    conversations: list[dict],
    date_from: str | None = None,
) -> list[tuple[int, dict]]:
    """Filter conversations by creation date, returning (original_index, conv) tuples.

    Args:
        date_from: ISO date string like '2025-01-01'. Only conversations created
                   on or after this date are included.
    """
    if date_from is None:
        return list(enumerate(conversations))

    cutoff = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    filtered = []
    for i, conv in enumerate(conversations):
        create_time = conv.get("create_time")
        if create_time is not None and create_time >= cutoff:
            filtered.append((i, conv))
    return filtered


# =============================================================================
# Timestamp Normalization
# =============================================================================

def normalize_timestamp(ts: float | None) -> str | None:
    """Convert a Unix epoch float to ISO 8601 UTC string.

    ChatGPT uses timestamps like 1711022696.888573.
    """
    if ts is None:
        return None
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError, OSError):
        return None


# =============================================================================
# Tree Walking
# =============================================================================

def walk_conversation_tree(mapping: dict) -> list[dict]:
    """Walk the ChatGPT conversation tree depth-first, returning ordered messages.

    ChatGPT uses a tree structure similar to DeepSeek, with a mapping dict where
    each node has id, parent, children, and an optional message.

    Each returned dict has: id, role, content, model, timestamp, parent_id.
    Only returns messages with extractable text content (skips system_error,
    execution_output, tether_browsing_display, tether_quote, and image parts).
    """
    messages = []

    def walk(node_id: str, parent_msg_id: str | None):
        node = mapping.get(node_id)
        if node is None:
            return

        msg = node.get("message")
        current_msg_id = None

        if msg:
            author = msg.get("author", {})
            role = author.get("role", "")
            content = msg.get("content", {})
            content_type = content.get("content_type", "")
            msg_id = msg.get("id", node_id)
            create_time = msg.get("create_time")
            metadata = msg.get("metadata", {})
            model_slug = metadata.get("model_slug")

            # Only process text content from user and assistant roles
            if role in ("user", "assistant") and content_type == "text":
                parts = content.get("parts", [])
                # Extract only string parts (skip image dicts and other non-text)
                text_parts = [p for p in parts if isinstance(p, str)]
                text = "\n".join(text_parts).strip()

                if text:
                    messages.append({
                        "id": msg_id,
                        "role": role,
                        "content": text,
                        "model": model_slug,
                        "timestamp": create_time,
                        "parent_id": parent_msg_id,
                    })
                    current_msg_id = msg_id

        # Use the current message ID as parent for children, or pass through
        next_parent = current_msg_id or parent_msg_id

        # Recurse into children
        children = node.get("children", [])
        for child_id in children:
            walk(child_id, next_parent)

    # Find root node (parent is None)
    for nid, node in mapping.items():
        if node.get("parent") is None:
            walk(nid, None)
            break

    return messages


# =============================================================================
# Graph Building
# =============================================================================

def build_graph(
    conversation: dict,
    export_dir: str,
    skip_extraction: bool = False,
    model=None,
    developer: str = "developer",
):
    """Build an RDF graph from a single ChatGPT conversation."""
    g = create_graph()

    # Developer node
    developer_uri = create_developer_node(g, developer)

    # Session node
    conv_id = conversation.get("id") or conversation.get("conversation_id", "unknown")
    title = conversation.get("title")
    created = normalize_timestamp(conversation.get("create_time"))
    modified = normalize_timestamp(conversation.get("update_time"))

    session_uri = create_session_node(
        g, conv_id, "chatgpt",
        created=created,
        modified=modified,
        title=title,
        source_file=str(Path(export_dir).resolve()),
    )
    g.add((session_uri, PROV.wasAssociatedWith, developer_uri))

    # Record the default model for the conversation if available
    default_model = conversation.get("default_model_slug")
    if default_model:
        model_node_uri = create_model_node(g, default_model)
        g.add((session_uri, PROV.wasAssociatedWith, model_node_uri))

    # Walk conversation tree
    mapping = conversation.get("mapping", {})
    messages = walk_conversation_tree(mapping)

    if not messages:
        print("  No messages found in conversation.", file=sys.stderr)
        return g

    # Track models and build URI lookup for parent references
    models_seen = {default_model} if default_model else set()
    id_to_uri = {}

    user_count = 0
    assistant_count = 0
    triple_count = 0
    cache_hits = 0

    for i, msg in enumerate(messages):
        msg_id = msg["id"]
        role = msg["role"]
        content = msg["content"]
        timestamp = normalize_timestamp(msg["timestamp"])
        msg_model = msg.get("model")

        # Resolve parent URI
        parent_uri = None
        if msg["parent_id"]:
            parent_uri = id_to_uri.get(msg["parent_id"])

        # Create message node with a globally unique ID
        global_msg_id = f"cgpt-{slug(conv_id[:12])}-{msg_id}"
        msg_uri = create_message_node(
            g, global_msg_id, role, session_uri,
            creator_uri=developer_uri if role == "user" else None,
            timestamp=timestamp,
            content=content if content.strip() else None,
            parent_uri=parent_uri,
        )
        id_to_uri[msg_id] = msg_uri

        if role == "user":
            user_count += 1
        else:
            assistant_count += 1
            if msg_model and msg_model not in models_seen:
                models_seen.add(msg_model)
                model_uri = create_model_node(g, msg_model)
                g.add((session_uri, PROV.wasAssociatedWith, model_uri))

        # Triple extraction (assistant messages only)
        if not skip_extraction and model is not None and content.strip() and role == "assistant":
            cached = get_cached_triples(global_msg_id)
            if cached is not None:
                triples = cached
                cache_hits += 1
            else:
                triples = extract_triples_gemini(model, content)
                cache_triples(global_msg_id, triples, content)
                time.sleep(0.5)

            add_triples_to_graph(g, msg_uri, triples, session_uri)
            triple_count += len(triples)

            if triples:
                label = "cached" if cached is not None else "extracted"
                print(f"  [{i+1}/{len(messages)}] {len(triples)} triples {label}",
                      file=sys.stderr)

    cache_msg = f", {cache_hits} cache hits" if cache_hits else ""
    print(f"  Processed: {user_count} user messages, {assistant_count} assistant messages, "
          f"{triple_count} knowledge triples{cache_msg}", file=sys.stderr)

    return g


# =============================================================================
# CLI
# =============================================================================

def list_conversations(filtered: list[tuple[int, dict]], date_from: str | None = None) -> None:
    """Print a table of conversations with global indices and summary stats.

    Args:
        filtered: List of (original_index, conversation) tuples.
        date_from: Date filter string for display purposes.
    """
    filter_msg = f" (from {date_from})" if date_from else ""
    print(f"\n{len(filtered)} conversations{filter_msg}:\n")
    print(f"{'#':>5}  {'Date':>10}  {'User':>4}  {'Asst':>4}  {'Model':>12}  Title")
    print(f"{'─'*5}  {'─'*10}  {'─'*4}  {'─'*4}  {'─'*12}  {'─'*50}")

    total_user = 0
    total_assistant = 0

    for orig_idx, conv in filtered:
        title = conv.get("title", "(untitled)")
        create_time = conv.get("create_time")
        if create_time:
            created = datetime.fromtimestamp(create_time, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            created = ""
        default_model = conv.get("default_model_slug") or ""

        # Count text messages by role
        mapping = conv.get("mapping", {})
        user_count = 0
        asst_count = 0
        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role", "")
            ct = msg.get("content", {}).get("content_type", "")
            if ct == "text":
                if role == "user":
                    user_count += 1
                elif role == "assistant":
                    asst_count += 1

        total_user += user_count
        total_assistant += asst_count

        print(f"{orig_idx:>5}  {created:>10}  {user_count:>4}  {asst_count:>4}  {default_model:>12}  {title[:60]}")

    print(f"\nTotals: {len(filtered)} conversations, {total_user} user messages, {total_assistant} assistant messages")
    print(f"Use --conversation N to process a specific conversation.")


def main():
    parser = argparse.ArgumentParser(description="Convert ChatGPT conversation exports to RDF Turtle")
    parser.add_argument("input", help="Path to ChatGPT export directory (contains conversations-*.json)")
    parser.add_argument("output", help="Path to output Turtle file")
    parser.add_argument("--conversation", type=int, default=None,
                        help="Global conversation index to process (omit to list all)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip LLM triple extraction")
    parser.add_argument("--provider", help="LLM provider: gemini, openai, anthropic, fireworks, ollama (auto-detect if omitted)")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--date-from", default=None,
                        help="Only include conversations from this date onwards (YYYY-MM-DD)")
    parser.add_argument("--developer", default="developer", help="Developer name for provenance (default: developer)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_dir():
        print(f"Error: Input must be a directory containing conversations-*.json files: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {input_path}", file=sys.stderr)
    conversations = load_all_conversations(str(input_path))
    print(f"  Found {len(conversations)} conversations across {len(find_conversation_files(str(input_path)))} files", file=sys.stderr)

    # Apply date filter
    filtered = filter_conversations_by_date(conversations, args.date_from)
    if args.date_from:
        print(f"  After date filter (>= {args.date_from}): {len(filtered)} conversations", file=sys.stderr)

    # List mode
    if args.conversation is None:
        list_conversations(filtered, date_from=args.date_from)
        sys.exit(0)

    # Validate conversation index (uses original global index)
    if args.conversation < 0 or args.conversation >= len(conversations):
        print(
            f"Error: Conversation index {args.conversation} out of range "
            f"(0-{len(conversations) - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check if the selected conversation passes the date filter
    if args.date_from:
        valid_indices = {idx for idx, _ in filtered}
        if args.conversation not in valid_indices:
            print(
                f"Warning: Conversation {args.conversation} is before {args.date_from}, "
                f"processing anyway since explicitly requested.",
                file=sys.stderr,
            )

    conv = conversations[args.conversation]
    print(f"  Selected: [{args.conversation}] {conv.get('title', '(untitled)')}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize LLM provider
    llm_model = None
    if not args.skip_extraction:
        from pipeline.llm_providers import get_extraction_model
        llm_model = get_extraction_model(provider_name=args.provider, model_name=args.model)

    g = build_graph(
        conv, str(input_path),
        skip_extraction=args.skip_extraction,
        model=llm_model,
        developer=args.developer,
    )

    print(f"  Total RDF triples: {len(g)}", file=sys.stderr)

    # Report truncation events if any occurred
    tc = get_truncation_count()
    if tc > 0:
        print(f"  Truncated responses: {tc} (salvaged where possible)", file=sys.stderr)

    print(f"  Writing to: {output_path}", file=sys.stderr)

    g.serialize(destination=str(output_path), format="turtle")
    print("  Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
