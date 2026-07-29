#!/usr/bin/env python3
"""Local side-by-side benchmark of cheaper LLM candidates for DevKG.

Runs the 4 real assistant messages (Claude, pi, Codex, Cursor) through:
  1. Triple extraction — records triple count + sample per model per message.
  2. Wikidata linking — records QID + latency per model per message.

LangSmith traces are tagged `benchmark` + `model:<name>` so runs are filterable
in the UI. Uses the project's configured LANGCHAIN_PROJECT (do not override).

Results are saved incrementally to _scripts/benchmark_results.json so partial
results survive an abort.

Usage:
    cd <repo root>
    source .venv/bin/activate
    python _scripts/benchmark_models.py            # all candidates
    python _scripts/benchmark_models.py --only gemini-2.5-flash-lite,deepseek-v4-flash
    python _scripts/benchmark_models.py --skip-linking   # extraction only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make pipeline importable when run from repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

# Tracing on by default; user can disable via env
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

from pipeline.llm_providers import get_provider, reset_extraction_model  # noqa: E402
import pipeline.agentic_linker_langgraph as linker  # noqa: E402
from pipeline.triple_extraction import extract_triples_gemini  # noqa: E402

# ---------------------------------------------------------------------------
# Candidate models — all cheaper than current production on at least one axis.
# Current: extraction=gemini-2.5-flash ($0.30/$2.50), linking=gemini-3-flash-preview ($0.50/$3.00)
# Pricing: $/1M tokens (input/output), verified from official pricing pages.
# ---------------------------------------------------------------------------
CANDIDATES = [
    # --- Gemini Flash-Lite family (cheapest Gemini) ---
    {
        "id": "gemini-2.5-flash-lite",
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite",
        "price_in": 0.10, "price_out": 0.40,
        "workloads": ["extraction", "linking"],
        "note": "cheapest overall; 2.5 deprecation path",
    },
    {
        "id": "gemini-3.1-flash-lite",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "price_in": 0.25, "price_out": 1.50,
        "workloads": ["extraction", "linking"],
        "note": "newest 3.x lite",
    },
    {
        "id": "gemini-3.5-flash-lite",
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "price_in": 0.30, "price_out": 2.50,
        "workloads": ["extraction", "linking"],
        "note": "same in-cost as current, cheaper out",
    },
    # --- OpenAI cheap tier ---
    {
        "id": "gpt-5.4-nano",
        "provider": "openai",
        "model": "gpt-5.4-nano",
        "price_in": 0.20, "price_out": 1.25,
        "workloads": ["extraction", "linking"],
        "note": "best OpenAI value",
    },
    {
        "id": "gpt-5.4-mini",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "price_in": 0.75, "price_out": 4.50,
        "workloads": ["extraction", "linking"],
        "note": "quality reference (more expensive)",
    },
    # --- Mistral ---
    {
        "id": "mistral-small-4",
        "provider": "mistral",
        "model": "mistral-small-latest",
        "price_in": 0.15, "price_out": 0.60,
        "workloads": ["extraction", "linking"],
        "note": "open-weight, agentic, very cheap",
    },
    {
        "id": "mistral-large-3",
        "provider": "mistral",
        "model": "mistral-large-latest",
        "price_in": 0.50, "price_out": 1.50,
        "workloads": ["linking"],
        "note": "flagship open-weight",
    },
    # --- Fireworks cheap tier ---
    {
        "id": "deepseek-v4-flash",
        "provider": "fireworks",
        "model": "accounts/fireworks/models/deepseek-v4-flash",
        "price_in": 0.14, "price_out": 0.28,
        "workloads": ["extraction", "linking"],
        "note": "cheapest non-Gemini",
    },
    {
        "id": "gpt-oss-120b",
        "provider": "fireworks",
        "model": "accounts/fireworks/models/gpt-oss-120b",
        "price_in": 0.15, "price_out": 0.60,
        "workloads": ["extraction", "linking"],
        "note": "OpenAI open-weight on Fireworks",
    },
    {
        "id": "qwen3p7-plus",
        "provider": "fireworks",
        "model": "accounts/fireworks/models/qwen3p7-plus",
        "price_in": 0.40, "price_out": 1.60,
        "workloads": ["linking"],
        "note": "strong agentic",
    },
    {
        "id": "minimax-m3",
        "provider": "fireworks",
        "model": "accounts/fireworks/models/minimax-m3",
        "price_in": 0.30, "price_out": 1.20,
        "workloads": ["extraction", "linking"],
        "note": "balanced",
    },
    {
        "id": "glm-5p2",
        "provider": "fireworks",
        "model": "accounts/fireworks/models/glm-5p2",
        "price_in": 1.40, "price_out": 4.40,
        "workloads": ["linking"],
        "note": "already tested; quality reference",
    },
]

# Current production baseline for comparison
BASELINES = {
    "extraction": {"id": "gemini-2.5-flash", "price_in": 0.30, "price_out": 2.50},
    "linking": {"id": "gemini-3-flash-preview", "price_in": 0.50, "price_out": 3.00},
}

RESULTS_PATH = REPO / "_scripts" / "benchmark_results.json"
PROBES_PATH = Path("/tmp/devkg-langsmith-probes.json")


# ---------------------------------------------------------------------------
# Probe selection — one real assistant message per platform
# ---------------------------------------------------------------------------
def select_probes() -> list[dict]:
    """Load cached probes or regenerate from real session files."""
    if PROBES_PATH.exists():
        return json.loads(PROBES_PATH.read_text())

    from glob import glob

    def text_from(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text")
            )
        return ""

    def newest(pattern):
        return sorted(
            (Path(p) for p in glob(str(Path.home() / pattern), recursive=True)
             if "/subagents/" not in p),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    out = []

    # Claude
    for p in newest(".claude/projects/**/*.jsonl"):
        sid = None
        for line in p.open(errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = sid or r.get("sessionId")
            if r.get("type") == "assistant":
                t = text_from((r.get("message") or {}).get("content"))
                if len(t.strip()) >= 100:
                    parts = p.parts
                    project = parts[parts.index("projects") + 1] if "projects" in parts else ""
                    out.append(dict(
                        source_platform="claude", session_id=sid or p.stem,
                        message_id=r.get("uuid") or p.stem, source_file=str(p.resolve()),
                        project=project, text=t.strip()))
                    break
        if out and out[-1]["source_platform"] == "claude":
            break

    # pi
    for p in newest(".pi/agent/sessions/**/*.jsonl"):
        sid = None
        project = ""
        for line in p.open(errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") == "session":
                sid = sid or r.get("id")
                project = r.get("cwd") or r.get("workingDirectory") or project
            if r.get("type") == "message" and (r.get("message") or {}).get("role") == "assistant":
                t = text_from((r.get("message") or {}).get("content"))
                if len(t.strip()) >= 100:
                    out.append(dict(
                        source_platform="pi", session_id=sid or p.stem,
                        message_id=r.get("id") or p.stem, source_file=str(p.resolve()),
                        project=Path(project).name if project else "", text=t.strip()))
                    break
        if out and out[-1]["source_platform"] == "pi":
            break

    # Codex
    for p in newest(".codex/sessions/**/*.jsonl"):
        sid = p.stem
        project = ""
        for line in p.open(errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            payload = r.get("payload") if isinstance(r.get("payload"), dict) else r
            if r.get("type") == "session_meta":
                sid = payload.get("id", sid)
                project = Path(payload.get("cwd", "")).name
            if r.get("type") == "response_item" and payload.get("type") == "message" and payload.get("role") == "assistant":
                t = text_from(payload.get("content"))
                if len(t.strip()) >= 100:
                    out.append(dict(
                        source_platform="codex", session_id=sid,
                        message_id=payload.get("id") or f"{sid}-probe",
                        source_file=str(p.resolve()), project=project, text=t.strip()))
                    break
        if out and out[-1]["source_platform"] == "codex":
            break

    # Cursor
    for p in newest(".cursor/projects/**/agent-transcripts/**/*.jsonl"):
        for n, line in enumerate(p.open(errors="ignore"), 1):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("role") == "assistant":
                m = r.get("message") if isinstance(r.get("message"), dict) else {}
                t = text_from(m.get("content"))
                if len(t.strip()) >= 100:
                    parts = p.parts
                    idx = parts.index("projects")
                    project = parts[idx + 1]
                    out.append(dict(
                        source_platform="cursor", session_id=p.parent.name,
                        message_id=r.get("id") or m.get("id") or f"{p.parent.name}-L{n}",
                        source_file=str(p.resolve()), project=project, text=t.strip()))
                    break
        if out and out[-1]["source_platform"] == "cursor":
            break

    PROBES_PATH.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_extraction(candidate: dict, probes: list[dict]) -> list[dict]:
    """Run triple extraction for one candidate across all probes."""
    reset_extraction_model()
    model = get_provider(
        provider_name=candidate["provider"],
        model_name=candidate["model"],
    )
    results = []
    for probe in probes:
        metadata = {
            k: probe[k] for k in ("source_platform", "session_id", "message_id", "source_file", "project")
        }
        metadata["benchmark_candidate"] = candidate["id"]
        t0 = time.time()
        try:
            triples = extract_triples_gemini(model, probe["text"], trace_metadata=metadata)
            elapsed = time.time() - t0
            results.append({
                "platform": probe["source_platform"],
                "triple_count": len(triples),
                "sample": triples[:3],
                "elapsed_s": round(elapsed, 2),
                "error": None,
            })
            print(f"    [{probe['source_platform']}] {len(triples)} triples ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "platform": probe["source_platform"],
                "triple_count": 0,
                "sample": [],
                "elapsed_s": round(elapsed, 2),
                "error": str(e)[:200],
            })
            print(f"    [{probe['source_platform']}] ERROR: {str(e)[:120]}")
    reset_extraction_model()
    return results


def run_linking(candidate: dict, probes: list[dict]) -> list[dict]:
    """Run Wikidata linking for one candidate.

    Uses the first extracted entity from a baseline extraction as the link
    target, so all candidates link the same entities (apples-to-apples).
    """
    # Reset linker singleton and point it at this candidate via env
    linker._shared_model = None
    os.environ["LLM_PROVIDER"] = candidate["provider"]
    os.environ["LLM_MODEL"] = candidate["model"]
    os.environ.pop("DEVKG_LINKER_MODEL", None)

    results = []
    for probe in probes:
        metadata = {
            k: probe[k] for k in ("source_platform", "session_id", "source_file", "project")
        }
        metadata["benchmark_candidate"] = candidate["id"]
        # Use a short context slice for linking
        context = probe["text"][:600]
        # Pick a representative entity from the text (first noun-ish word)
        entity = probe["text"].split()[0].strip(",.;:()[]{}\"'").lower()
        if not entity or len(entity) < 3:
            entity = "python"
        t0 = time.time()
        try:
            match, elapsed = linker.link_entity(entity, context, trace_metadata=metadata)
            results.append({
                "platform": probe["source_platform"],
                "entity": entity,
                "qid": match.qid if match else None,
                "confidence": match.confidence if match else 0.0,
                "elapsed_s": round(elapsed, 2),
                "error": None,
            })
            qid = match.qid if match else "none"
            print(f"    [{probe['source_platform']}] {entity!r} -> {qid} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "platform": probe["source_platform"],
                "entity": entity,
                "qid": None,
                "confidence": 0.0,
                "elapsed_s": round(elapsed, 2),
                "error": str(e)[:200],
            })
            print(f"    [{probe['source_platform']}] ERROR: {str(e)[:120]}")

    linker._shared_model = None
    return results


def save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark cheaper LLM candidates for DevKG")
    parser.add_argument("--only", help="Comma-separated candidate IDs to run (default: all)")
    parser.add_argument("--skip-linking", action="store_true", help="Skip Wikidata linking tests")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction tests")
    args = parser.parse_args()

    probes = select_probes()
    print(f"Loaded {len(probes)} probes: {[p['source_platform'] for p in probes]}")
    if not probes:
        print("ERROR: no probes found", file=sys.stderr)
        sys.exit(1)

    candidates = CANDIDATES
    if args.only:
        ids = [x.strip() for x in args.only.split(",")]
        candidates = [c for c in CANDIDATES if c["id"] in ids]
        if not candidates:
            print(f"ERROR: no candidates matched --only {args.only}", file=sys.stderr)
            sys.exit(1)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baselines": BASELINES,
        "probes": [{"platform": p["source_platform"], "chars": len(p["text"])} for p in probes],
        "candidates": [],
    }

    for c in candidates:
        print(f"\n{'='*70}")
        print(f"CANDIDATE: {c['id']}  ({c['provider']}/{c['model']})")
        print(f"  price: ${c['price_in']}/${c['price_out']} per M  | note: {c['note']}")
        print(f"  workloads: {c['workloads']}")
        print(f"{'='*70}")

        entry = {
            "id": c["id"],
            "provider": c["provider"],
            "model": c["model"],
            "price_in": c["price_in"],
            "price_out": c["price_out"],
            "note": c["note"],
            "extraction": None,
            "linking": None,
        }

        if not args.skip_extraction and "extraction" in c["workloads"]:
            print("  EXTRACTION:")
            try:
                entry["extraction"] = run_extraction(c, probes)
            except Exception as e:
                entry["extraction"] = [{"error": f"construction failed: {str(e)[:200]}"}]
                print(f"  EXTRACTION FAILED: {str(e)[:120]}")
                traceback.print_exc(file=sys.stderr)

        if not args.skip_linking and "linking" in c["workloads"]:
            print("  LINKING:")
            try:
                entry["linking"] = run_linking(c, probes)
            except Exception as e:
                entry["linking"] = [{"error": f"construction failed: {str(e)[:200]}"}]
                print(f"  LINKING FAILED: {str(e)[:120]}")
                traceback.print_exc(file=sys.stderr)

        results["candidates"].append(entry)
        save_results(results)  # incremental save

    # Flush tracers so LangSmith receives everything
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"DONE. Results saved to {RESULTS_PATH}")
    print(f"{'='*70}")
    print_summary(results)


def print_summary(results: dict) -> None:
    print(f"\nSUMMARY (baseline extraction={BASELINES['extraction']['id']}, "
          f"linking={BASELINES['linking']['id']})\n")
    print(f"{'Candidate':<26} {'$/M in/out':<16} {'Ext triples':<14} {'Link QIDs':<14} {'Note'}")
    print("-" * 100)
    for c in results["candidates"]:
        price = f"${c['price_in']}/${c['price_out']}"
        ext_str = "-"
        if c["extraction"]:
            counts = [r.get("triple_count", 0) for r in c["extraction"] if not r.get("error")]
            if counts:
                ext_str = f"{sum(counts)}/{len(counts)} msgs"
            elif any(r.get("error") for r in c["extraction"]):
                ext_str = "ERROR"
        link_str = "-"
        if c["linking"]:
            qids = [r.get("qid") for r in c["linking"] if r.get("qid")]
            total = len([r for r in c["linking"] if not r.get("error")])
            if total:
                link_str = f"{len(qids)}/{total} hit"
            elif any(r.get("error") for r in c["linking"]):
                link_str = "ERROR"
        print(f"{c['id']:<26} {price:<16} {ext_str:<14} {link_str:<14} {c['note']}")


if __name__ == "__main__":
    main()
