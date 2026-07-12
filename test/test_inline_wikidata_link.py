#!/usr/bin/env python3
"""E2E test: cache hits + true cache misses through ReAct Wikidata linker."""

import os
import sys

# Match consumer credential init (PYTHONPATH must include /app)
from docker import queue_consumer as _qc  # noqa: F401 — triggers _init_vertex_credentials()

from rdflib import Literal
from rdflib.namespace import RDF, RDFS, OWL

from pipeline.common import DEVKG, entity_uri, create_graph
from pipeline.link_entities import link_entities_into_graph, init_cache, cache_get


def main() -> int:
    print("GAC set:", bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")))
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    print("GAC path exists:", bool(gac and os.path.exists(gac)))

    hits = ["fastapi", "neo4j"]
    misses = ["apache kafka", "prometheus", "golang"]

    # Clear polluted negative-cache from prior failed test
    conn = init_cache()
    for label in misses:
        conn.execute("DELETE FROM wikidata_cache WHERE label = ?", (label,))
    conn.commit()
    conn.close()
    print("Cleared prior negative-cache for test misses")

    g = create_graph()
    for label in hits + misses:
        uri = entity_uri(label)
        g.add((uri, RDF.type, DEVKG.Entity))
        g.add((uri, RDFS.label, Literal(label)))

    print("\n=== BEFORE ===")
    conn = init_cache()
    for label in hits + misses:
        print(f"  cache[{label}] = {cache_get(conn, label)}")
    assert all(cache_get(conn, l) is not None for l in hits)
    assert all(cache_get(conn, l) is None for l in misses)
    conn.close()
    print("Confirmed: 2 hits, 3 true misses")

    print("\n=== RUNNING link_entities_into_graph (max_agentic_calls=3) ===")
    stats = link_entities_into_graph(g, max_agentic_calls=3, max_workers=3, agentic=True)
    print("stats:", stats)

    print("\n=== AFTER ===")
    conn = init_cache()
    for label in hits + misses:
        c = cache_get(conn, label)
        sameas = [str(x) for x in g.objects(entity_uri(label), OWL.sameAs)]
        qid = c.get("qid") if c else None
        conf = c.get("confidence") if c else None
        print(f"  {label}: cache_qid={qid} conf={conf} sameAs={sameas}")
    conn.close()

    assert stats["cache_hits"] == 2, stats
    assert stats["agentic_calls"] == 3, stats
    assert stats["linked"] >= 2, stats

    conn = init_cache()
    for label in misses:
        assert cache_get(conn, label) is not None, f"{label} not written to cache"
    conn.close()

    linked_misses = sum(
        1 for label in misses if any(g.triples((entity_uri(label), OWL.sameAs, None)))
    )
    print(f"\nMisses that got owl:sameAs: {linked_misses}/3")
    assert linked_misses >= 1, "expected at least 1 agentic success on well-known entities"

    print("\nPASS: full cache + agentic ReAct path validated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
