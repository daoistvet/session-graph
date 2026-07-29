from types import SimpleNamespace

import pytest

from pipeline import llm_providers
from pipeline.agentic_linker_langgraph import WikidataMatch, link_entity
from pipeline.triple_extraction import extract_triples_gemini


_PROVIDER_ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FIREWORKS_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    for key in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_resolve_provider_honors_explicit_alias():
    assert llm_providers.resolve_provider_name("claude") == "anthropic"
    assert llm_providers.resolve_provider_name("gemini-vertex") == "gemini"


def test_resolve_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown provider"):
        llm_providers.resolve_provider_name("not-a-provider")


def test_fireworks_requires_explicit_model():
    with pytest.raises(ValueError, match="requires an explicit model"):
        llm_providers.get_default_model("fireworks")


def test_get_provider_delegates_vertex_configuration(monkeypatch):
    calls = []
    sentinel = object()

    def fake_init_chat_model(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")

    result = llm_providers.get_provider("gemini", "gemini-2.5-flash")

    assert result is sentinel
    assert calls == [
        {
            "model": "gemini-2.5-flash",
            "model_provider": "google_genai",
            "temperature": 0.2,
            "max_tokens": 8192,
            "max_retries": 6,
            "vertexai": True,
            "project": "test-project",
            "location": "us-east5",
        }
    ]


def test_triple_extraction_uses_invoke_and_langsmith_metadata():
    class RetryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt, config):
            self.calls.append((prompt, config))
            if len(self.calls) == 1:
                raise RuntimeError("temporary failure")
            return SimpleNamespace(
                text='[{"subject":"neo4j","predicate":"storesIn","object":"graph database"}]'
            )

    model = RetryModel()
    triples = extract_triples_gemini(
        model,
        "Neo4j stores application knowledge in a graph database for retrieval.",
        trace_metadata={
            "source_platform": "cursor",
            "session_id": "session-123",
            "message_id": "message-456",
        },
    )

    assert triples == [
        {"subject": "neo4j", "predicate": "storesIn", "object": "graph database"}
    ]
    assert len(model.calls) == 2
    assert model.calls[1][1]["run_name"] == "devkg.triple_extraction"
    assert model.calls[1][1]["tags"] == [
        "devkg",
        "triple-extraction",
        "platform:cursor",
    ]
    assert model.calls[1][1]["metadata"] == {
        "source_platform": "cursor",
        "session_id": "session-123",
        "message_id": "message-456",
        "attempt": 2,
        "input_chars": 69,
    }


def test_wikidata_linking_propagates_provenance(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.calls = []

        def invoke(self, payload, config):
            self.calls.append((payload, config))
            return {
                "structured_response": WikidataMatch(
                    qid="Q101119404",
                    confidence=0.99,
                    label="FastAPI",
                    description="Python web framework",
                    reasoning="Exact result",
                )
            }

    agent = FakeAgent()
    monkeypatch.setattr(
        "pipeline.agentic_linker_langgraph.create_linker_agent", lambda: agent
    )

    result, _ = link_entity(
        "fastapi",
        "FastAPI is a Python web framework",
        trace_metadata={
            "source_platform": "claude",
            "session_id": "session-789",
            "message_id": "message-012",
        },
    )

    assert result.qid == "Q101119404"
    assert agent.calls[0][1] == {
        "run_name": "devkg.wikidata_linking",
        "tags": ["devkg", "wikidata-linking", "platform:claude"],
        "metadata": {
            "source_platform": "claude",
            "session_id": "session-789",
            "message_id": "message-012",
            "entity": "fastapi",
        },
    }


def test_link_entities_into_graph_propagates_provenance(monkeypatch):
    from rdflib import Graph, Literal, RDF, RDFS, URIRef
    from pipeline import link_entities as le
    from pipeline.common import DEVKG, entity_uri

    captured = {}

    def fake_agentic_link_one(label, context="developer knowledge graph entity", trace_metadata=None):
        captured["label"] = label
        captured["trace_metadata"] = trace_metadata
        return (label, "Q101119404", 0.99, "FastAPI", "exact", 0.1)

    monkeypatch.setattr(le, "_agentic_link_one", fake_agentic_link_one)
    monkeypatch.setattr(le, "_ensure_agentic_init", lambda: None)
    monkeypatch.setattr(le, "is_linkable_entity", lambda label: True)
    monkeypatch.setattr(le, "normalize_label", lambda raw, aliases: raw)
    monkeypatch.setattr(le, "load_aliases", lambda: {})
    monkeypatch.setattr(le, "init_cache", lambda: __import__("types").SimpleNamespace(
        close=lambda: None,
    ))
    monkeypatch.setattr(le, "cache_get", lambda conn, label: None)
    monkeypatch.setattr(le, "cache_put", lambda *a, **k: None)

    g = Graph()
    node = entity_uri("fastapi")
    g.add((node, RDF.type, DEVKG.Entity))
    g.add((node, RDFS.label, Literal("fastapi")))

    stats = le.link_entities_into_graph(
        g,
        trace_metadata={
            "source_platform": "cursor",
            "session_id": "sess-abc",
            "source_file": "/tmp/x.jsonl",
            "project": "demo",
        },
    )

    assert stats["agentic_calls"] == 1
    assert captured["trace_metadata"] == {
        "source_platform": "cursor",
        "session_id": "sess-abc",
        "source_file": "/tmp/x.jsonl",
        "project": "demo",
    }


def test_linker_model_honors_llm_model(monkeypatch):
    import pipeline.agentic_linker_langgraph as linker

    monkeypatch.setenv("LLM_PROVIDER", "fireworks")
    monkeypatch.setenv("LLM_MODEL", "accounts/fireworks/models/glm-5p2")
    monkeypatch.delenv("DEVKG_LINKER_MODEL", raising=False)

    captured = {}

    def fake_get_provider(provider_name=None, model_name=None, **kwargs):
        captured["provider"] = provider_name
        captured["model"] = model_name
        return object()

    monkeypatch.setattr(linker, "get_provider", fake_get_provider)
    linker._shared_model = None
    linker._get_shared_model()
    assert captured["provider"] == "fireworks"
    assert captured["model"] == "accounts/fireworks/models/glm-5p2"
    linker._shared_model = None


def test_extraction_model_singleton_caches(monkeypatch):
    import pipeline.llm_providers as lp

    calls = []

    def fake_get_provider(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(lp, "get_provider", fake_get_provider)
    lp.reset_extraction_model()

    first = lp.get_extraction_model(provider_name="gemini", model_name="gemini-2.5-flash")
    second = lp.get_extraction_model(provider_name="gemini", model_name="gemini-2.5-flash")
    assert first is second
    assert len(calls) == 1

    # Different args reconstruct the model
    third = lp.get_extraction_model(provider_name="openai", model_name="gpt-4o-mini")
    assert third is not first
    assert len(calls) == 2

    lp.reset_extraction_model()
    assert lp._extraction_model is None
