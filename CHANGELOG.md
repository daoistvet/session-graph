# Changelog

All notable changes to session-graph are documented here.

## [0.8.0] - 2026-04-03

### Added
- **`pi` Coding Agent Support** (`pi_to_rdf.py`) — ingests `pi`'s native `.jsonl` session format. Extracts assistant messages and custom tool calls (`toolCall` / `toolResult`) mapping them to the same RDF ontology alongside Claude Code.
- **`pi` Native Extension Hook** — added `devkg-hook.ts` as a native `pi` extension that listens to `agent_end` and `session_shutdown` events, safely POSTing to RabbitMQ just like `stop_hook.sh`.
- **Dual Session Processing** — updated background consumer, `bulk_process`, and `bulk_batch` to seamlessly process `.pi/agent/sessions` and `.claude/projects/` paths with isolated parsing logic but unified graph insertion.
- **Docker Mounts** — `pipeline-runner` container now automatically mounts `~/.pi/agent/sessions` as a read-only volume (`/pi-sessions`).

## [0.7.0] - 2026-03-04

### Added
- **ChatGPT parser** (`chatgpt_to_rdf.py`) — ingests OpenAI conversation export JSON, producing same RDF ontology as all other parsers. 639 conversations → 29,249 knowledge triples.
- **Multi-platform Vertex AI batch ingestion** — extended batch pipeline to Grok (416 msgs), DeepSeek (132 msgs), and ChatGPT (3,412 msgs) using `gemini-3-flash-preview` on `global` endpoint.
- **Gemini 3 Flash Preview** — upgraded both triple extraction (batch) and agentic entity linker from Gemini 2.5 Flash to `gemini-3-flash-preview`.

### Changed
- **`agentic_linker_langgraph.py`** — Vertex AI is now the first detection priority (before `GEMINI_API_KEY`). Uses `vertexai=True, project=project, location="global"` for Gemini 3.x compatibility.
- **Knowledge graph scale** — 1,537 sessions / 97,910 knowledge triples / 2M+ RDF triples across 4 platforms (claude-code 734, chatgpt 639, grok 122, deepseek 42).

### Fixed
- **Batch metadata format** — Vertex AI requires metadata as a JSON string, not a raw dict. ChatGPT batch initially failed with `"unsupported type"` error; fixed by wrapping metadata with `json.dumps()`.

## [0.6.0] - 2026-02-21

### Added
- **Interactive `setup.sh`** — single-command setup from clone to working pipeline. Checks prerequisites (Python 3.11+, Docker, jq), creates `.env` with interactive provider selection, installs provider-specific Python dependencies, creates output directories, starts Docker Compose, optionally installs Claude Code stop hook, and runs a smoke test. Idempotent — re-running skips completed steps.
- **`--auth` CLI flag on `load_fuseki.py`** — `--auth admin:admin` for Docker Fuseki authentication. Fixes 401 errors newcomers hit out of the box. Also added `auth` parameter to `count_triples()`.
- **Sample session fixture** (`tests/fixtures/sample_session.jsonl`) — minimal 5-message Claude Code session (FastAPI/SQLAlchemy/Docker) that produces 48 RDF triples with `--skip-extraction` (no LLM calls needed).
- **README troubleshooting table** — covers Fuseki 401, RabbitMQ unreachable, no sessions, output buffering, stop hook, ModuleNotFoundError.

### Changed
- **`hooks/stop_hook.sh`** — replaced hardcoded `/Users/robertoshimizu/...` paths with dynamic `$(dirname "$0")/..` resolution. Now works from any clone location.
- **README Quick Start** — `./setup.sh` is now the primary path. Manual 8-step instructions moved to collapsible `<details>` section.
- **`.gitignore`** — `hooks/` directory is now tracked (was fully ignored). Only `hooks/*.log` is excluded.
- All `load_fuseki` examples in README updated to include `--auth admin:admin`.

## [0.5.0] - 2026-02-21

### Added
- **Triple extraction cache** (`.triple_cache.db`) — SQLite cache keyed by message UUID prevents redundant Gemini API calls when stop hook re-processes the same session. Re-runs rebuild full RDF graph but skip all cached messages (0 API calls).
- **Full pipeline automation verified** — stop hook → RabbitMQ → pipeline-runner → Fuseki is now the production path. Docker Fuseki promoted to primary on port 3030, local Java Fuseki retired.
- Cleaned up 101 empty session files from `~/.claude/projects/`.
- Fixed stale `_init_vertex_credentials` import in `link_entities.py`.

### Changed
- `docker/queue_consumer.py` — removed mtime-based watermark check (cache makes it unnecessary).
- `docker-compose.yml` — added volume mount for `.triple_cache.db` persistence between container and host.
- Docker Fuseki is now the primary triplestore (was local Java standalone).

## [0.4.0] - 2026-02-21

### Added
- **RabbitMQ-based pipeline automation** — stop hook now publishes to RabbitMQ (33ms) instead of running Python in a background subshell (which got killed by Claude Code). A long-running `pipeline-runner` container consumes the queue independently.
- **Docker Compose stack** — `rabbitmq` (management UI on :15672), `pipeline-runner` (pika consumer), and existing `fuseki` in a single `docker compose up -d`. Dead-letter queue (`devkg_jobs_failed`) for failed jobs.
- **Vertex AI credentials in container** — `queue_consumer.py` decodes `GOOGLE_APPLICATION_CREDENTIALS_BASE64` from `.env` at startup, enabling Gemini via Vertex AI inside Docker.
- **Fuseki auth support** — `load_fuseki.py` functions now accept optional `auth` tuple; Docker Fuseki uses admin:admin.
- **Integration test** (`tests/test_integration.sh`) — 16-point end-to-end test covering services, queue, consumer processing, .ttl output, and Fuseki upload.

### Changed
- `hooks/stop_hook.sh` rewritten: `curl` POST to RabbitMQ HTTP API replaces background Python process.
- `pipeline/load_fuseki.py`: `ensure_dataset()` and `upload_turtle()` accept optional `auth` parameter (backward-compatible).

## [0.3.0] - 2026-02-21

### Added
- **Two-level entity filtering** — Level 1 (`is_valid_entity()`) rejects garbage at extraction time with 13 filter groups: filenames, hex colors, CLI flags, ICD codes, snake_case identifiers, DOM selectors, version strings, CSS dimensions, issue refs, function calls, npm scopes, percentage values. Level 2 (`is_linkable_entity()`) pre-filters before Wikidata API calls.
- **Top-10 extraction cap** — prompt extracts at most 10 triples per message, prioritizing architectural decisions and technology choices. Hard cap enforced in parsing.
- **Frequency-based entity linking** — `--min-sessions N` flag (default: 2) only links entities appearing in N+ sessions. ~77% of entities are single-session noise, reducing linking cost from ~37K to ~8.6K entities.
- **48 whitelisted short terms** (`ai`, `api`, `llm`, `rdf`, `sql`, etc.) bypass all entity filters.

### Changed
- Entity linking pipeline now runs pre-filter before any Wikidata API calls.
- Extraction prompt updated to enforce top-10 cap.

## [0.2.0] - 2026-02-20

### Added
- **Context-aware entity linking** — neighboring KnowledgeTriple relationships extracted from `.ttl` files and passed as disambiguation context to the ReAct agent. Resolves ambiguous labels (e.g., "condition" → disease instead of programming conditional).
- **`snapshot_links.py`** — inspect intermediate entity linking progress without interrupting a running job.
- **`devkg-sparql` skill** — 14 local + 6 Wikidata SPARQL query templates for Claude Code.

### Changed
- Entity linking agent receives triple context for better disambiguation.

## [0.1.0] - 2026-02-16

### Added
- **Batch processing via Vertex AI Batch Prediction** (`bulk_batch.py`) — submit/status/collect workflow with 50% cost discount.
- **Sequential bulk processor** (`bulk_process.py`) — watermarks, subagent filtering, `--dry-run`, `--limit`, `--force`.
- **Agentic entity linking** — LangGraph ReAct agent (Gemini 2.5 Flash + Wikidata API tool) replaces heuristic. 7/7 precision.
- **161 entity alias mappings** (`entity_aliases.json`) — k8s→kubernetes, otel→OpenTelemetry, etc.
- **Multi-platform parsers** — Claude Code (JSONL), DeepSeek (JSON zip), Grok (JSON), Warp (SQLite).
- **OWL ontology** composing PROV-O + SIOC + SKOS + Dublin Core + Schema.org with 24 curated predicates.
- **Dual storage** — direct edges for traversal + reified KnowledgeTriple nodes for provenance.
- **Apache Jena Fuseki** integration for SPARQL queries.

## [0.0.1] - 2026-02-13

### Added
- Initial project: JSONL→RDF pipeline, OWL ontology, Fuseki loader.
