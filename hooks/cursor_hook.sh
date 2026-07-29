#!/bin/bash
# DevKG Cursor Stop Hook — publishes pipeline job to RabbitMQ after each Cursor agent turn.
#
# Wired via ~/.cursor/hooks.json (event: stop). Same RabbitMQ contract as hooks/stop_hook.sh.
# Consumer: translate_path → cursor_to_rdf → Wikidata link → Fuseki.

set -euo pipefail

DEVKG_RABBITMQ_HOST="${DEVKG_RABBITMQ_HOST:-localhost}"
DEVKG_RABBITMQ_PORT="${DEVKG_RABBITMQ_PORT:-15672}"
DEVKG_RABBITMQ_USER="${DEVKG_RABBITMQ_USER:-devkg}"
DEVKG_RABBITMQ_PASS="${DEVKG_RABBITMQ_PASS:-devkg}"
DEVKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVKG_OUTPUT_DIR="${DEVKG_OUTPUT_DIR:-${DEVKG_ROOT}/output/cursor}"
DEVKG_LOG_DIR="${DEVKG_LOG_DIR:-${DEVKG_ROOT}/logs}"
CURSOR_PROJECTS_DIR="${CURSOR_PROJECTS_DIR:-${HOME}/.cursor/projects}"

mkdir -p "$DEVKG_OUTPUT_DIR" "$DEVKG_LOG_DIR"

LOG_FILE="$DEVKG_LOG_DIR/cursor_hook.log"

INPUT=$(cat)

TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.conversation_id // .session_id // empty')

# If transcript_path is missing, resolve from conversation_id under ~/.cursor/projects
if [ -z "$TRANSCRIPT_PATH" ] && [ -n "$SESSION_ID" ] && [ -d "$CURSOR_PROJECTS_DIR" ]; then
    CANDIDATE=$(find "$CURSOR_PROJECTS_DIR" -path "*/agent-transcripts/${SESSION_ID}/${SESSION_ID}.jsonl" -type f 2>/dev/null | head -1 || true)
    if [ -n "$CANDIDATE" ]; then
        TRANSCRIPT_PATH="$CANDIDATE"
    fi
fi

# Fall back: session_id from path basename
if [ -z "$SESSION_ID" ] && [ -n "$TRANSCRIPT_PATH" ]; then
    SESSION_ID="$(basename "$TRANSCRIPT_PATH" .jsonl)"
fi

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    exit 0
fi

# Only Cursor agent transcripts
if ! echo "$TRANSCRIPT_PATH" | grep -q "/.cursor/projects/"; then
    exit 0
fi

if echo "$TRANSCRIPT_PATH" | grep -q "/subagents/"; then
    exit 0
fi

BASENAME="${SESSION_ID:-$(basename "$TRANSCRIPT_PATH" .jsonl)}"
OUTPUT_FILE="$DEVKG_OUTPUT_DIR/${BASENAME}.ttl"

if [ -f "$OUTPUT_FILE" ] && [ "$OUTPUT_FILE" -nt "$TRANSCRIPT_PATH" ]; then
    exit 0
fi

PAYLOAD=$(jq -n \
    --arg tp "$TRANSCRIPT_PATH" \
    --arg sid "$SESSION_ID" \
    '{transcript_path: $tp, session_id: $sid}')

RABBIT_BODY=$(jq -n \
    --arg payload "$PAYLOAD" \
    '{
        properties: {delivery_mode: 2},
        routing_key: "devkg_jobs",
        payload: $payload,
        payload_encoding: "string"
    }')

RABBIT_URL="http://${DEVKG_RABBITMQ_HOST}:${DEVKG_RABBITMQ_PORT}/api/exchanges/%2f/amq.default/publish"

if curl -s -f -u "${DEVKG_RABBITMQ_USER}:${DEVKG_RABBITMQ_PASS}" \
    -H "Content-Type: application/json" \
    -d "$RABBIT_BODY" \
    "$RABBIT_URL" > /dev/null 2>&1; then
    echo "[$(date)] Queued: $TRANSCRIPT_PATH (session: $SESSION_ID)" >> "$LOG_FILE"
else
    echo "[$(date)] ERROR: Failed to publish to RabbitMQ at $RABBIT_URL" >> "$LOG_FILE"
    echo "[$(date)]   Transcript: $TRANSCRIPT_PATH" >> "$LOG_FILE"
fi

exit 0
