"""
CAG-style triple extraction from Claude Code assistant responses.

Extracts decisions, failed attempts, observations, and next steps
using a focused prompt separate from the devkg extraction pipeline.
Output is posted to the compounding-agents server as KnowledgeTriples.

Usage:
    from pipeline.cag_extraction import extract_cag_triples

    model = get_provider("gemini", "gemini-2.5-flash")
    triples = extract_cag_triples(model, assistant_text)
    # [{"type": "decision", "subject": "auth", "predicate": "decided",
    #   "object": "JWT", "selectedOption": "JWT", "rejectedOptions": ["session cookies"]}]
"""

import json
import sys

# CAG predicate vocabulary
CAG_PREDICATES = {
    "decided": "Chose X over alternatives",
    "rejectedBecause": "Why an alternative was rejected",
    "failedWith": "Something was tried and failed",
    "observed": "A learning or insight discovered",
    "blockedBy": "Progress blocked by issue",
    "nextStep": "What remains to be done",
}

_CAG_PREDICATE_SET = set(CAG_PREDICATES.keys())


def build_cag_prompt(text: str, prior_triples: list[dict] | None = None) -> str:
    """Build the CAG extraction prompt for a given text.

    Args:
        text: The assistant message text to analyze.
        prior_triples: Previously extracted triples for this session (if any).
            When provided, the LLM returns the complete updated set.
    """

    predicate_docs = "\n".join(
        f'  - "{k}": {v}' for k, v in CAG_PREDICATES.items()
    )

    # Truncate very long texts to stay within model limits
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    prior_section = ""
    if prior_triples:
        prior_json = json.dumps(prior_triples, indent=2)
        prior_section = f"""
## Previously extracted triples for this session:
```json
{prior_json}
```
These triples were already extracted from earlier messages in this session.
Return the **complete updated set** — keep unchanged ones, add new ones from the new text below,
and remove any that are no longer accurate given the new context.
"""

    return f"""You are a knowledge extraction assistant. Extract engineering decisions,
failed attempts, observations, and next steps from the following Claude Code assistant response.

## Allowed predicates:
{predicate_docs}

## Output format:
Return a JSON object with a "triples" array. Each triple has:
- "type": one of "decision", "observation", "failure"
- "subject": the thing being decided/observed/attempted (short identifier, e.g., "auth-middleware")
- "predicate": one of the allowed predicates above
- "object": the target/result (short identifier)
- "context": brief explanation (1 sentence)
- For type "decision": also include "selectedOption" (string) and "rejectedOptions" (array of strings) if alternatives were discussed

## Rules:
- Only extract concrete, actionable knowledge — not generic descriptions
- Subject and object should be short, specific technical identifiers (2-4 words, kebab-case preferred, e.g., "cag-extraction", "redis-streams", "hooks-ts")
- Do NOT use generic subjects like "current-task", "updated-tests", "implementation", "code-changes". Use the actual thing: "cag-cache-module", "queue-consumer-parallelization"
- Do NOT extract tautological observations like "tests observed pass" or "task observed completed" — these carry no useful information for a future agent
- Skip greetings, acknowledgements, and meta-commentary
- If no relevant knowledge found AND no prior triples were provided, return {{"triples": []}}
- **CRITICAL**: When prior triples are provided, you MUST return them unchanged if the new text has no relevant knowledge to add or modify. NEVER return an empty array when prior triples exist. Prior triples should only be removed if the new text explicitly contradicts or supersedes them.
- Merge duplicate subjects: if a prior triple already covers the same subject+predicate, update it rather than adding a new one
{prior_section}
## Text to analyze:
{text}

## Response (JSON only):"""


def extract_cag_triples(model, text: str, prior_triples: list[dict] | None = None) -> list[dict]:
    """Extract CAG-style triples from text using an LLM.

    Args:
        model: An LLM provider with generate_content() method
        text: The assistant response text to analyze
        prior_triples: Previously extracted triples for this session (if any).
            When provided, the LLM returns the complete updated set.

    Returns:
        List of triple dicts with type, subject, predicate, object, context fields.
    """
    if not text or len(text.strip()) < 50:
        return []

    prompt = build_cag_prompt(text, prior_triples=prior_triples)

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        print(f"  [cag] LLM call failed: {e}", file=sys.stderr)
        return []

    # Parse JSON response
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            triples = data
        else:
            triples = data.get("triples", [])
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", raw)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    triples = data
                else:
                    triples = data.get("triples", [])
            except json.JSONDecodeError:
                print(f"  [cag] Failed to parse LLM response as JSON", file=sys.stderr)
                return []
        else:
            print(f"  [cag] Failed to parse LLM response as JSON", file=sys.stderr)
            return []

    # Validate and normalize triples
    valid = []
    for t in triples:
        if not isinstance(t, dict):
            continue

        triple_type = t.get("type", "observation")
        if triple_type not in ("decision", "observation", "failure"):
            triple_type = "observation"

        predicate = t.get("predicate", "observed")
        if predicate not in _CAG_PREDICATE_SET:
            predicate = "observed"

        subject = str(t.get("subject", "")).strip().strip('`"\' ')
        obj = str(t.get("object", "")).strip().strip('`"\' ')
        if not subject or not obj:
            continue

        result = {
            "type": triple_type,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "context": str(t.get("context", "")).strip() or None,
        }

        # Decision-specific fields
        if triple_type == "decision":
            selected = t.get("selectedOption")
            if selected:
                result["selectedOption"] = {"title": str(selected)}
            rejected = t.get("rejectedOptions", [])
            if rejected and isinstance(rejected, list):
                result["rejectedOptions"] = [
                    {"title": str(r)} for r in rejected if r
                ]

        valid.append(result)

    # Guard: if prior triples existed but LLM returned empty, preserve prior set
    if not valid and prior_triples:
        print("  [cag] LLM returned empty with prior triples — preserving prior set", file=sys.stderr)
        return prior_triples

    return valid
