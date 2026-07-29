"""LangChain-backed chat model factory for DevKG.

``get_provider`` remains the project-level adapter for provider selection and
configuration policy. The returned object is a native LangChain
``BaseChatModel`` supporting ``invoke()``, ``stream()``, ``batch()``, async
methods, and automatic LangSmith tracing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_PROVIDER_IDS = {
    "gemini": "google_genai",
    "openai": "openai",
    "anthropic": "anthropic",
    "fireworks": "fireworks",
    "mistral": "mistralai",
    "ollama": "ollama",
}

_PROVIDER_ALIASES = {
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "gemini-vertex": "gemini",
    "vertex": "gemini",
    "claude": "anthropic",
    "mistralai": "mistral",
}

_DEFAULT_MODELS: dict[str, str | None] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-latest",
    # Fireworks model availability changes independently of LangChain. Require
    # an explicit model rather than silently selecting a stale serverless ID.
    "fireworks": None,
    "mistral": "mistral-small-latest",
    "ollama": "llama3.1",
}


def _normalize_provider(provider_name: str) -> str:
    normalized = provider_name.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def resolve_provider_name(provider_name: str | None = None) -> str:
    """Resolve an explicit, configured, or credential-detected provider name."""
    selected = provider_name or os.environ.get("LLM_PROVIDER")
    if selected:
        normalized = _normalize_provider(selected)
        if normalized not in _PROVIDER_IDS:
            raise ValueError(
                f"Unknown provider '{selected}'. Supported: {', '.join(list_providers())}"
            )
        return normalized

    if (
        os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    ):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("FIREWORKS_API_KEY"):
        return "fireworks"
    if os.environ.get("MISTRAL_API_KEY"):
        return "mistral"

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import requests

        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama"
    except Exception:
        pass

    raise RuntimeError(
        "No LLM provider detected. Set LLM_PROVIDER and its credentials, or "
        "configure one of GOOGLE_CLOUD_PROJECT, GEMINI_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, FIREWORKS_API_KEY, or a reachable Ollama server."
    )


def get_default_model(provider_name: str) -> str:
    """Return DevKG's default model, requiring explicit Fireworks selection."""
    provider = _normalize_provider(provider_name)
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider '{provider_name}'. Supported: {', '.join(list_providers())}"
        )
    model = _DEFAULT_MODELS[provider]
    if model is None:
        raise ValueError(
            f"Provider '{provider}' requires an explicit model via --model or LLM_MODEL."
        )
    return model


def _resolve_model_name(
    provider: str,
    requested_provider: str | None,
    model_name: str | None,
) -> str:
    if model_name:
        return model_name

    configured_model = os.environ.get("LLM_MODEL")
    configured_provider = os.environ.get("LLM_PROVIDER")
    if configured_model and (
        requested_provider is None
        or (
            configured_provider
            and _normalize_provider(configured_provider) == provider
        )
    ):
        return configured_model

    return get_default_model(provider)


def get_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    max_retries: int = 6,
    **model_kwargs: Any,
) -> BaseChatModel:
    """Return a native LangChain chat model using DevKG configuration policy.

    Provider construction is delegated to LangChain's ``init_chat_model``.
    Provider-specific packages are loaded lazily by LangChain.
    """
    from langchain.chat_models import init_chat_model

    provider = resolve_provider_name(provider_name)
    model = _resolve_model_name(provider, provider_name, model_name)

    init_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        **model_kwargs,
    }
    backend = _PROVIDER_IDS[provider]
    if provider == "gemini":
        project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )
        if project:
            location = (
                "global"
                if model.startswith("gemini-3")
                else os.environ.get("CLOUD_ML_REGION", "us-east5")
            )
            init_kwargs.setdefault("vertexai", True)
            init_kwargs.setdefault("project", project)
            init_kwargs.setdefault("location", location)
            backend = f"google_genai/vertex-ai ({project}/{location})"
    elif provider == "ollama":
        init_kwargs.setdefault(
            "base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    chat_model = init_chat_model(
        model=model,
        model_provider=_PROVIDER_IDS[provider],
        **init_kwargs,
    )
    print(f"  LangChain provider: {provider}/{model} ({backend})", file=sys.stderr)
    return chat_model


# ---------------------------------------------------------------------------
# Process-level singletons per workload
# ---------------------------------------------------------------------------
#
# Extraction and linking each keep their own cached chat model. A long-lived
# process (queue consumer, bulk processor) pays model construction once;
# env changes require a process restart. Tests reset via the clear_* helpers.

_extraction_model: BaseChatModel | None = None
_extraction_model_key: tuple | None = None


def get_extraction_model(
    provider_name: str | None = None,
    model_name: str | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    max_retries: int = 6,
    **model_kwargs: Any,
) -> BaseChatModel:
    """Return the process-level singleton chat model for triple extraction.

    First call constructs the model via ``get_provider``; subsequent calls
    return the cached instance, ignoring arguments. Use ``reset_extraction_model``
    to clear the cache (mainly for tests).
    """
    global _extraction_model, _extraction_model_key
    key = (provider_name, model_name, temperature, max_tokens, max_retries)
    if _extraction_model is not None and _extraction_model_key == key:
        return _extraction_model
    _extraction_model = get_provider(
        provider_name=provider_name,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        **model_kwargs,
    )
    _extraction_model_key = key
    return _extraction_model


def reset_extraction_model() -> None:
    """Clear the extraction model singleton (test hook)."""
    global _extraction_model, _extraction_model_key
    _extraction_model = None
    _extraction_model_key = None



def list_providers() -> list[str]:
    """Return supported DevKG provider names."""
    return list(_PROVIDER_IDS)
