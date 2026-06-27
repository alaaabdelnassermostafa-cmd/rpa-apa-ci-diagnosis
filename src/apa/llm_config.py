"""Central LLM provider configuration helper.

Set LLM_PROVIDER to one of: deepseek (default), openrouter, openai.
Set either the provider-specific env var (DEEPSEEK_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY)
or a generic LLM_API_KEY. Call `make_client()` to get an OpenAI client configured
for the selected provider.
"""
from openai import OpenAI
import os

PROVIDERS = {
    "deepseek": {"env": "DEEPSEEK_API_KEY", "base": "https://api.deepseek.com"},
    "openrouter": {"env": "OPENROUTER_API_KEY", "base": "https://openrouter.ai/api/v1"},
    "openai": {"env": "OPENAI_API_KEY", "base": None},
    "gemini": {"env": "GEMINI_API_KEY", "base": "https://generativelanguage.googleapis.com/v1beta/openai/"},
}


def get_provider() -> str:
    configured = os.environ.get("LLM_PROVIDER")
    if configured:
        return configured.lower()

    # Auto-detect provider from available keys when LLM_PROVIDER is unset.
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return "deepseek"


def get_provider_info(provider: str | None = None) -> dict:
    p = (provider or get_provider()).lower()
    return PROVIDERS.get(p, PROVIDERS["deepseek"]) 


def get_api_key(provider: str | None = None) -> str | None:
    info = get_provider_info(provider)
    return os.environ.get(info["env"]) or os.environ.get("LLM_API_KEY")


def make_client(client: object | None = None, api_key: str | None = None, provider: str | None = None) -> OpenAI:
    """Return an OpenAI-compatible client.

    If `client` is provided, it's returned unchanged. Otherwise the function
    reads the provider configuration and constructs an OpenAI client using
    the provider-specific API key and base URL.
    """
    if client:
        return client

    info = get_provider_info(provider)
    key = api_key or get_api_key(provider)
    if not key:
        raise RuntimeError(f"Missing API key for provider (env var {info['env']} or LLM_API_KEY)")

    base = info.get("base")
    if base:
        return OpenAI(api_key=key, base_url=base, timeout=120.0, max_retries=3)
    return OpenAI(api_key=key, timeout=120.0, max_retries=3)


def provider_env_name(provider: str | None = None) -> str:
    return get_provider_info(provider)["env"]
