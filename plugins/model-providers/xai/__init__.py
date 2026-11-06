"""xAI (Grok) provider profile."""

from providers import register_provider
from providers.base import ProviderProfile, versioned_user_agent

xai = ProviderProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), api_mode="codex_responses", env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1", auth_type="api_key",
    default_headers={"User-Agent": versioned_user_agent("Hermes-Agent")},
)

register_provider(xai)
