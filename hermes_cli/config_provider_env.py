"""Deferred OPTIONAL_ENV_VARS injection from model-provider profiles (YIE-66).

Kept out of ``hermes_cli.config`` so importing config (``hermes --version`` →
``detect_install_method``) never runs ``list_providers()`` / plugin import /
``get_version_info()`` git tax. Setup, pickers, and config surfaces call
:func:`ensure_provider_profile_env_vars` when they need the descriptions.
"""

from __future__ import annotations

_provider_profile_env_vars_injected = False


def provider_api_key_env_names() -> frozenset[str]:
    """Env var names declared by ``auth_type="api_key"`` provider profiles.

    Used to seed ``CORE_DECLARED_ENV_NAMES`` so a platform manifest cannot
    reclassify a provider secret before ``ensure_provider_profile_env_vars``
    runs. Descriptions stay deferred; this only collects names (YIE-66).
    Plugins no longer call ``get_version_info()`` at import, so this is ~0.1s
    rather than multi-second git tax.
    """
    try:
        from providers import list_providers
        return frozenset(
            var
            for profile in list_providers()
            if profile.auth_type == "api_key"
            for var in (profile.env_vars or ())
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return frozenset()


def _inject_profile_env_vars() -> None:
    """Expose env_vars of every ``auth_type="api_key"`` provider via OPTIONAL_ENV_VARS."""
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS

    try:
        from providers import list_providers
    except ImportError:
        return
    try:
        profiles = list_providers()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # A broken/partial plugin tree must not break CLI import or setup entry.
        return
    for _pp in profiles:
        if _pp.auth_type != "api_key":
            continue
        for _var in _pp.env_vars:
            if _var in OPTIONAL_ENV_VARS:
                continue
            _is_key = not _var.endswith(("_BASE_URL", "_URL"))
            _label = _pp.display_name or _pp.name
            OPTIONAL_ENV_VARS[_var] = {
                "description": f"{_label} {'API key' if _is_key else 'base URL override'}",
                "prompt": f"{_label} {'API key' if _is_key else 'base URL (leave empty for default)'}",
                "url": _pp.signup_url or None,
                "password": _is_key,
                "category": "provider",
                "advanced": True}



def ensure_provider_profile_env_vars() -> None:
    """Populate provider API-key descriptions into OPTIONAL_ENV_VARS on first need.

    Not run at import: ``list_providers()`` loads every model-provider plugin, and several of
    those used to call ``get_version_info()`` at module scope (git tax on every cold start).
    ``hermes --version`` / ``-z`` / chat / resident never read those descriptions — only setup,
    pickers, and config surfaces do (YIE-66).
    """
    global _provider_profile_env_vars_injected
    if _provider_profile_env_vars_injected:
        return
    _inject_profile_env_vars()
    _provider_profile_env_vars_injected = True
