"""Provider OPTIONAL_ENV_VARS injection is deferred off the cold-start import path (YIE-66)."""

from __future__ import annotations

from types import SimpleNamespace


def test_config_provider_env_does_not_discover_until_ensure(monkeypatch):
    """The deferred sibling must not call list_providers until ensure() runs."""
    import sys

    from hermes_cli import config_provider_env as provider_env

    calls: list[str] = []

    def boom():
        calls.append("list_providers")
        raise AssertionError("list_providers must not run until ensure()")

    monkeypatch.setitem(sys.modules, "providers", SimpleNamespace(list_providers=boom))
    provider_env._provider_profile_env_vars_injected = False

    assert calls == []
    # Importing the sibling alone must not discover.
    assert provider_env._provider_profile_env_vars_injected is False


def test_ensure_provider_profile_env_vars_discovers_once(monkeypatch):
    from hermes_cli import config
    from hermes_cli import config_provider_env as provider_env
    import providers as providers_mod

    calls: list[int] = []

    class _P:
        auth_type = "api_key"
        env_vars = ("YIE66_PROBE_API_KEY",)
        display_name = "YIE66 Probe"
        name = "yie66-probe"
        signup_url = ""

    def fake_list():
        calls.append(1)
        return [_P()]

    monkeypatch.setattr(providers_mod, "list_providers", fake_list)

    provider_env._provider_profile_env_vars_injected = False
    config.OPTIONAL_ENV_VARS.pop("YIE66_PROBE_API_KEY", None)

    config.ensure_provider_profile_env_vars()
    config.ensure_provider_profile_env_vars()

    assert calls == [1]
    assert config.OPTIONAL_ENV_VARS["YIE66_PROBE_API_KEY"]["category"] == "provider"
    assert config.OPTIONAL_ENV_VARS["YIE66_PROBE_API_KEY"]["password"] is True
