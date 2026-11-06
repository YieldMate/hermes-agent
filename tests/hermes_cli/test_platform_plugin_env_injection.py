"""User-installed platform plugins feed OPTIONAL_ENV_VARS like bundled ones (#46600, redo of #46964).

The Desktop Gateway form and ``hermes config`` render env fields from ``OPTIONAL_ENV_VARS``; before
this, only ``plugins/platforms/*`` in the repo was scanned, so a third-party platform's
``requires_env`` prompts/descriptions/password flags never reached the UI.
"""

import hermes_cli.config as config_mod


def _manifest(path, text):
    path.mkdir(parents=True)
    (path / "plugin.yaml").write_text(text, encoding="utf-8")


def test_user_platform_plugins_inject_env_metadata_but_non_platforms_do_not(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    _manifest(home / "plugins" / "demo-platform", (
        "name: demo-platform\nkind: platform\nlabel: Demo Platform\n"
        "requires_env:\n  - name: DEMO_PLATFORM_TOKEN\n    description: Token for the demo platform\n"
        "    prompt: Demo token\n    url: https://example.invalid/demo\n"
        "optional_env:\n  - name: DEMO_PLATFORM_ROOM\n    password: false\n"))
    _manifest(home / "plugins" / "platforms" / "nested-platform", (
        "name: nested-platform\nrequires_env:\n  - NESTED_PLATFORM_SECRET\n"))
    _manifest(home / "plugins" / "ignore-me", "name: ignore-me\nkind: backend\nrequires_env:\n  - IGNORE_ME_TOKEN\n")
    keys = ["DEMO_PLATFORM_TOKEN", "DEMO_PLATFORM_ROOM", "NESTED_PLATFORM_SECRET", "IGNORE_ME_TOKEN"]
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
    for key in keys:
        monkeypatch.delitem(config_mod.OPTIONAL_ENV_VARS, key, raising=False)

    config_mod._inject_platform_plugin_env_vars()

    try:
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_TOKEN"] == {
            "description": "Token for the demo platform", "prompt": "Demo token",
            "url": "https://example.invalid/demo", "password": True, "category": "messaging"}
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_ROOM"]["password"] is False
        assert config_mod.OPTIONAL_ENV_VARS["NESTED_PLATFORM_SECRET"]["password"] is True  # category dir needs no kind
        assert "IGNORE_ME_TOKEN" not in config_mod.OPTIONAL_ENV_VARS
    finally:
        for key in keys:
            config_mod.OPTIONAL_ENV_VARS.pop(key, None)


def test_platform_inject_skips_core_declared_provider_names(monkeypatch, tmp_path):
    """Injection-only provider keys stay provider even when a platform lists them (YIE-66).

    Descriptions are deferred, so these names are absent from OPTIONAL_ENV_VARS at
    platform-inject time; CORE_DECLARED_ENV_NAMES must still block reclassification.
    """
    home = tmp_path / "hermes-home"
    # Use a name that is harvested into CORE_DECLARED via provider profiles but is
    # typically absent from OPTIONAL_ENV_VARS until ensure_provider_profile_env_vars().
    probe = next(
        (n for n in sorted(config_mod.CORE_DECLARED_ENV_NAMES)
         if n not in config_mod.OPTIONAL_ENV_VARS and n.endswith("_API_KEY")),
        None,
    )
    assert probe, "expected at least one CORE_DECLARED provider key deferred from OPTIONAL_ENV_VARS"
    _manifest(home / "plugins" / "claim-provider", (
        f"name: claim-provider\nkind: platform\nlabel: Claim Provider\n"
        f"requires_env:\n  - {probe}\n"))
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
    before = config_mod.OPTIONAL_ENV_VARS.get(probe)

    config_mod._inject_platform_plugin_env_vars()

    assert config_mod.OPTIONAL_ENV_VARS.get(probe) is before
    assert probe not in config_mod.OPTIONAL_ENV_VARS or (
        config_mod.OPTIONAL_ENV_VARS[probe].get("category") != "messaging"
    )
    config_mod.ensure_provider_profile_env_vars()
    assert config_mod.OPTIONAL_ENV_VARS[probe]["category"] == "provider"
