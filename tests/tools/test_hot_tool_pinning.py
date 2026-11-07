"""Hot MCP tool pinning v1 (YIE-70): one flag gates pinning, compact tool_search, the tool_describe
cache, MCP spillover at 15k chars, and the usage-report telemetry. Each test drives real imports
against the per-test temp ``HERMES_HOME``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import hot_tool_pinning as hot

REPO_ROOT = Path(__file__).resolve().parents[2]
CAP_SCRIPT = REPO_ROOT / "scripts" / "ci" / "pinned_tools_token_cap.py"
FIXTURE_OVERLAYS = REPO_ROOT / "tests" / "fixtures" / "overlays"
FLAG = "HERMES_HOT_TOOL_PINNING_V1"


def _function(name: str, description: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}}}


def _td(name: str, description: str) -> dict[str, Any]:
    return {"type": "function", "function": _function(name, description)}


@pytest.fixture(autouse=True)
def _clean_pinning_state(monkeypatch):
    for var in (*hot.FLAG_ENV_VARS, hot.PINNED_TOOLS_ENV, hot.PINNED_TOOLS_FILE_ENV):
        monkeypatch.delenv(var, raising=False)
    hot.clear_hot_pinning_runtime_state()
    yield
    hot.clear_hot_pinning_runtime_state()


@pytest.fixture
def mcp_tools():
    """Register MCP tools and return their tool-defs; deregistered on teardown."""
    from tools.registry import registry

    registered: list[str] = []

    def _register(specs: dict[str, str]) -> list[dict[str, Any]]:
        for name, description in specs.items():
            registry.register(name=name, handler=lambda args, **kw: "{}",
                              schema=_function(name, description), toolset="mcp-test")
            registered.append(name)
        return [_td(name, description) for name, description in specs.items()]

    yield _register
    for name in registered:
        registry.deregister(name)


def _tool_names(tool_defs: list[dict[str, Any]]) -> set[str]:
    return {td["function"]["name"] for td in tool_defs}


# --- 1. flag rollback -------------------------------------------------------------------------

def test_flag_off_leaves_no_pin_cache_or_threshold_residue(monkeypatch, mcp_tools):
    from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS, budget_for_context_window
    from tools.tool_search import (
        ToolSearchConfig, assemble_tool_defs, dispatch_tool_describe, dispatch_tool_search,
        is_deferrable_tool_name)

    defs = mcp_tools({"mcp_test_tool": "Send a ping to the test server.",
                      "mcp_test_other": "Look up a record on the test server."})
    config = ToolSearchConfig.from_raw({"enabled": "on"})

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(hot.PINNED_TOOLS_ENV, json.dumps(["mcp_test_tool"]))
    assert hot.pinned_tool_names() == {"mcp_test_tool"}
    assert not is_deferrable_tool_name("mcp_test_tool")
    assert is_deferrable_tool_name("mcp_test_other")
    assembled = assemble_tool_defs(defs, context_length=200_000, config=config)
    assert "mcp_test_tool" in _tool_names(assembled.tool_defs)
    assert "mcp_test_other" not in _tool_names(assembled.tool_defs)
    dispatch_tool_describe({"names": ["mcp_test_other"]}, current_tool_defs=defs)
    hot.record_tool_call("mcp_test_other")
    assert hot.describe_cache_size() == 1 and hot.tool_calls_total() == 1
    assert budget_for_context_window(None).mcp_result_size == hot.HOT_PINNING_MCP_RESULT_SIZE_CHARS

    monkeypatch.delenv(FLAG)
    assert hot.pinned_tool_names() == frozenset()
    assert not hot.is_pinned_tool("mcp_test_tool")
    assert is_deferrable_tool_name("mcp_test_tool")
    assert budget_for_context_window(None).mcp_result_size == DEFAULT_MCP_RESULT_SIZE_CHARS
    assembled = assemble_tool_defs(defs, context_length=200_000, config=config)
    assert "mcp_test_tool" not in _tool_names(assembled.tool_defs)
    assert hot.describe_cache_size() == 0 and hot.tool_calls_total() == 0
    rich = json.loads(dispatch_tool_search({"queries": ["ping test server"]}, current_tool_defs=defs))
    assert isinstance(rich, dict) and {"queries", "results", "tools"} <= set(rich)
    assert rich["tools"]["mcp_test_tool"]["source"] == "mcp"


def test_explicit_clear_drops_override_manifest_cache_and_counters(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    hot.set_pinned_override(["mcp_test_tool"])
    hot.set_describe_cache("mcp_test_tool", "{}")
    hot.record_tool_call("mcp_test_tool")
    assert hot.is_pinned_tool("mcp_test_tool")

    hot.clear_hot_pinning_runtime_state()

    assert not hot.is_pinned_tool("mcp_test_tool")
    assert hot.describe_cache_size() == 0
    assert hot.snapshot_hot_pinning_telemetry()["tool_calls_total"] == 0


def test_paperclip_env_alias_enables_the_flag(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_HOT_TOOL_PINNING_V1", "yes")
    assert hot.is_hot_tool_pinning_v1_enabled()
    monkeypatch.setenv("PAPERCLIP_HOT_TOOL_PINNING_V1", "0")
    assert not hot.is_hot_tool_pinning_v1_enabled()


def test_pinned_names_load_from_manifest_file_or_bare_list(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "on")
    manifest = tmp_path / "pinned-tools.json"
    manifest.write_text(json.dumps({"version": 1, "tools": [{"name": "mcp_a", "callCount": 2}, {"name": "mcp_b"}]}))
    monkeypatch.setenv(hot.PINNED_TOOLS_FILE_ENV, str(manifest))
    assert hot.pinned_tool_names() == {"mcp_a", "mcp_b"}

    monkeypatch.delenv(hot.PINNED_TOOLS_FILE_ENV)
    monkeypatch.setenv(hot.PINNED_TOOLS_ENV, json.dumps(["mcp_c"]))
    assert hot.pinned_tool_names() == {"mcp_c"}


# --- caps -------------------------------------------------------------------------------------

def test_pin_cap_keeps_the_ten_most_used_names():
    entries = [{"name": f"mcp_t{i:02d}", "callCount": i} for i in range(13)]
    names, _ = hot.select_pinned_entries(entries)
    assert len(names) == hot.MAX_PINNED_TOOLS
    assert set(names) == {f"mcp_t{i:02d}" for i in range(3, 13)}


def test_token_cap_drops_least_used_then_later_name_and_logs(monkeypatch, caplog):
    from hermes_cli import tools_config

    monkeypatch.setattr(tools_config, "estimate_schema_tokens", lambda schema: schema["tokens"])
    entries = [
        {"name": "mcp_hot", "callCount": 50, "schema": {"tokens": 4000}},
        {"name": "mcp_b", "callCount": 5, "schema": {"tokens": 4000}},
        {"name": "mcp_a", "callCount": 5, "schema": {"tokens": 4000}},
    ]
    with caplog.at_level("WARNING", logger=hot.logger.name):
        names, total = hot.select_pinned_entries(entries)

    assert names == ["mcp_hot", "mcp_a"] and total == 8000
    assert any("mcp_b" in r.getMessage() for r in caplog.records)


# --- 2. truncation boundary -------------------------------------------------------------------

@pytest.mark.parametrize("use_default_budget", [True, False])
def test_mcp_result_spills_only_past_15000_chars_under_the_flag(monkeypatch, use_default_budget):
    from hermes_constants import get_hermes_home
    from tools.budget_config import DEFAULT_BUDGET, budget_for_context_window
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, maybe_persist_tool_result

    monkeypatch.setenv(FLAG, "1")
    hot.set_pinned_override(["mcp_test_tool"])
    budget = DEFAULT_BUDGET if use_default_budget else budget_for_context_window(None)
    assert budget.resolve_threshold("mcp_test_tool") == hot.HOT_PINNING_MCP_RESULT_SIZE_CHARS

    at_limit = "x" * hot.HOT_PINNING_MCP_RESULT_SIZE_CHARS
    assert maybe_persist_tool_result(at_limit, "mcp_test_tool", "call_at", config=budget) == at_limit

    spilled = maybe_persist_tool_result(at_limit + "x", "mcp_test_tool", "call_over", config=budget)
    assert PERSISTED_OUTPUT_TAG in spilled
    spill_file = get_hermes_home() / "cache" / "spillover" / "call_over.txt"
    assert str(spill_file) in spilled
    assert spill_file.read_text(encoding="utf-8") == at_limit + "x"


def test_flag_off_keeps_the_configured_mcp_threshold():
    from tools.budget_config import DEFAULT_BUDGET, DEFAULT_MCP_RESULT_SIZE_CHARS
    from tools.tool_result_storage import maybe_persist_tool_result

    assert DEFAULT_BUDGET.resolve_threshold("mcp_test_tool") == DEFAULT_MCP_RESULT_SIZE_CHARS
    body = "x" * 20_000
    assert maybe_persist_tool_result(body, "mcp_test_tool", "call_flag_off") == body


# --- 3. tool_search determinism and cap -------------------------------------------------------

def test_search_is_alphabetical_capped_and_deterministic(monkeypatch, mcp_tools):
    from tools.tool_search import dispatch_tool_search

    specs = {f"mcp_test_issue_{c}": f"Manage tracker issue {c} in the test tracker." for c in "abcdefghijklmnopqrstuvwxyz"}
    defs = mcp_tools(specs)
    monkeypatch.setenv(FLAG, "1")

    first = dispatch_tool_search({"queries": ["tracker issue"]}, current_tool_defs=defs)
    again = dispatch_tool_search({"queries": ["tracker issue"]}, current_tool_defs=list(reversed(defs)))

    assert first == again
    records = json.loads(first)
    assert isinstance(records, list) and len(records) == hot.SEARCH_RESULT_CAP
    assert all(set(r) == {"name", "description"} for r in records)
    names = [r["name"] for r in records]
    assert names == sorted(names)
    assert all(len(r["description"]) <= 81 for r in records)


def test_search_merges_queries_into_one_unique_alphabetical_list(monkeypatch, mcp_tools):
    from tools.tool_search import dispatch_tool_search

    defs = mcp_tools({"mcp_test_zeta_ticket": "Create a ticket.", "mcp_test_alpha_calendar": "List calendar events.",
                      "mcp_test_mid_ticket": "Close a ticket."})
    monkeypatch.setenv(FLAG, "1")

    records = json.loads(dispatch_tool_search({"queries": ["ticket", "calendar", "ticket"]}, current_tool_defs=defs))

    assert [r["name"] for r in records] == ["mcp_test_alpha_calendar", "mcp_test_mid_ticket", "mcp_test_zeta_ticket"]


# --- 4. tool_describe cache -------------------------------------------------------------------

def test_describe_second_call_hits_the_cache_until_the_registry_generation_changes(monkeypatch, mcp_tools):
    from tools.tool_search import dispatch_tool_describe

    defs = mcp_tools({"mcp_test_describe": "Describe me."})
    monkeypatch.setenv(FLAG, "1")
    writes: list[str] = []
    real_set = hot.set_describe_cache
    monkeypatch.setattr(hot, "set_describe_cache", lambda name, payload: (writes.append(name), real_set(name, payload))[1])

    first = dispatch_tool_describe({"names": ["mcp_test_describe"]}, current_tool_defs=defs)
    second = dispatch_tool_describe({"names": ["mcp_test_describe"]}, current_tool_defs=defs)

    assert first == second and writes == ["mcp_test_describe"]
    assert hot.describe_cache_size() == 1
    assert json.loads(first)["tools"]["mcp_test_describe"] == {
        "description": "Describe me.", "parameters": {"type": "object", "properties": {}}}

    mcp_tools({"mcp_test_describe_new": "A later registration."})
    dispatch_tool_describe({"names": ["mcp_test_describe"]}, current_tool_defs=defs)
    assert writes == ["mcp_test_describe", "mcp_test_describe"]


def test_describe_payload_matches_flag_off_bytes_and_keeps_not_found(monkeypatch, mcp_tools):
    from tools.tool_search import dispatch_tool_describe

    defs = mcp_tools({"mcp_test_same": "Same bytes either way."})
    args = {"names": ["mcp_test_same", "mcp_test_missing"]}
    flag_off = dispatch_tool_describe(args, current_tool_defs=defs)
    monkeypatch.setenv(FLAG, "1")

    flag_on = dispatch_tool_describe(args, current_tool_defs=defs)

    assert flag_on == flag_off
    assert json.loads(flag_on)["not_found"] == ["mcp_test_missing"]


# --- 5. token cap -----------------------------------------------------------------------------

def _run_cap(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CAP_SCRIPT), *map(str, paths)],
                          capture_output=True, text=True, check=False, cwd=REPO_ROOT)


def test_under_cap_overlay_fixture_passes():
    result = _run_cap(FIXTURE_OVERLAYS)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok: under-cap" in result.stdout


def test_over_cap_overlay_fails_naming_the_agent(tmp_path):
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    text = " ".join(words[(i * 7) % len(words)] + str(i % 97) for i in range(9000))
    tools = [{"name": f"mcp_big_{i}", "callCount": i, "schema": _function(f"mcp_big_{i}", text[i * 2000:])}
             for i in range(4)]
    overlay = tmp_path / "overlays" / "greedy-agent"
    overlay.mkdir(parents=True)
    (overlay / "pinned-tools.json").write_text(json.dumps({"version": 1, "tools": tools}))

    result = _run_cap(tmp_path / "overlays")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "OVER CAP: greedy-agent" in result.stdout


def test_estimate_schema_tokens_counts_the_wire_shape():
    from hermes_cli.tools_config import estimate_schema_tokens, sum_pinned_schema_tokens

    small = _function("mcp_t", "short")
    large = _function("mcp_t", "a much longer description " * 40)
    assert 0 < estimate_schema_tokens(small) < estimate_schema_tokens(large)
    assert sum_pinned_schema_tokens([small, large]) == estimate_schema_tokens(small) + estimate_schema_tokens(large)


# --- 6. telemetry -----------------------------------------------------------------------------

def test_telemetry_snapshot_has_the_four_fields(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    hot.set_pinned_override(["mcp_test_tool"])
    for name in ("tool_search", "tool_describe", "mcp_test_tool", "tool_call"):
        hot.record_tool_call(name)

    snapshot = hot.snapshot_hot_pinning_telemetry()

    assert snapshot == {"tool_search_calls": 1, "tool_describe_calls": 1, "tool_calls_total": 3,
                        "pinned_schema_tokens": snapshot["pinned_schema_tokens"]}
    hot.reset_hot_pinning_telemetry()
    assert hot.snapshot_hot_pinning_telemetry()["tool_calls_total"] == 0


def test_telemetry_is_inert_when_the_flag_is_off():
    hot.record_tool_call("tool_search")
    assert hot.snapshot_hot_pinning_telemetry() == {
        "tool_search_calls": 0, "tool_describe_calls": 0, "tool_calls_total": 0, "pinned_schema_tokens": 0}


def test_handle_function_call_counts_real_tool_calls(monkeypatch, mcp_tools):
    import model_tools

    mcp_tools({"mcp_test_counted": "Counted."})
    monkeypatch.setenv(FLAG, "1")

    model_tools.handle_function_call("mcp_test_counted", {})

    assert hot.tool_calls_total() == 1


def test_usage_report_carries_telemetry_only_under_the_flag(monkeypatch, tmp_path):
    from hermes_cli.oneshot import _write_usage_file

    off, on = tmp_path / "off.json", tmp_path / "on.json"
    _write_usage_file(str(off), {})
    assert "tool_calls_total" not in json.loads(off.read_text())

    monkeypatch.setenv(FLAG, "1")
    hot.record_tool_call("tool_search")
    _write_usage_file(str(on), {})
    report = json.loads(on.read_text())
    assert {"tool_search_calls", "tool_describe_calls", "tool_calls_total", "pinned_schema_tokens"} <= set(report)
    assert report["tool_search_calls"] == 1 and report["tool_calls_total"] == 1
