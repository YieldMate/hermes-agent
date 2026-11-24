"""Decision table for HERMES_PAPERCLIP_MODEL_ROUTE wire-model selection."""

from __future__ import annotations

import json

import pytest

from agent.paperclip_model_route import (
    DEFAULT_CHEAP_TOOL_ALLOWLIST,
    apply_paperclip_model_route,
    choose_paperclip_wire_model,
    load_paperclip_model_route,
)


ROUTE = {
    "primaryModel": "primary-model",
    "cheapModel": "cheap-model",
    "maxCheapInputTokens": 100,
}


@pytest.fixture
def route_env(monkeypatch):
    monkeypatch.setenv(
        "HERMES_PAPERCLIP_MODEL_ROUTE",
        json.dumps(ROUTE),
    )


def _assistant_call(call_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id: str, name: str | None = None, content: str = "ok") -> dict:
    msg: dict = {"role": "tool", "tool_call_id": call_id, "content": content}
    if name is not None:
        msg["name"] = name
    return msg


def test_no_env_leaves_request_unchanged(monkeypatch):
    monkeypatch.delenv("HERMES_PAPERCLIP_MODEL_ROUTE", raising=False)
    agent = type("A", (), {"model": "primary-model"})()
    kwargs = {"model": "primary-model", "messages": [{"role": "user", "content": "hi"}]}
    apply_paperclip_model_route(agent, kwargs)
    assert kwargs["model"] == "primary-model"
    assert getattr(agent, "_paperclip_wire_model", None) is None


def test_invalid_json_leaves_request_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_PAPERCLIP_MODEL_ROUTE", "{not-json")
    assert load_paperclip_model_route() is None
    agent = type("A", (), {"model": "primary-model"})()
    kwargs = {"model": "primary-model", "messages": []}
    apply_paperclip_model_route(agent, kwargs)
    assert kwargs["model"] == "primary-model"


def test_load_defaults_cheap_tool_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_PAPERCLIP_MODEL_ROUTE", json.dumps(ROUTE))
    route = load_paperclip_model_route()
    assert route is not None
    assert route["cheapToolAllowlist"] == list(DEFAULT_CHEAP_TOOL_ALLOWLIST)


def test_choose_under_cap_uses_cheap():
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=[{"role": "user", "content": "hi"}], approx_tokens=10, locked=False,
    )
    assert model == "cheap-model"
    assert locked is False


def test_choose_over_cap_locks_primary():
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=[{"role": "user", "content": "hi"}], approx_tokens=100, locked=False,
    )
    assert model == "primary-model"
    assert locked is True


def test_choose_nameless_tool_message_locks_primary():
    model, locked = choose_paperclip_wire_model(
        ROUTE,
        messages=[
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
        ],
        approx_tokens=1,
        locked=False,
    )
    assert model == "primary-model"
    assert locked is True


def test_choose_nameless_assistant_tool_calls_locks_primary():
    model, locked = choose_paperclip_wire_model(
        ROUTE,
        messages=[{"role": "assistant", "content": None, "tool_calls": [{"id": "t"}]}],
        approx_tokens=1,
        locked=False,
    )
    assert model == "primary-model"
    assert locked is True


def test_choose_already_locked_stays_primary():
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=[{"role": "user", "content": "hi"}], approx_tokens=1, locked=True,
    )
    assert model == "primary-model"
    assert locked is True


def test_choose_discovery_and_linear_stay_cheap():
    messages = [
        {"role": "user", "content": "triage"},
        _assistant_call("1", "tool_search"),
        _tool_result("1", "tool_search"),
        _assistant_call("2", "tool_describe"),
        _tool_result("2", "tool_describe"),
        _assistant_call(
            "3",
            "mcp__paperclip_assigned__mcp_app_gallery_linear_69f0e56_get_issue",
        ),
        _tool_result(
            "3",
            "mcp__paperclip_assigned__mcp_app_gallery_linear_69f0e56_get_issue",
        ),
    ]
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=messages, approx_tokens=10, locked=False,
    )
    assert model == "cheap-model"
    assert locked is False


def test_choose_linear_mixed_with_terminal_locks():
    messages = [
        _assistant_call("1", "mcp__paperclip_assigned__mcp_linear_abc_get_issue"),
        _tool_result("1", "mcp__paperclip_assigned__mcp_linear_abc_get_issue"),
        _assistant_call("2", "terminal"),
        _tool_result("2", "terminal"),
    ]
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=messages, approx_tokens=10, locked=False,
    )
    assert model == "primary-model"
    assert locked is True


def test_choose_custom_allowlist_honored():
    route = {
        **ROUTE,
        "cheapToolAllowlist": ["paperclip_self", "tool_search"],
    }
    messages = [
        _assistant_call("1", "tool_search"),
        _tool_result("1", "tool_search"),
        _assistant_call("2", "mcp__paperclip_assigned__paperclip_self_get_issue"),
        _tool_result("2", "mcp__paperclip_assigned__paperclip_self_get_issue"),
    ]
    model, locked = choose_paperclip_wire_model(
        route, messages=messages, approx_tokens=10, locked=False,
    )
    assert model == "cheap-model"
    assert locked is False

    # Linear is not on the custom allowlist → lock
    messages2 = [
        _assistant_call("3", "mcp__paperclip_assigned__mcp_linear_get_issue"),
        _tool_result("3", "mcp__paperclip_assigned__mcp_linear_get_issue"),
    ]
    model2, locked2 = choose_paperclip_wire_model(
        route, messages=messages2, approx_tokens=10, locked=False,
    )
    assert model2 == "primary-model"
    assert locked2 is True


def test_choose_unresolved_tool_result_fails_closed():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
    ]
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=messages, approx_tokens=1, locked=False,
    )
    assert model == "primary-model"
    assert locked is True


def test_choose_resolves_tool_name_from_assistant_call_id():
    messages = [
        _assistant_call("1", "tool_search"),
        # No name on the tool result — resolve via tool_call_id.
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=messages, approx_tokens=1, locked=False,
    )
    assert model == "cheap-model"
    assert locked is False


def test_choose_allowlisted_tools_still_lock_on_token_cap():
    messages = [
        _assistant_call("1", "tool_search"),
        _tool_result("1", "tool_search"),
    ]
    model, locked = choose_paperclip_wire_model(
        ROUTE, messages=messages, approx_tokens=100, locked=False,
    )
    assert model == "primary-model"
    assert locked is True


def test_apply_rewrites_model_and_sticks_after_non_allowlisted_tools(route_env, monkeypatch):
    monkeypatch.setattr(
        "agent.paperclip_model_route._rough_input_tokens",
        lambda api_kwargs, messages: 5,
    )
    agent = type("A", (), {"model": "primary-model"})()
    kwargs = {"model": "primary-model", "messages": [{"role": "user", "content": "hi"}]}
    apply_paperclip_model_route(agent, kwargs)
    assert kwargs["model"] == "cheap-model"
    assert agent._paperclip_wire_model == "cheap-model"
    assert not getattr(agent, "_paperclip_model_route_locked", False)

    kwargs_linear = {
        "model": "primary-model",
        "messages": [
            {"role": "user", "content": "hi"},
            _assistant_call("1", "mcp.linear-abc:get_issue"),
            _tool_result("1", "mcp.linear-abc:get_issue"),
        ],
    }
    apply_paperclip_model_route(agent, kwargs_linear)
    assert kwargs_linear["model"] == "cheap-model"
    assert not getattr(agent, "_paperclip_model_route_locked", False)

    kwargs2 = {
        "model": "primary-model",
        "messages": [
            {"role": "user", "content": "hi"},
            _assistant_call("1", "terminal"),
            _tool_result("1", "terminal"),
        ],
    }
    apply_paperclip_model_route(agent, kwargs2)
    assert kwargs2["model"] == "primary-model"
    assert agent._paperclip_model_route_locked is True

    kwargs3 = {"model": "primary-model", "messages": [{"role": "user", "content": "again"}]}
    apply_paperclip_model_route(agent, kwargs3)
    assert kwargs3["model"] == "primary-model"


def test_token_estimate_failure_fails_closed_to_primary(route_env, monkeypatch):
    def boom(api_kwargs, messages):
        raise RuntimeError("estimate failed")

    monkeypatch.setattr("agent.paperclip_model_route._rough_input_tokens", boom)
    agent = type("A", (), {"model": "primary-model"})()
    kwargs = {"model": "primary-model", "messages": [{"role": "user", "content": "hi"}]}
    apply_paperclip_model_route(agent, kwargs)
    assert kwargs["model"] == "primary-model"
    assert agent._paperclip_model_route_locked is True
    assert agent._paperclip_wire_model == "primary-model"


def test_bypass_skips_paperclip_routing(route_env):
    agent = type("A", (), {"model": "primary-model", "_paperclip_model_route_bypass": True})()
    kwargs = {"model": "primary-model", "messages": [{"role": "user", "content": "hi"}]}
    apply_paperclip_model_route(agent, kwargs)
    assert kwargs["model"] == "primary-model"
    assert getattr(agent, "_paperclip_wire_model", None) is None
