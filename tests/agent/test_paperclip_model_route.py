"""Decision table for HERMES_PAPERCLIP_MODEL_ROUTE wire-model selection."""

from __future__ import annotations

import json

import pytest

from agent.paperclip_model_route import (
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


def test_choose_tool_message_locks_primary():
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


def test_choose_assistant_tool_calls_locks_primary():
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


def test_apply_rewrites_model_and_sticks_after_tools(route_env, monkeypatch):
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

    kwargs2 = {
        "model": "primary-model",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
        ],
    }
    apply_paperclip_model_route(agent, kwargs2)
    assert kwargs2["model"] == "primary-model"
    assert agent._paperclip_model_route_locked is True

    kwargs3 = {"model": "primary-model", "messages": [{"role": "user", "content": "again"}]}
    apply_paperclip_model_route(agent, kwargs3)
    assert kwargs3["model"] == "primary-model"
