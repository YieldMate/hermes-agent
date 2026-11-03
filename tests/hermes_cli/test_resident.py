"""Resident JSONL loop: one live agent across turns; rebind the run token each turn."""

from __future__ import annotations

import io
import json

import pytest


def test_resident_two_turns_reuse_agent_and_rebind_api_key(monkeypatch):
    from hermes_cli import resident
    import hermes_cli.oneshot as oneshot

    seen = {"opens": 0, "turns": 0, "keys": [], "agents": []}

    class FakeHandle:
        def __init__(self):
            self.agent = object()
            seen["agents"].append(self.agent)

    def fake_open(**kwargs):
        seen["opens"] += 1
        return FakeHandle()

    def fake_run(handle, prompt, *, ledger=False):
        seen["turns"] += 1
        seen["keys"].append(__import__("os").environ.get("PAPERCLIP_API_KEY"))
        return f"answer-{seen['turns']}", {"session_id": "sess-1", "completed": True}

    closed = {"n": 0}

    def fake_close(handle):
        if handle is not None:
            closed["n"] += 1

    monkeypatch.setattr(oneshot, "open_live_oneshot_agent", fake_open)
    monkeypatch.setattr(oneshot, "run_live_oneshot_turn", fake_run)
    monkeypatch.setattr(oneshot, "close_live_oneshot_agent", fake_close)
    monkeypatch.setattr(oneshot, "_normalize_toolsets", lambda v: None if not v else ["terminal"])

    lines = [
        json.dumps({
            "prompt": "first",
            "env": {"PAPERCLIP_API_KEY": "key-1", "PAPERCLIP_RUN_ID": "run-1"},
        }),
        json.dumps({
            "prompt": "second",
            "env": {"PAPERCLIP_API_KEY": "key-2", "PAPERCLIP_RUN_ID": "run-2"},
            "session_id": "sess-1",
        }),
        json.dumps({"shutdown": True}),
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(resident.sys, "stdin", stdin)
    monkeypatch.setattr(resident.sys, "stdout", stdout)

    rc = resident.run_resident(idle_sec=60)
    assert rc == 0
    assert seen["opens"] == 1
    assert seen["turns"] == 2
    assert seen["keys"] == ["key-1", "key-2"]
    assert closed["n"] == 1
    out_lines = [json.loads(line) for line in stdout.getvalue().strip().splitlines() if line.strip()]
    assert out_lines[0]["ok"] is True
    assert out_lines[1]["ok"] is True
    assert out_lines[2].get("shutdown") == "requested"


def test_resident_rebuilds_when_model_changes(monkeypatch):
    from hermes_cli import resident
    import hermes_cli.oneshot as oneshot

    opens = {"n": 0}

    class FakeHandle:
        pass

    def fake_open(**kwargs):
        opens["n"] += 1
        return FakeHandle()

    def fake_run(handle, prompt, *, ledger=False):
        return "ok", {"session_id": "sess-1", "completed": True}

    monkeypatch.setattr(oneshot, "open_live_oneshot_agent", fake_open)
    monkeypatch.setattr(oneshot, "run_live_oneshot_turn", fake_run)
    monkeypatch.setattr(oneshot, "close_live_oneshot_agent", lambda h: None)
    monkeypatch.setattr(oneshot, "_normalize_toolsets", lambda v: None if not v else ["terminal"])

    lines = [
        json.dumps({"prompt": "first", "model": "m1"}),
        json.dumps({"prompt": "second", "model": "m2"}),
        json.dumps({"shutdown": True}),
    ]
    monkeypatch.setattr(resident.sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    monkeypatch.setattr(resident.sys, "stdout", io.StringIO())
    assert resident.run_resident(idle_sec=60) == 0
    assert opens["n"] == 2
