"""Resident JSONL loop: one live agent across turns; idle fires mid-read."""

from __future__ import annotations

import io
import json
import os
import threading
import time

import pytest


def test_resident_two_turns_reuse_agent_and_rebind_api_key(monkeypatch):
    from hermes_cli import resident
    from hermes_cli import oneshot

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
    from hermes_cli import oneshot

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


def _install_silent_pipe_stdin(monkeypatch, resident):
    """Open pipe stdin (write end held) so select can idle without EOF."""
    r_fd, w_fd = os.pipe()

    class PipeStdin:
        def fileno(self):
            return r_fd

        def readline(self):
            raise AssertionError("readline must not run while select reports idle")

    monkeypatch.setattr(resident.sys, "stdin", PipeStdin())
    stdout = io.StringIO()
    monkeypatch.setattr(resident.sys, "stdout", stdout)
    return r_fd, w_fd, stdout


def test_idle_exactly_idle_sec_still_alive(monkeypatch):
    """(a) idle exactly idle_sec with stdin open and silent: resident STILL ALIVE."""
    from hermes_cli import resident

    idle_sec = 10.0
    clock = {"t": 0.0}
    select_calls = {"n": 0}
    monkeypatch.setattr(resident.time, "monotonic", lambda: clock["t"])

    def fake_select(_r, _w, _x, timeout):
        select_calls["n"] += 1
        if select_calls["n"] == 1:
            # Land exactly on the idle boundary; must NOT exit (remaining == 0).
            clock["t"] = idle_sec
            return [], [], []
        # Stay on the boundary until the test thread advances past it.
        return [], [], []

    monkeypatch.setattr(resident.select, "select", fake_select)
    r_fd, w_fd, _stdout = _install_silent_pipe_stdin(monkeypatch, resident)
    result = {"reason": None, "done": False}

    def runner():
        _line, reason = resident._wait_for_turn_line(idle_sec, last_activity=0.0)
        result["reason"] = reason
        result["done"] = True

    try:
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        deadline = time.time() + 2.0
        while select_calls["n"] < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert select_calls["n"] >= 2, "expected a wait at exactly idle_sec"
        assert result["done"] is False, "must still be alive at exactly idle_sec"
        clock["t"] = idle_sec + 1e-6
        thread.join(timeout=2.0)
        assert result["done"] is True
        assert result["reason"] == "idle"
    finally:
        os.close(w_fd)
        os.close(r_fd)


def test_idle_sec_plus_one_tick_exits_idle(monkeypatch):
    """(b) idle_sec + 1 tick: exits 0 with shutdown idle."""
    from hermes_cli import resident
    from hermes_cli import oneshot

    monkeypatch.setattr(oneshot, "close_live_oneshot_agent", lambda h: None)
    idle_sec = 5.0
    # last_activity is sampled at t=0 inside run_resident; first select tips past.
    clock = {"t": 0.0}

    def fake_select(_r, _w, _x, timeout):
        clock["t"] = idle_sec + 1e-6
        return [], [], []

    monkeypatch.setattr(resident.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(resident.time, "sleep", lambda _s: None)
    monkeypatch.setattr(resident.select, "select", fake_select)
    r_fd, w_fd, stdout = _install_silent_pipe_stdin(monkeypatch, resident)
    try:
        rc = resident.run_resident(idle_sec=idle_sec)
        assert rc == 0
        out = [json.loads(line) for line in stdout.getvalue().strip().splitlines() if line.strip()]
        assert out[-1] == {"ok": True, "shutdown": "idle"}
    finally:
        os.close(w_fd)
        os.close(r_fd)


def test_turn_near_deadline_resets_idle_window(monkeypatch):
    """(c) A turn at idle_sec - epsilon resets last_activity for another full window."""
    from hermes_cli import resident
    from hermes_cli import oneshot

    monkeypatch.setattr(oneshot, "open_live_oneshot_agent", lambda **kw: object())
    monkeypatch.setattr(
        oneshot,
        "run_live_oneshot_turn",
        lambda handle, prompt, *, ledger=False: ("ok", {"session_id": "s", "completed": True}),
    )
    monkeypatch.setattr(oneshot, "close_live_oneshot_agent", lambda h: None)
    monkeypatch.setattr(oneshot, "_normalize_toolsets", lambda v: None)

    idle_sec = 10.0
    clock = {"t": 0.0}
    last_activity_samples = []
    monkeypatch.setattr(resident.time, "monotonic", lambda: clock["t"])

    r_fd, w_fd = os.pipe()
    turns = [
        json.dumps({"prompt": "keep-alive"}) + "\n",
        json.dumps({"shutdown": True}) + "\n",
    ]
    turn_i = {"i": 0}

    class PipeStdin:
        def fileno(self):
            return r_fd

        def readline(self):
            i = turn_i["i"]
            turn_i["i"] += 1
            return turns[i] if i < len(turns) else ""

    real_wait = resident._wait_for_turn_line

    def wrapping_wait(idle, last_activity):
        last_activity_samples.append(last_activity)
        return real_wait(idle, last_activity)

    def fake_select(_r, _w, _x, timeout):
        if turn_i["i"] == 0:
            clock["t"] = idle_sec - 0.01
            return [r_fd], [], []
        if turn_i["i"] == 1:
            # Still inside the reset window after the keep-alive turn.
            clock["t"] = last_activity_samples[-1] + 0.05
            return [r_fd], [], []
        clock["t"] += float(timeout or 0)
        return [], [], []

    monkeypatch.setattr(resident, "_wait_for_turn_line", wrapping_wait)
    monkeypatch.setattr(resident.select, "select", fake_select)
    monkeypatch.setattr(resident.sys, "stdin", PipeStdin())
    stdout = io.StringIO()
    monkeypatch.setattr(resident.sys, "stdout", stdout)

    try:
        rc = resident.run_resident(idle_sec=idle_sec)
        assert rc == 0
        out = [json.loads(line) for line in stdout.getvalue().strip().splitlines() if line.strip()]
        assert out[0]["ok"] is True and out[0].get("response") == "ok"
        assert out[1].get("shutdown") == "requested"
        # last_activity was reset after the keep-alive turn (second sample > first start).
        assert len(last_activity_samples) >= 2
        assert last_activity_samples[1] >= idle_sec - 0.01
    finally:
        os.close(w_fd)
        os.close(r_fd)


def test_held_open_silent_stdin_exits_at_deadline(monkeypatch):
    """(d) Regression: stdin held open with no data still exits at the idle deadline."""
    from hermes_cli import resident
    from hermes_cli import oneshot

    monkeypatch.setattr(oneshot, "close_live_oneshot_agent", lambda h: None)
    idle_sec = 3.0
    clock = {"t": 0.0}

    def fake_select(_r, _w, _x, timeout):
        timeout = float(timeout or 0)
        if timeout <= 0:
            # Tip past the strict boundary the way wall-clock sleep would.
            clock["t"] += 1e-6
        else:
            clock["t"] += timeout
        return [], [], []

    monkeypatch.setattr(resident.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(resident.time, "sleep", lambda _s: None)
    monkeypatch.setattr(resident.select, "select", fake_select)
    r_fd, w_fd, stdout = _install_silent_pipe_stdin(monkeypatch, resident)
    try:
        rc = resident.run_resident(idle_sec=idle_sec)
        assert rc == 0
        out = [json.loads(line) for line in stdout.getvalue().strip().splitlines() if line.strip()]
        assert out[-1] == {"ok": True, "shutdown": "idle"}
        assert clock["t"] > idle_sec
    finally:
        os.close(w_fd)
        os.close(r_fd)


@pytest.mark.platforms("linux")
def test_held_open_pipe_exits_without_data_real_select(tmp_path):
    """Live pipe + real select: process exits idle without any turn bytes."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    idle_sec = 0.35
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    code = f"""
import sys
sys.path.insert(0, {str(repo)!r})
from hermes_cli import resident
import hermes_cli.oneshot as oneshot
oneshot.close_live_oneshot_agent = lambda h: None
sys.exit(resident.run_resident(idle_sec={idle_sec}))
"""
    r, w = os.pipe()
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=r,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(repo),
            env=env,
        )
        deadline = time.time() + idle_sec + 2.5
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None, "resident did not exit while stdin stayed open"
        assert proc.returncode == 0, proc.stderr.read() if proc.stderr else ""
        out_lines = [json.loads(line) for line in (proc.stdout.read() or "").splitlines() if line.strip()]
        assert out_lines[-1] == {"ok": True, "shutdown": "idle"}
    finally:
        os.close(r)
        os.close(w)
        if proc is not None and proc.poll() is None:
            proc.kill()


def test_resident_writes_usage_file_and_clears_pinning_state(tmp_path, monkeypatch):
    """YIE-70: warm turns must emit HERMES_USAGE_FILE with hot-pinning counters; exit clears cache."""
    from hermes_cli import resident
    from hermes_cli import oneshot
    from tools import hot_tool_pinning

    usage_a = tmp_path / "run-a.usage.json"
    usage_b = tmp_path / "run-b.usage.json"
    monkeypatch.setenv("HERMES_HOT_TOOL_PINNING_V1", "1")
    hot_tool_pinning.clear_hot_pinning_runtime_state()

    class FakeHandle:
        def __init__(self):
            self.agent = object()

    calls = {"n": 0}

    def fake_run(handle, prompt, *, ledger=False):
        calls["n"] += 1
        hot_tool_pinning.record_tool_call("tool_search")
        hot_tool_pinning.record_tool_call("terminal")
        hot_tool_pinning.set_describe_cache("mcp__x__y", '{"description":"d","parameters":{}}')
        return f"ok-{calls['n']}", {"session_id": "sess", "completed": True, "input_tokens": 3}

    monkeypatch.setattr(oneshot, "open_live_oneshot_agent", lambda **kw: FakeHandle())
    monkeypatch.setattr(oneshot, "run_live_oneshot_turn", fake_run)
    monkeypatch.setattr(oneshot, "close_live_oneshot_agent", lambda h: None)
    monkeypatch.setattr(oneshot, "_normalize_toolsets", lambda v: None)

    lines = [
        json.dumps({"prompt": "a", "env": {"HERMES_USAGE_FILE": str(usage_a)}}),
        json.dumps({"prompt": "b", "env": {"HERMES_USAGE_FILE": str(usage_b)}}),
        json.dumps({"shutdown": True}),
    ]
    monkeypatch.setattr(resident.sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    monkeypatch.setattr(resident.sys, "stdout", io.StringIO())

    assert resident.run_resident(idle_sec=60) == 0

    report_a = json.loads(usage_a.read_text())
    report_b = json.loads(usage_b.read_text())
    assert report_a["tool_search_calls"] == 1
    assert report_a["tool_calls_total"] == 2
    # Per-run reset: second turn must not accumulate the first turn's counts.
    assert report_b["tool_search_calls"] == 1
    assert report_b["tool_calls_total"] == 2
    assert hot_tool_pinning.describe_cache_size() == 0
