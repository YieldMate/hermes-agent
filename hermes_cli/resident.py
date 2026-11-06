"""Resident multi-turn mode for Paperclip: one AIAgent, many JSONL turns.

Paperclip keeps this process alive across heartbeats for a single issue so the
next comment is another turn on the same agent instead of a cold ``chat -q``.

Stdin: one JSON object per line::

    {"prompt": "...", "ephemeral_system_prompt": "...", "env": {...},
     "mcp_overlay": "/path/overlay.json", "session_id": "...", "model": "...",
     "provider": "...", "toolsets": "terminal,file,...", "shutdown": false}

Stdout: one JSON object per line::

    {"ok": true, "response": "...", "session_id": "...", "disposition": {...}}

Idle timeout (default 900s) or a ``{"shutdown": true}`` line exits cleanly.
Idle is measured from the last *completed* turn and is checked with
``select.select`` while waiting on stdin so a held-open silent pipe still
exits (Paperclip adapters pass ``--resident-idle-sec 86400``).
"""

from __future__ import annotations

import io
import json
import logging
import os
import select
import sys
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_IDLE_SEC = 900.0


def _read_turn(line: str) -> dict[str, Any]:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("resident turn must be a JSON object")
    return data


def _write_result(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _apply_turn_env(env: object) -> dict[str, Optional[str]]:
    """Set PAPERCLIP_* / overlay env for this turn; return prior values to restore."""
    prior: dict[str, Optional[str]] = {}
    if not isinstance(env, dict):
        return prior
    for key, value in env.items():
        if not isinstance(key, str) or not key.startswith(("PAPERCLIP_", "HERMES_")):
            continue
        prior[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return prior


def _restore_env(prior: dict[str, Optional[str]]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_disposition_from_result(result: dict, response: str) -> Optional[dict]:
    """Read HERMES_DISPOSITION_FILE if the agent wrote it; else None."""
    path = (os.getenv("HERMES_DISPOSITION_FILE") or "").strip()
    if not path:
        return None
    try:
        raw = open(path, encoding="utf-8").read()
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


_PIN_ENV_KEYS = (
    "PAPERCLIP_HOT_TOOL_PINNING_V1",
    "HERMES_HOT_TOOL_PINNING_V1",
    "HERMES_PINNED_TOOLS_FILE",
    "HERMES_PINNED_TOOLS",
)

_SESSION_USAGE_ATTRS = (
    ("input_tokens", "session_input_tokens"),
    ("output_tokens", "session_output_tokens"),
    ("total_tokens", "session_total_tokens"),
    ("estimated_cost_usd", "session_estimated_cost_usd"),
    ("cache_read_tokens", "session_cache_read_tokens"),
    ("cache_write_tokens", "session_cache_write_tokens"),
    ("reasoning_tokens", "session_reasoning_tokens"),
)


def _clear_pin_env() -> dict[str, Optional[str]]:
    """Pop pinning vars so a flag-off turn cannot inherit spawn-env residual state."""
    prior: dict[str, Optional[str]] = {}
    for key in _PIN_ENV_KEYS:
        prior[key] = os.environ.get(key)
        os.environ.pop(key, None)
    return prior


def _session_usage_snapshot(agent: object) -> dict[str, float]:
    snap: dict[str, float] = {}
    for out_key, attr in _SESSION_USAGE_ATTRS:
        val = getattr(agent, attr, None)
        if isinstance(val, (int, float)):
            snap[out_key] = float(val)
    return snap


def _delta_usage(before: dict[str, float], result: dict) -> dict:
    """Convert session-cumulative token/cost fields into per-turn deltas for Paperclip billing."""
    out = dict(result) if isinstance(result, dict) else {}
    for key, prior in before.items():
        after = out.get(key)
        if isinstance(after, (int, float)):
            delta = after - prior
            out[key] = int(delta) if key != "estimated_cost_usd" and float(delta).is_integer() else max(0.0, delta)
            if out[key] < 0:
                out[key] = 0 if key != "estimated_cost_usd" else 0.0
    return out


def _write_resident_usage(result: dict, failure: Optional[str] = None) -> None:
    """Mirror oneshot ``--usage-file`` so Paperclip can merge YIE-70 telemetry on warm turns."""
    path = (os.getenv("HERMES_USAGE_FILE") or "").strip()
    if not path:
        return
    from hermes_cli.oneshot import _write_usage_file
    _write_usage_file(path, result if isinstance(result, dict) else {}, failure=failure)


def _fingerprint(model, provider, toolsets) -> tuple:
    return (model or None, provider or None, toolsets if toolsets is not None else None)


def _stdin_readable(timeout: float) -> bool:
    """True when stdin has data/EOF ready within timeout; False on idle wait expiry.

    Stdin is a pipe under Paperclip, so select is safe. Non-selectable test
    doubles (StringIO) fall through to "ready" and let readline drive EOF.
    """
    if timeout < 0:
        return False
    try:
        fileno = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        # Tests inject StringIO; treat as ready so readline returns immediately.
        return True
    try:
        ready, _, _ = select.select([fileno], [], [], max(timeout, 0.0))
        return bool(ready)
    except (OSError, ValueError):
        return True


def _wait_for_turn_line(idle_sec: float, last_activity: float) -> tuple[Optional[str], Optional[str]]:
    """Block until a stdin line, EOF, or strict idle (> idle_sec since last_activity).

    Returns ``(line, None)`` for data (including blank lines), ``("", None)`` is
    not used — EOF is ``(None, "eof")``, idle is ``(None, "idle")``.
    Boundary: idle fires only when elapsed is *strictly greater* than idle_sec
    (remaining < 0). Exactly idle_sec keeps waiting.
    """
    while True:
        remaining = idle_sec - (time.monotonic() - last_activity)
        # Strictly greater than idle_sec → remaining < 0.
        if remaining < 0:
            return None, "idle"
        if not _stdin_readable(remaining):
            # select timed out; re-check the clock (exactly idle_sec stays alive).
            # Yield so a zero-timeout poll cannot peg the CPU until wall time
            # tips remaining below zero.
            if remaining == 0:
                time.sleep(0.001)
            continue
        line = sys.stdin.readline()
        if line == "":
            return None, "eof"
        return line, None


def run_resident(
    *,
    idle_sec: float = _DEFAULT_IDLE_SEC,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    resume: Optional[str] = None,
) -> int:
    """Serve turns until stdin closes, idle timeout, or shutdown.

    Builds one AIAgent on the first turn and reuses it for later turns so
    ``_paperclip_model_route_locked`` and in-memory history stick across wakes.
    Rebuilds only when model/provider/toolsets change.
    """
    from hermes_cli.oneshot import (
        _normalize_toolsets,
        close_live_oneshot_agent,
        open_live_oneshot_agent,
        run_live_oneshot_turn,
    )

    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["HERMES_ACCEPT_HOOKS"] = "1"
    os.environ["HERMES_INPROCESS_CHAT"] = "1"

    agent_model = model
    agent_provider = provider
    agent_toolsets = toolsets
    session_id = resume
    last_activity = time.monotonic()
    handle = None
    handle_fp = None

    try:
        while True:
            line, reason = _wait_for_turn_line(idle_sec, last_activity)
            if reason == "idle":
                _write_result({"ok": True, "shutdown": "idle"})
                return 0
            if reason == "eof":
                _write_result({"ok": True, "shutdown": "eof"})
                return 0
            assert line is not None
            line = line.strip()
            if not line:
                continue

            try:
                turn = _read_turn(line)
            except Exception as exc:
                _write_result({"ok": False, "error": f"invalid turn: {exc}"})
                continue

            if turn.get("shutdown") is True:
                _write_result({"ok": True, "shutdown": "requested"})
                return 0

            prompt = turn.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                _write_result({"ok": False, "error": "missing prompt"})
                continue

            ephemeral = turn.get("ephemeral_system_prompt")
            if isinstance(ephemeral, str) and ephemeral.strip():
                os.environ["HERMES_EPHEMERAL_SYSTEM_PROMPT"] = ephemeral
            overlay = turn.get("mcp_overlay")
            if isinstance(overlay, str) and overlay.strip():
                os.environ["HERMES_MCP_OVERLAY"] = overlay.strip()
            elif "mcp_overlay" in turn:
                os.environ.pop("HERMES_MCP_OVERLAY", None)

            # Pinning is turn-scoped: clear spawn residuals first, then apply turn.env
            # (explicit nulls from Paperclip pop keys for flag-off rollback).
            _clear_pin_env()
            prior = _apply_turn_env(turn.get("env"))
            turn_model = turn.get("model") if isinstance(turn.get("model"), str) else agent_model
            turn_provider = turn.get("provider") if isinstance(turn.get("provider"), str) else agent_provider
            turn_toolsets = turn.get("toolsets") if turn.get("toolsets") is not None else agent_toolsets
            turn_resume = turn.get("session_id") if isinstance(turn.get("session_id"), str) else session_id
            turn_fp = _fingerprint(turn_model, turn_provider, turn_toolsets)
            from tools.hot_tool_pinning import (
                freeze_pins_from_env,
                release_if_disabled,
                reset_hot_pinning_telemetry,
            )
            # Each Paperclip run owns its own HERMES_USAGE_FILE; reset per-turn so counters
            # are run-scoped, not cumulative across the warm resident's lifetime (YIE-70).
            reset_hot_pinning_telemetry()
            release_if_disabled()
            usage_file_set = bool((os.getenv("HERMES_USAGE_FILE") or "").strip())
            before_usage: dict[str, float] = {}

            try:
                if handle is None or handle_fp != turn_fp:
                    if handle is not None:
                        close_live_oneshot_agent(handle)
                        handle = None
                    # Freeze pins at agent build so dispatch gates match the tool surface.
                    freeze_pins_from_env()
                    handle = open_live_oneshot_agent(
                        model=turn_model,
                        provider=turn_provider,
                        toolsets=_normalize_toolsets(turn_toolsets),
                        use_config_toolsets=_normalize_toolsets(turn_toolsets) is None,
                        resume=turn_resume,
                    )
                    handle_fp = turn_fp
                else:
                    # Keep the frozen pin set; still clear residual state when the flag flips off.
                    release_if_disabled()

                before_usage = _session_usage_snapshot(getattr(handle, "agent", None) or object())
                response, result = run_live_oneshot_turn(
                    handle, prompt, ledger=usage_file_set,
                )
                usage_payload = _delta_usage(
                    before_usage, result if isinstance(result, dict) else {},
                )
                if isinstance(result, dict) and result.get("session_id"):
                    session_id = result["session_id"]
                disposition = _write_disposition_from_result(
                    result if isinstance(result, dict) else {},
                    response or "",
                )
                _write_result({
                    "ok": True,
                    "response": response or "",
                    "session_id": session_id,
                    "disposition": disposition,
                    "result": {
                        k: result.get(k)
                        for k in ("failed", "completed", "partial")
                        if isinstance(result, dict)
                    },
                })
                _write_resident_usage(usage_payload)
            except Exception as exc:
                logger.exception("resident turn failed")
                failure_usage = {}
                if handle is not None:
                    failure_usage = _delta_usage(
                        before_usage,
                        _session_usage_snapshot(getattr(handle, "agent", None) or object()),
                    )
                _write_result({"ok": False, "error": str(exc), "session_id": session_id})
                _write_resident_usage(failure_usage, failure=str(exc))
                # Drop a broken agent so the next turn rebuilds cleanly.
                if handle is not None:
                    close_live_oneshot_agent(handle)
                    handle = None
                    handle_fp = None
            finally:
                _restore_env(prior)
                # Drop the run-scoped API key between turns. Never restore pin_prior (spawn
                # residuals); leave pin keys cleared until the next turn.env re-applies them.
                os.environ.pop("PAPERCLIP_API_KEY", None)
                for key in _PIN_ENV_KEYS:
                    os.environ.pop(key, None)

            # Idle counts from the last *completed* turn (success or failure).
            last_activity = time.monotonic()
            agent_model = turn_model or agent_model
            agent_provider = turn_provider or agent_provider
    finally:
        close_live_oneshot_agent(handle)
        # Describe cache + pin override live for the warm resident's lifetime; clear on exit.
        from tools.hot_tool_pinning import clear_hot_pinning_runtime_state
        clear_hot_pinning_runtime_state()


def cmd_resident_from_args(args) -> None:
    """Entry from ``hermes chat --resident``."""
    idle = getattr(args, "resident_idle_sec", None)
    try:
        idle_sec = float(idle) if idle is not None else _DEFAULT_IDLE_SEC
    except (TypeError, ValueError):
        idle_sec = _DEFAULT_IDLE_SEC
    rc = run_resident(
        idle_sec=idle_sec,
        model=getattr(args, "model", None),
        provider=getattr(args, "provider", None),
        toolsets=getattr(args, "toolsets", None),
        resume=getattr(args, "resume", None),
    )
    raise SystemExit(rc)
