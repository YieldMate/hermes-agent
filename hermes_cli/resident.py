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
"""

from __future__ import annotations

import json
import logging
import os
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


def _fingerprint(model, provider, toolsets) -> tuple:
    return (model or None, provider or None, toolsets if toolsets is not None else None)


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
            remaining = idle_sec - (time.monotonic() - last_activity)
            if remaining <= 0:
                _write_result({"ok": True, "shutdown": "idle"})
                return 0

            # Blocking readline; Paperclip holds the process and writes the next turn.
            # Idle is checked between turns, not mid-read (select would need nonblock tty).
            line = sys.stdin.readline()
            if line == "":
                _write_result({"ok": True, "shutdown": "eof"})
                return 0
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

            prior = _apply_turn_env(turn.get("env"))
            turn_model = turn.get("model") if isinstance(turn.get("model"), str) else agent_model
            turn_provider = turn.get("provider") if isinstance(turn.get("provider"), str) else agent_provider
            turn_toolsets = turn.get("toolsets") if turn.get("toolsets") is not None else agent_toolsets
            turn_resume = turn.get("session_id") if isinstance(turn.get("session_id"), str) else session_id
            turn_fp = _fingerprint(turn_model, turn_provider, turn_toolsets)

            try:
                if handle is None or handle_fp != turn_fp:
                    if handle is not None:
                        close_live_oneshot_agent(handle)
                        handle = None
                    handle = open_live_oneshot_agent(
                        model=turn_model,
                        provider=turn_provider,
                        toolsets=_normalize_toolsets(turn_toolsets),
                        use_config_toolsets=_normalize_toolsets(turn_toolsets) is None,
                        resume=turn_resume,
                    )
                    handle_fp = turn_fp

                response, result = run_live_oneshot_turn(handle, prompt)
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
            except Exception as exc:
                logger.exception("resident turn failed")
                _write_result({"ok": False, "error": str(exc), "session_id": session_id})
                # Drop a broken agent so the next turn rebuilds cleanly.
                if handle is not None:
                    close_live_oneshot_agent(handle)
                    handle = None
                    handle_fp = None
            finally:
                _restore_env(prior)
                # Drop the run-scoped API key between turns.
                os.environ.pop("PAPERCLIP_API_KEY", None)

            last_activity = time.monotonic()
            agent_model = turn_model or agent_model
            agent_provider = turn_provider or agent_provider
    finally:
        close_live_oneshot_agent(handle)


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
