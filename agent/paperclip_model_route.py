"""Opt-in Paperclip per-request model routing via ``HERMES_PAPERCLIP_MODEL_ROUTE``.

The adapter may set that env to a JSON object::

    {
      "primaryModel": "...",
      "cheapModel": "...",
      "maxCheapInputTokens": 12000,
      "cheapToolAllowlist": ["tool_search", "tool_describe", "linear"]
    }

When present and valid, ``apply_paperclip_model_route`` may replace only
``api_kwargs["model"]``. ``agent.model``, the provider client, and the stored
session route stay on the primary so a later ``--resume`` still matches the
adapter's ``-m``.

Cheap stays active while tool traffic is only from allowlisted tools (default:
Hermes discovery + Linear MCP substring matches). Any non-allowlisted tool, an
unresolvable tool name, or exceeding the cheap token cap locks the rest of the
process to the primary model.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV = "HERMES_PAPERCLIP_MODEL_ROUTE"

# Case-insensitive substring patterns. Covers wire names such as
# ``mcp__paperclip_assigned__mcp_app_gallery_linear_…`` and ``mcp.linear-…``.
DEFAULT_CHEAP_TOOL_ALLOWLIST: tuple[str, ...] = (
    "tool_search",
    "tool_describe",
    "linear",
)


def _parse_allowlist(raw: Any) -> list[str]:
    if raw is None:
        return list(DEFAULT_CHEAP_TOOL_ALLOWLIST)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        parsed = [p for p in parts if p]
        return parsed or list(DEFAULT_CHEAP_TOOL_ALLOWLIST)
    if isinstance(raw, (list, tuple)):
        parsed = [str(p).strip() for p in raw if str(p).strip()]
        return parsed or list(DEFAULT_CHEAP_TOOL_ALLOWLIST)
    return list(DEFAULT_CHEAP_TOOL_ALLOWLIST)


def _parse_route_env(raw: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    primary = str(data.get("primaryModel") or "").strip()
    cheap = str(data.get("cheapModel") or "").strip()
    if not primary or not cheap:
        return None
    max_tokens = data.get("maxCheapInputTokens", 12000)
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return None
    if max_tokens < 0:
        return None
    return {
        "primaryModel": primary,
        "cheapModel": cheap,
        "maxCheapInputTokens": max_tokens,
        "cheapToolAllowlist": _parse_allowlist(data.get("cheapToolAllowlist")),
    }


def load_paperclip_model_route() -> Optional[dict[str, Any]]:
    """Parse ``HERMES_PAPERCLIP_MODEL_ROUTE`` or return None when unset/invalid."""
    raw = (os.getenv(_ENV) or "").strip()
    if not raw:
        return None
    return _parse_route_env(raw)


def _request_messages(api_kwargs: Any) -> list:
    if not isinstance(api_kwargs, dict):
        return []
    messages = api_kwargs.get("messages")
    if isinstance(messages, list):
        return messages
    messages = api_kwargs.get("input")
    return messages if isinstance(messages, list) else []


def _tool_call_name(tc: Any) -> str:
    if not isinstance(tc, dict):
        return ""
    fn = tc.get("function")
    if isinstance(fn, dict):
        name = str(fn.get("name") or "").strip()
        if name:
            return name
    return str(tc.get("name") or "").strip()


def _tool_call_id(tc: Any) -> str:
    if not isinstance(tc, dict):
        return ""
    return str(tc.get("id") or tc.get("tool_call_id") or "").strip()


def _messages_have_tools(messages: list) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            return True
        if msg.get("tool_calls"):
            return True
    return False


def _collect_tool_names(messages: list) -> tuple[list[str], bool]:
    """Return ``(names, complete)``.

    ``complete`` is False when any tool call/result has an empty unresolved name
    (fail closed to primary).
    """
    id_to_name: dict[str, str] = {}
    names: list[str] = []
    complete = True

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                name = _tool_call_name(tc)
                call_id = _tool_call_id(tc)
                if call_id and name:
                    id_to_name[call_id] = name
                if name:
                    names.append(name)
                else:
                    complete = False

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = str(msg.get("name") or "").strip()
        if not name:
            call_id = str(msg.get("tool_call_id") or "").strip()
            name = id_to_name.get(call_id, "")
        if name:
            names.append(name)
        else:
            complete = False

    return names, complete


def _name_matches_allowlist(name: str, allowlist: list[str]) -> bool:
    lowered = name.lower()
    for pattern in allowlist:
        if pattern.lower() in lowered:
            return True
    return False


def _tools_are_cheap_safe(messages: list, allowlist: list[str]) -> bool:
    """True when there is no tool traffic, or every tool name matches the allowlist."""
    if not _messages_have_tools(messages):
        return True
    names, complete = _collect_tool_names(messages)
    if not complete:
        return False
    if not names:
        # Tool traffic markers with zero extractable names → fail closed.
        return False
    return all(_name_matches_allowlist(n, allowlist) for n in names)


def _rough_input_tokens(api_kwargs: Any, messages: list) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough

    total = estimate_messages_tokens_rough(messages) if messages else 0
    system = api_kwargs.get("system") if isinstance(api_kwargs, dict) else None
    if isinstance(system, str) and system:
        total += estimate_tokens_rough(system)
    instructions = api_kwargs.get("instructions") if isinstance(api_kwargs, dict) else None
    if isinstance(instructions, str) and instructions:
        total += estimate_tokens_rough(instructions)
    return total


def choose_paperclip_wire_model(
    route: dict[str, Any],
    *,
    messages: list,
    approx_tokens: int,
    locked: bool,
) -> tuple[str, bool]:
    """Return ``(wire_model, lock_primary)`` for one provider request.

    Decision table (tests pin this):
    - locked → primary, stay locked
    - non-allowlisted / unresolved tool traffic → primary, lock
    - approx_tokens >= maxCheapInputTokens → primary, lock
    - else → cheap, unlocked (including allowlisted-only tools)
    """
    primary = route["primaryModel"]
    cheap = route["cheapModel"]
    cap = route["maxCheapInputTokens"]
    allowlist = route.get("cheapToolAllowlist") or list(DEFAULT_CHEAP_TOOL_ALLOWLIST)
    if locked:
        return primary, True
    if not _tools_are_cheap_safe(messages, allowlist):
        return primary, True
    if approx_tokens >= cap:
        return primary, True
    return cheap, False


def apply_paperclip_model_route(agent: Any, api_kwargs: Any) -> None:
    """Possibly rewrite ``api_kwargs["model"]``; leave the request unchanged otherwise.

    Stashes ``agent._paperclip_wire_model`` for usage accounting when a route is active.
    """
    route = load_paperclip_model_route()
    if route is None:
        # Clear any prior wire-model stash so usage falls back to agent.model.
        if hasattr(agent, "_paperclip_wire_model"):
            agent._paperclip_wire_model = None
        return
    if not isinstance(api_kwargs, dict):
        return

    # Side-question / auxiliary forks must stay on the primary session model.
    if getattr(agent, "_paperclip_model_route_bypass", False):
        if hasattr(agent, "_paperclip_wire_model"):
            agent._paperclip_wire_model = None
        return

    agent._paperclip_model_route_active = True
    messages = _request_messages(api_kwargs)
    locked = bool(getattr(agent, "_paperclip_model_route_locked", False))
    try:
        approx = _rough_input_tokens(api_kwargs, messages)
    except Exception:
        # Fail closed: treat the estimate as at-cap so we stick to primary.
        logger.debug("paperclip model route token estimate failed; using primary", exc_info=True)
        approx = route["maxCheapInputTokens"]

    wire, now_locked = choose_paperclip_wire_model(
        route, messages=messages, approx_tokens=approx, locked=locked,
    )
    if now_locked:
        agent._paperclip_model_route_locked = True
    api_kwargs["model"] = wire
    agent._paperclip_wire_model = wire
