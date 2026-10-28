"""Opt-in Paperclip per-request model routing via ``HERMES_PAPERCLIP_MODEL_ROUTE``.

The adapter may set that env to a JSON object::

    {"primaryModel": "...", "cheapModel": "...", "maxCheapInputTokens": 12000}

When present and valid, ``apply_paperclip_model_route`` may replace only
``api_kwargs["model"]``. ``agent.model``, the provider client, and the stored
session route stay on the primary so a later ``--resume`` still matches the
adapter's ``-m``. Once a request has tool traffic or exceeds the cheap token
cap, the rest of the process sticks to the primary model.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV = "HERMES_PAPERCLIP_MODEL_ROUTE"


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


def _messages_have_tools(messages: list) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            return True
        if msg.get("tool_calls"):
            return True
    return False


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
    - tool message / assistant tool_calls → primary, lock
    - approx_tokens >= maxCheapInputTokens → primary, lock
    - else → cheap, unlocked
    """
    primary = route["primaryModel"]
    cheap = route["cheapModel"]
    cap = route["maxCheapInputTokens"]
    if locked:
        return primary, True
    if _messages_have_tools(messages):
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

    agent._paperclip_model_route_active = True
    messages = _request_messages(api_kwargs)
    locked = bool(getattr(agent, "_paperclip_model_route_locked", False))
    try:
        approx = _rough_input_tokens(api_kwargs, messages)
    except Exception:
        logger.debug("paperclip model route token estimate failed", exc_info=True)
        approx = 0

    wire, now_locked = choose_paperclip_wire_model(
        route, messages=messages, approx_tokens=approx, locked=locked,
    )
    if now_locked:
        agent._paperclip_model_route_locked = True
    api_kwargs["model"] = wire
    agent._paperclip_wire_model = wire
