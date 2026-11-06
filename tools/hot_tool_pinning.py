"""Hot MCP tool pinning v1 (YIE-70): keep a bounded set of hot tools visible, shrink tool_search /
tool_describe output, and tighten the MCP spillover threshold — all behind ONE flag.

The flag is ``HERMES_HOT_TOOL_PINNING_V1`` (``PAPERCLIP_HOT_TOOL_PINNING_V1`` is accepted so
Paperclip can inject a single variable). Default OFF; every public function degrades to the
pre-flag behaviour when it is off. Pinned names come from ``HERMES_PINNED_TOOLS_FILE`` (a
``pinned-tools.json`` manifest) or ``HERMES_PINNED_TOOLS`` (a JSON array of names).

Telemetry counters are process-wide: ``model_tools.handle_function_call`` is the single recording
seam (bridge ``tool_search`` / ``tool_describe`` calls and every real tool call, including those
unwrapped from ``tool_call``). Callers that dispatch tools without it (inline agent-loop tools)
do not count."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

FLAG_ENV_VARS = ("HERMES_HOT_TOOL_PINNING_V1", "PAPERCLIP_HOT_TOOL_PINNING_V1")
PINNED_TOOLS_FILE_ENV = "HERMES_PINNED_TOOLS_FILE"
PINNED_TOOLS_ENV = "HERMES_PINNED_TOOLS"

MAX_PINNED_TOOLS = 10
MAX_PINNED_SCHEMA_TOKENS = 10_000
HOT_PINNING_MCP_RESULT_SIZE_CHARS = 15_000
SEARCH_RESULT_CAP = 20

_TRUTHY = frozenset({"1", "true", "on", "yes"})
_lock = threading.RLock()
_pinned_override: Optional[frozenset[str]] = None
_pinned_cache: Optional[tuple[tuple, frozenset[str], int]] = None
_describe_cache: dict[str, dict[tuple[int, str], str]] = defaultdict(dict)
_counters = {"tool_search_calls": 0, "tool_describe_calls": 0, "tool_calls_total": 0}


def is_hot_tool_pinning_v1_enabled() -> bool:
    return any((os.environ.get(var) or "").strip().lower() in _TRUTHY for var in FLAG_ENV_VARS)


# --- pinned set -------------------------------------------------------------------------------

def _wire_schema_tokens(schema: dict[str, Any]) -> int:
    from hermes_cli.tools_config import estimate_schema_tokens
    return estimate_schema_tokens(schema)


def _read_manifest_entries() -> list[dict[str, Any]]:
    """Raw entries from ``HERMES_PINNED_TOOLS_FILE`` (preferred) or ``HERMES_PINNED_TOOLS``.
    A bare list of names and the ``{version, tools: [...]}`` manifest are both accepted;
    unreadable input pins nothing (the flag must never break an agent)."""
    path = (os.environ.get(PINNED_TOOLS_FILE_ENV) or "").strip()
    raw_text = (os.environ.get(PINNED_TOOLS_ENV) or "").strip()
    try:
        if path:
            with open(os.path.expanduser(path), encoding="utf-8-sig") as fh:
                data = json.load(fh)
        elif raw_text:
            data = json.loads(raw_text)
        else:
            return []
    except (OSError, ValueError) as exc:
        logger.warning("hot tool pinning: cannot read pinned tools (%s)", exc)
        return []
    items = data.get("tools") if isinstance(data, dict) else data
    entries: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        entry = {"name": item} if isinstance(item, str) else item
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"].strip():
            entries.append(entry)
    return entries


def _entry_schema(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Function schema for a manifest entry: its own ``schema`` (either shape), else the registry's."""
    schema = entry.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    if isinstance(schema, dict):
        return schema
    from tools.registry import registry
    return registry.get_schema(entry["name"].strip())


def _call_count(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("callCount") or 0)
    except (TypeError, ValueError):
        return 0


def select_pinned_entries(entries: Iterable[dict[str, Any]],
                          ) -> tuple[list[str], int]:
    """Apply the caps: at most ``MAX_PINNED_TOOLS`` names, then drop least-used until the wire-shape
    schema tokens fit ``MAX_PINNED_SCHEMA_TOKENS``. Priority is highest ``callCount``, ties keep
    the earlier name (so the lexicographically later name is dropped first). Returns
    ``(names, schema_tokens)``."""
    unique: dict[str, dict[str, Any]] = {}
    for entry in entries:
        unique.setdefault(entry["name"].strip(), entry)
    ranked = sorted(unique.items(), key=lambda kv: (-_call_count(kv[1]), kv[0]))
    if len(ranked) > MAX_PINNED_TOOLS:
        logger.warning("hot tool pinning: %d pinned tools exceed the cap of %d; dropping %s",
                       len(ranked), MAX_PINNED_TOOLS, [n for n, _ in ranked[MAX_PINNED_TOOLS:]])
        ranked = ranked[:MAX_PINNED_TOOLS]
    tokens = {}
    for name, entry in ranked:
        schema = _entry_schema(entry)
        tokens[name] = _wire_schema_tokens(schema) if schema else 0
    total = sum(tokens.values())
    while ranked and total > MAX_PINNED_SCHEMA_TOKENS:
        dropped, _ = ranked.pop()
        total -= tokens[dropped]
        logger.warning("hot tool pinning: pinned schemas exceed %d tokens; dropped least-used %r",
                       MAX_PINNED_SCHEMA_TOKENS, dropped)
    return [name for name, _ in ranked], total


def _manifest_fingerprint() -> tuple:
    path = (os.environ.get(PINNED_TOOLS_FILE_ENV) or "").strip()
    try:
        mtime = os.stat(os.path.expanduser(path)).st_mtime_ns if path else None
    except OSError:
        mtime = None
    return (path, mtime, (os.environ.get(PINNED_TOOLS_ENV) or "").strip(), _generation())


def _resolve_pinned() -> tuple[frozenset[str], int]:
    global _pinned_cache
    with _lock:
        if _pinned_override is not None:
            return _pinned_override, 0
        key = _manifest_fingerprint()
        if _pinned_cache is None or _pinned_cache[0] != key:
            names, tokens = select_pinned_entries(_read_manifest_entries())
            _pinned_cache = (key, frozenset(names), tokens)
        return _pinned_cache[1], _pinned_cache[2]


def pinned_tool_names() -> frozenset[str]:
    return _resolve_pinned()[0] if is_hot_tool_pinning_v1_enabled() else frozenset()


def is_pinned_tool(name: str) -> bool:
    return name in pinned_tool_names()


def pin_cache_fingerprint() -> tuple:
    """Memo key fragment for tool-definition caches: flag + frozen/resolved pin set."""
    if not is_hot_tool_pinning_v1_enabled():
        return (False, frozenset(), 0)
    names, tokens = _resolve_pinned()
    return (True, names, tokens)


def set_pinned_override(names: Optional[Iterable[str]]) -> None:
    """Pin an explicit set (hosts that resolve pins themselves); ``None`` restores manifest loading."""
    global _pinned_override
    with _lock:
        _pinned_override = None if names is None else frozenset(names)


def freeze_pins_from_env() -> frozenset[str]:
    """Resolve the pin set once from env/manifest and lock it for the session (prompt-cache safe).

    Long-lived processes (resident) must not re-read a changing ``HERMES_PINNED_TOOLS_FILE`` mid-session:
    dispatch gates and ``agent.tools`` would otherwise disagree. Call again only when rebuilding the agent.
    """
    global _pinned_override, _pinned_cache
    with _lock:
        _pinned_override = None
        _pinned_cache = None
    if not is_hot_tool_pinning_v1_enabled():
        release_if_disabled()
        return frozenset()
    names = pinned_tool_names()
    set_pinned_override(names)
    return names


# --- tool_describe cache ----------------------------------------------------------------------

def _scope_key() -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key()


def _generation() -> int:
    from tools.registry import registry
    return registry._generation


def get_describe_cache(name: str) -> Optional[str]:
    """Serialized describe payload for ``name`` at the current registry generation, if cached."""
    with _lock:
        return _describe_cache[_scope_key()].get((_generation(), name))


def set_describe_cache(name: str, payload: str) -> None:
    with _lock:
        bucket = _describe_cache[_scope_key()]
        generation = _generation()
        for stale in [k for k in bucket if k[0] != generation]:
            del bucket[stale]
        bucket[(generation, name)] = payload


def describe_cache_size() -> int:
    with _lock:
        return len(_describe_cache[_scope_key()])


def clear_describe_cache() -> None:
    with _lock:
        _describe_cache.clear()


def render_describe_result(tools: dict[str, Any], serialized: dict[str, str], rest: dict[str, Any]) -> str:
    """JSON text byte-identical to ``json.dumps({"tools": tools, **rest})`` that splices the
    pre-serialized entries in ``serialized`` instead of encoding them again."""
    entries = ", ".join(
        f"{json.dumps(name, ensure_ascii=False)}: "
        f"{serialized[name] if name in serialized else json.dumps(value, ensure_ascii=False)}"
        for name, value in tools.items())
    tail = "".join(f", {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                   for k, v in rest.items())
    return f'{{"tools": {{{entries}}}{tail}}}'


# --- tool_search ranking ----------------------------------------------------------------------

def merge_ranked_names(ranked_lists: list[list[str]], cap: int = SEARCH_RESULT_CAP) -> list[str]:
    """Top ``cap`` unique names across per-query rankings, taken rank by rank (every query's best
    hit before any query's second), returned alphabetically so the output is order-stable."""
    seen: dict[str, None] = {}
    depth = max((len(r) for r in ranked_lists), default=0)
    for rank in range(depth):
        for names in ranked_lists:
            if rank < len(names) and len(seen) < cap:
                seen.setdefault(names[rank])
    return sorted(seen)


# --- telemetry --------------------------------------------------------------------------------

def record_tool_call(name: str) -> None:
    """Count one tool invocation (no-op with the flag off). The ``tool_call`` bridge wrapper is not
    counted; the underlying tool is, when it is dispatched."""
    if not is_hot_tool_pinning_v1_enabled() or name == "tool_call":
        return
    with _lock:
        _counters["tool_calls_total"] += 1
        if name == "tool_search":
            _counters["tool_search_calls"] += 1
        elif name == "tool_describe":
            _counters["tool_describe_calls"] += 1


def tool_search_calls() -> int:
    return _counters["tool_search_calls"]


def tool_describe_calls() -> int:
    return _counters["tool_describe_calls"]


def tool_calls_total() -> int:
    return _counters["tool_calls_total"]


def pinned_schema_tokens() -> int:
    return _resolve_pinned()[1] if is_hot_tool_pinning_v1_enabled() else 0


def reset_hot_pinning_telemetry() -> None:
    with _lock:
        for key in _counters:
            _counters[key] = 0


def snapshot_hot_pinning_telemetry() -> dict[str, int]:
    with _lock:
        return {**_counters, "pinned_schema_tokens": pinned_schema_tokens()}


# --- lifecycle --------------------------------------------------------------------------------

def clear_hot_pinning_runtime_state() -> None:
    """Drop the describe cache, the pinned set (override and loaded manifest) and the counters."""
    global _pinned_override, _pinned_cache
    with _lock:
        _pinned_override = None
        _pinned_cache = None
        _describe_cache.clear()
        reset_hot_pinning_telemetry()


def release_if_disabled() -> None:
    """Flag-off cleanup: a process whose flag was switched off keeps no pin, cache or counter."""
    if is_hot_tool_pinning_v1_enabled():
        return
    with _lock:
        if (_pinned_override is not None or _pinned_cache is not None
                or any(_describe_cache.values()) or any(_counters.values())):
            clear_hot_pinning_runtime_state()


__all__ = [
    "HOT_PINNING_MCP_RESULT_SIZE_CHARS", "MAX_PINNED_SCHEMA_TOKENS", "MAX_PINNED_TOOLS", "SEARCH_RESULT_CAP",
    "clear_describe_cache", "clear_hot_pinning_runtime_state", "describe_cache_size", "freeze_pins_from_env",
    "get_describe_cache", "is_hot_tool_pinning_v1_enabled", "is_pinned_tool", "merge_ranked_names",
    "pin_cache_fingerprint", "pinned_schema_tokens", "pinned_tool_names", "record_tool_call",
    "release_if_disabled", "render_describe_result", "reset_hot_pinning_telemetry", "select_pinned_entries",
    "set_describe_cache", "set_pinned_override", "snapshot_hot_pinning_telemetry", "tool_calls_total",
    "tool_describe_calls", "tool_search_calls",
]
