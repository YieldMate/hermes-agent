#!/usr/bin/env python3
"""CI job ``pinned-tools-token-cap``: fail when an agent's pinned MCP tool schemas exceed the budget.

Each ``overlays/<agent>/pinned-tools.json`` manifest (``{version, updatedAt?, tools: [{name,
callCount?, schema?}]}``) lists the tools Hermes keeps visible under hot tool pinning v1. The
runtime drops least-used pins past ``MAX_PINNED_SCHEMA_TOKENS`` (``tools/hot_tool_pinning.py``);
this check stops a manifest from shipping over the cap in the first place, using the same
cl100k_base count on the OpenAI wire shape as ``hermes_cli.tools_config.estimate_schema_tokens``.

    python scripts/ci/pinned_tools_token_cap.py                       # overlays/**/pinned-tools.json
    python scripts/ci/pinned_tools_token_cap.py path/to/overlays      # a directory or a manifest file
    python scripts/ci/pinned_tools_token_cap.py --require-tiktoken    # CI: refuse the chars/4 fallback

Without a path argument it searches ``overlays/`` under the current directory and under
``$HERMES_HOME``. Exit codes: 0 = every agent fits (or nothing found), 1 = at least one agent over
the cap, 2 = unreadable manifest or missing tiktoken under ``--require-tiktoken``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.hot_tool_pinning import MAX_PINNED_SCHEMA_TOKENS

MANIFEST_NAME = "pinned-tools.json"


@dataclass(frozen=True)
class AgentBudget:
    manifest: Path
    agent: str
    tokens: int
    per_tool: dict[str, int]
    unsized: tuple[str, ...]

    @property
    def over_cap(self) -> bool:
        return self.tokens > MAX_PINNED_SCHEMA_TOKENS


def find_manifests(paths: list[str]) -> list[Path]:
    """Manifests under the given files/directories, else ``overlays/`` in CWD and ``$HERMES_HOME``."""
    if paths:
        roots = [Path(p) for p in paths]
    else:
        roots = [Path.cwd() / "overlays"]
        if home := os.environ.get("HERMES_HOME"):
            roots.append(Path(home) / "overlays")
    found: dict[Path, None] = {}
    for root in roots:
        if root.is_file():
            found[root.resolve()] = None
        elif root.is_dir():
            found.update((p.resolve(), None) for p in root.rglob(MANIFEST_NAME))
    return sorted(found)


def _function_schema(entry: dict) -> dict | None:
    schema = entry.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    return schema if isinstance(schema, dict) else None


def measure_manifest(manifest: Path) -> AgentBudget:
    from hermes_cli.tools_config import estimate_schema_tokens

    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    items = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{manifest}: expected a tools list")
    per_tool: dict[str, int] = {}
    unsized: list[str] = []
    for item in items:
        entry = {"name": item} if isinstance(item, str) else item
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"{manifest}: tool entries need a string name")
        schema = _function_schema(entry)
        if schema is None:
            unsized.append(entry["name"])
        else:
            per_tool[entry["name"]] = estimate_schema_tokens(schema)
    return AgentBudget(manifest, manifest.parent.name, sum(per_tool.values()), per_tool, tuple(unsized))


def check_manifests(paths: list[str]) -> list[AgentBudget]:
    return [measure_manifest(manifest) for manifest in find_manifests(paths)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="overlay directories or pinned-tools.json files")
    parser.add_argument("--require-tiktoken", action="store_true",
                        help="fail instead of falling back to a chars/4 estimate")
    args = parser.parse_args(argv)

    if args.require_tiktoken:
        try:
            import tiktoken
            tiktoken.get_encoding("cl100k_base")
        except (ImportError, OSError) as exc:
            print(f"pinned-tools-token-cap: tiktoken cl100k_base unavailable: {exc}", file=sys.stderr)
            return 2
    try:
        budgets = check_manifests(args.paths)
    except (OSError, ValueError) as exc:
        print(f"pinned-tools-token-cap: {exc}", file=sys.stderr)
        return 2
    if not budgets:
        print("pinned-tools-token-cap: no pinned-tools.json manifests found")
        return 0
    failed = False
    for budget in budgets:
        status = "OVER CAP" if budget.over_cap else "ok"
        print(f"{status}: {budget.agent} ({budget.manifest}) {budget.tokens}/{MAX_PINNED_SCHEMA_TOKENS} tokens")
        if budget.over_cap:
            failed = True
            for name, tokens in sorted(budget.per_tool.items(), key=lambda kv: -kv[1]):
                print(f"    {tokens:>6}  {name}")
        for name in budget.unsized:
            print(f"    note: {name} has no schema in the manifest and was not counted", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
