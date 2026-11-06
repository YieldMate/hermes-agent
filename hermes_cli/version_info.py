"""Canonical runtime identity for Hermes.

Resolution order:
1. Install stamp (``install-stamp.json``) — written at build time by
   ``scripts/write_install_stamp.py`` for every packager (Docker, Nix, and
   the desktop app). The stamp is authoritative
   for packaged builds.
2. Live git — for unstamped source/dev installs with a ``.git`` directory.
3. Unknown — no stamp and no git. The provenance is unknown.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hermes_cli.steward import UPDATE_MECHANISMS
from hermes_cli.update_channel import STABLE_TAG_RE


@dataclass(frozen=True)
class VersionInfo:
    base_version: str
    derived_version: str
    distance: int | None
    commit: str | None
    branch: str | None
    source: Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"]
    dirty: bool = False
    commit_date: int | None = None
    distribution: Literal["docker", "nix", "desktop-app"] | None = None

    @property
    def display_version(self) -> str:
        """``<base>+<distance>`` when a release is known, else the derived identity.
        Keep a tagless checkout's ``git.<sha>`` rather than reducing it to unknown.
        """
        if self.base_version == "unknown":
            return self.derived_version
        return _derived_version(self.base_version, self.distance)


def _derived_version(
    base_version: str,
    distance: int | None,
    dirty: bool = False,
    short_commit: str | None = None,
) -> str:
    if distance and distance > 0:
        suffix = f"{distance}.g{short_commit}" if short_commit else str(distance)
        return f"{base_version}+{suffix}{'.dirty' if dirty and short_commit else ''}"
    if dirty and distance is None:
        return f"{base_version}+?"
    return base_version


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3, cwd=str(repo_dir),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return value if result.returncode == 0 and value else None


def _resolve_repo_dir() -> Path | None:
    """Use the executing checkout before a profile's optional clone."""
    repo_dir = Path(__file__).parent.parent.resolve()
    if (repo_dir / ".git").exists():
        return repo_dir
    # The PROCESS home: this is the running code's identity and is cached
    # process-wide, so a profile's context-local override must not pick it.
    from hermes_constants import get_process_hermes_home

    candidate = get_process_hermes_home() / "hermes-agent"
    if (candidate / ".git").exists():
        return candidate
    return None


def _parse_nonnegative(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _calver_release_version(repo_dir: Path) -> tuple[str, int] | None:
    """The version the nearest CalVer release shipped, and the commits since it.

    Releases before semver tags existed are tagged ``vYYYY.M.D`` only; the
    version users actually run is in that tag's pyproject. Without this, a
    checkout past such a release would compare as "unknown" against plugins'
    ``requires_hermes``.
    """
    described = _run_git(repo_dir, "describe", "--tags", "--long", "--match", "v2[0-9][0-9][0-9].*", "HEAD")
    if not described:
        return None
    tag, count, _ = described.rsplit("-", 2)
    distance = _parse_nonnegative(count)
    try:
        project = tomllib.loads(_run_git(repo_dir, "show", f"{tag}:pyproject.toml") or "").get("project", {})
    except tomllib.TOMLDecodeError:
        return None
    version = project.get("version")
    if distance is None or not isinstance(version, str) or not STABLE_TAG_RE.fullmatch(f"v{version}"):
        return None
    return version, distance


# --- Install stamp reader ---------------------------------------------------

def _resolve_stamp_file() -> Path | None:
    """The executing tree's stamp (pm.paths.install_stamp_path owns the location)."""
    from pm.paths import install_stamp_path, repo_root

    p = install_stamp_path(repo_root())
    return p if p.is_file() else None


def _stamp_version_info() -> VersionInfo | None:
    """Read provenance from a build-time install stamp."""
    stamp_file = _resolve_stamp_file()
    if stamp_file is None:
        return None
    try:
        raw = stamp_file.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "commit" not in data:
        return None

    # updateMechanism is required in every stamp. A stamp without it means
    # the writing build lane must be fixed, not tolerated.
    if data.get("updateMechanism") not in UPDATE_MECHANISMS:
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} is missing a valid "
            f"'updateMechanism' (one of {', '.join(UPDATE_MECHANISMS)}). The "
            "build lane that wrote this stamp must pass --update-mechanism to "
            "scripts/write_install_stamp.py (or bake the field directly)."
        )
    # A light artifact ships no Python runtime. Reading a light stamp from
    # a Python process means the artifact was mispackaged.
    if data.get("payload") == "light":
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} marks this artifact as 'light' "
            "(no agent runtime). No Python process can legitimately run from a "
            "light artifact — this build is mispackaged."
        )

    commit = data.get("commit") or None
    if not commit or set(commit) == {"0"}:
        # All-zero placeholder = fallback stamp, not real provenance.
        return None
    stamp_source = str(data.get("source") or "")
    if stamp_source == "git" and (stamp_file.parent / ".git").exists():
        live_commit = _run_git(stamp_file.parent, "rev-parse", "HEAD")
        if live_commit and live_commit != commit:
            return None

    base_version = data.get("baseVersion") or "unknown"
    display_version = data.get("displayVersion") or base_version
    distance = data.get("distance")
    if isinstance(distance, str):
        distance = _parse_nonnegative(distance)

    # ``source`` describes build provenance, while ``distribution`` identifies
    # the package form users installed. Keep both facts intact for support.
    source = (
        cast(Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"], stamp_source)
        if stamp_source in {"commit-build", "ci", "docker", "fallback", "git", "local", "nix"}
        else "build"
    )
    distribution = data.get("distribution")
    if distribution not in {"docker", "nix", "desktop-app"}:
        distribution = None

    commit_date = data.get("commitDate")
    if not isinstance(commit_date, int):
        commit_date = None

    return VersionInfo(
        base_version,
        display_version,
        distance if isinstance(distance, int) else None,
        commit,
        data.get("branch") or None,
        source,
        bool(data.get("dirty")),
        commit_date,
        distribution,
    )


# --- Git provenance (source/dev installs) -----------------------------------


def _git_version_info(repo_dir: Path, *, include_untracked: bool = False) -> VersionInfo:
    commit = _run_git(repo_dir, "rev-parse", "HEAD")
    # A detached HEAD has no branch. Leave the field None: every formatter
    # already prints the commit separately and handles a missing branch.
    branch = _run_git(repo_dir, "branch", "--show-current")
    commit_date_raw = _run_git(repo_dir, "log", "-1", "--format=%ct", "HEAD")
    commit_date: int | None = None
    if commit_date_raw and commit_date_raw.isdigit():
        commit_date = int(commit_date_raw)
    try:
        # -uno: skip the untracked-file scan. This runs on the startup-banner
        # path, and a full working-tree walk costs real time on large or cold
        # checkouts. Same semantics as write_install_stamp.py.
        # The 3 s timeout kills a slow status (a partial clone lazily fetching a tree, a cold or
        # scanned disk), and a status killed while refreshing the index strands .git/index.lock:
        # the update's own merge then dies on "File exists" (#132089). A read-only probe takes
        # no optional lock.
        status_command = ["git", "--no-optional-locks", "status", "--porcelain"]
        if not include_untracked:
            status_command.append("-uno")
        dirty_result = subprocess.run(
            status_command,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=3,
            cwd=str(repo_dir),
            check=False,
        )
        dirty = dirty_result.returncode == 0 and bool((dirty_result.stdout or "").strip())
    except (OSError, subprocess.SubprocessError):
        dirty = False

    tags = _run_git(repo_dir, "tag", "--merged", "HEAD", "--list", "v[0-9]*")
    releases = [
        tag[1:]
        for tag in (tags or "").splitlines()
        if STABLE_TAG_RE.fullmatch(tag)
    ]
    base_version = (
        max(releases, key=lambda value: tuple(int(part) for part in value.split(".")))
        if releases else "unknown"
    )
    distance = _parse_nonnegative(
        _run_git(repo_dir, "rev-list", "--count", f"v{base_version}..HEAD")
    ) if releases else None
    if not releases:
        base_version, distance = _calver_release_version(repo_dir) or ("unknown", None)
    short_commit = _run_git(repo_dir, "rev-parse", "--short=7", "HEAD")
    if base_version == "unknown" and short_commit:
        display_version = f"git.{short_commit}{'.dirty' if dirty else ''}"
    else:
        display_version = _derived_version(base_version, distance, dirty, short_commit)

    return VersionInfo(
        base_version, display_version, distance, commit, branch, "git", dirty, commit_date
    )


# --- Cache + public API -----------------------------------------------------

_cached_version_info: VersionInfo | None = None
# Cross-process memo of the last resolution. Stamp reads are already cheap; the
# disk cache mainly avoids repeating the multi-git-subprocess path on unstamped
# checkouts (and after a rejected source=git stamp) within the TTL (YIE-66).
_DISK_CACHE_TTL_SECONDS = 3600
_DISK_CACHE_SCHEMA = 1


def _disk_cache_path() -> Path | None:
    try:
        from hermes_constants import get_process_hermes_home

        return get_process_hermes_home() / "cache" / "version_info.json"
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _clear_disk_cache() -> None:
    path = _disk_cache_path()
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _reset_version_info_cache() -> None:
    """Drop process + disk memos (tests, stamp writers, ``get_code_identity(refresh=True)``)."""
    global _cached_version_info
    _cached_version_info = None
    _clear_disk_cache()


def _force_live_git() -> bool:
    """Support/diagnostics: ``version.force_git: true`` in config.yaml prefers live git.

    Intentionally a cheap text probe — ``get_version_info`` sits on the
    ``hermes --version`` path and must not load the full config stack. No new
    ``HERMES_*`` env var (behaviour belongs in config.yaml). Only the
    ``version:`` mapping's ``force_git`` key counts (not a same-named key elsewhere).
    """
    try:
        from hermes_constants import get_hermes_home

        text = (get_hermes_home() / "config.yaml").read_text(encoding="utf-8-sig")
    except OSError:
        return False
    in_version = False
    version_indent: int | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if indent == 0 and stripped.startswith("version:"):
            in_version = True
            version_indent = 0
            # Inline map: ``version: {force_git: true}`` — rare; ignore, require nested form.
            continue
        if in_version:
            if version_indent is not None and indent <= version_indent and not stripped.startswith("#"):
                in_version = False
                continue
            key, _, value = stripped.partition(":")
            if key.strip() == "force_git":
                return value.split("#", 1)[0].strip().lower() in {"true", "yes", "1"}
    return False


def _read_disk_cache() -> VersionInfo | None:
    import time

    path = _disk_cache_path()
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("schemaVersion") != _DISK_CACHE_SCHEMA:
        return None
    cached_at = data.get("cachedAt")
    if not isinstance(cached_at, (int, float)):
        return None
    if time.time() - cached_at > _DISK_CACHE_TTL_SECONDS:
        return None
    info = data.get("info")
    if not isinstance(info, dict):
        return None
    source = info.get("source")
    if source not in {"build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"}:
        return None
    return VersionInfo(
        str(info.get("base_version") or "unknown"),
        str(info.get("derived_version") or "unknown"),
        info.get("distance") if isinstance(info.get("distance"), int) else None,
        info.get("commit") if isinstance(info.get("commit"), str) else None,
        info.get("branch") if isinstance(info.get("branch"), str) else None,
        cast(Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"], source),
        bool(info.get("dirty")),
        info.get("commit_date") if isinstance(info.get("commit_date"), int) else None,
        info.get("distribution") if info.get("distribution") in {"docker", "nix", "desktop-app"} else None,
    )


def _write_disk_cache(info: VersionInfo) -> None:
    path = _disk_cache_path()
    if path is None:
        return
    import time

    payload = {
        "schemaVersion": _DISK_CACHE_SCHEMA,
        "cachedAt": time.time(),
        "info": {
            "base_version": info.base_version,
            "derived_version": info.derived_version,
            "distance": info.distance,
            "commit": info.commit,
            "branch": info.branch,
            "source": info.source,
            "dirty": info.dirty,
            "commit_date": info.commit_date,
            "distribution": info.distribution,
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    except OSError:
        pass


def get_version_info() -> VersionInfo:
    """Return cached provenance from install stamp, git, or unknown.

    Resolution order (unless ``version.force_git`` is set for diagnostics):
    process cache → install stamp → disk TTL cache (git/unknown only) → live git → unknown.
    Stamp always wins over disk so a re-stamp or rejected ``source=git`` stamp is
    never masked for up to an hour (YIE-66). Disk memoizes the expensive git path
    across processes when no usable stamp exists.
    """
    global _cached_version_info
    if _cached_version_info is not None:
        return _cached_version_info

    force_git = _force_live_git()
    info: VersionInfo | None = None
    # 1. Install stamp (packaged builds: Docker, Nix; also re-stamped local branches)
    # A malformed stamp (missing/illegal updateMechanism, mispackaged "light"
    # payload) raises RuntimeError — the build lane that wrote it must be
    # fixed, but a crashing CLI is the wrong failure mode for every uncached
    # caller. Degrade past the stamp and let the authoring lane's own tests
    # surface the malformed stamp.
    if not force_git:
        try:
            info = _stamp_version_info()
        except RuntimeError:
            info = None

    # 2. Disk TTL — only when the stamp was absent/rejected; never masks a new stamp.
    if info is None and not force_git:
        disk = _read_disk_cache()
        if disk is not None:
            _cached_version_info = disk
            return disk

    # 3. Live git (source/dev installs, or diagnostics via version.force_git)
    if info is None:
        repo_dir = _resolve_repo_dir()
        if repo_dir is not None:
            info = _git_version_info(repo_dir)

    # 4. Unknown — no stamp, no git
    if info is None:
        info = VersionInfo("unknown", "unknown", None, None, None, "unknown")

    _cached_version_info = info
    # Cache git/unknown resolutions only; stamped identity is already a file on disk.
    if not force_git and info.source in {"git", "unknown"}:
        _write_disk_cache(info)
    return info


def get_code_identity(refresh: bool = False) -> dict:
    """Return the running install's code identity as a flat dict.

    Shape: ``{"sha": full sha | None, "short_sha": str | None, "version":
    base package version | None, "source": str, "commit_date": epoch s | None}`` — what the update
    receipt, runtime inventory, and gateway status stamping consume.
    Backed by :func:`get_version_info` (install stamp first, live git
    second, unknown third), so every consumer shares one resolution
    policy. Returned dicts are fresh copies, never the shared cache.

    ``refresh=True`` drops the per-process cache first — the updater uses
    it to re-read identity after swapping the checkout out from under the
    running process.
    """
    if refresh:
        _reset_version_info_cache()
    info = get_version_info()
    return {
        "sha": info.commit,
        "short_sha": info.commit[:8] if info.commit else None,
        "version": info.base_version,
        "source": info.source,
        "commit_date": info.commit_date,
    }
