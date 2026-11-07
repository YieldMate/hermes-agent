import json
import os
from pathlib import Path
import subprocess

import pytest

from hermes_cli.version_info import (
    VersionInfo,
    _derived_version,
    _git_version_info,
    _reset_version_info_cache,
    _resolve_stamp_file,
    _stamp_version_info,
    get_version_info,
)


def setup_function():
    _reset_version_info_cache()


def test_derived_version_shows_plus_question_for_dirty_unknown_distance():
    assert _derived_version("0.19.0", None, dirty=True) == "0.19.0+?"
    assert _derived_version("0.19.0", None, dirty=False) == "0.19.0"
    assert _derived_version("0.19.0", 5, dirty=True) == "0.19.0+5"
    assert _derived_version("0.19.0", 0, dirty=True) == "0.19.0"


def test_display_version_names_the_distance_without_the_commit():
    """Labels say how far past the release an install is; the commit is shown
    beside them where there is room (`hermes --version` keeps `.g<sha>`)."""
    ahead = VersionInfo("0.21.5", "0.21.5+1913.gf83a9e9", 1913, "f83a9e9" + "0" * 33, "main", "git")
    assert ahead.display_version == "0.21.5+1913"
    assert VersionInfo("0.21.5", "0.21.5", 0, None, None, "git").display_version == "0.21.5"


def test_display_version_keeps_tagless_stamp_identity(tmp_path, monkeypatch):
    stamp = {
        "commit": "a" * 40,
        "baseVersion": "unknown",
        "displayVersion": "git.aaaaaaa",
        "source": "git",
        "payload": "bootstrap",
        "updateMechanism": "self",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info.base_version == "unknown"
    assert info.display_version == stamp["displayVersion"]
    assert info.commit == stamp["commit"]


def test_display_version_keeps_unknown_when_no_identity():
    info = VersionInfo("unknown", "unknown", None, None, None, "unknown")
    assert info.display_version == "unknown"


def test_stamp_version_info_reads_nix_stamp(tmp_path, monkeypatch):
    stamp = {
        "schemaVersion": 2,
        "commit": "a" * 40,
        "branch": "feature/version",
        "baseVersion": "0.19.0",
        "displayVersion": "0.19.0+3",
        "distance": 3,
        "dirty": False,
        "source": "nix",
        "distribution": "nix",
        "updateMechanism": "external",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+3", 3, "a" * 40, "feature/version", "nix", distribution="nix")


def test_stamp_version_info_preserves_ci_provenance_and_docker_distribution(tmp_path, monkeypatch):
    stamp = {"commit": "d" * 40, "source": "ci", "distribution": "docker", "updateMechanism": "external"}
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info.source == "ci"
    assert info.distribution == "docker"


def test_stamp_version_info_preserves_missing_branch(tmp_path, monkeypatch):
    stamp = {
        "schemaVersion": 2,
        "commit": "b" * 40,
        "branch": None,
        "baseVersion": "0.19.0",
        "displayVersion": "0.19.0+?",
        "distance": None,
        "dirty": True,
        "source": "docker",
        "updateMechanism": "external",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+?", None, "b" * 40, None, "docker", True)


def test_stamp_version_info_ignores_fallback_commit(tmp_path, monkeypatch):
    """All-zero commit means the stamp has no real SHA — skip it."""
    stamp = {
        "schemaVersion": 2,
        "commit": "0" * 40,
        "branch": "main",
        "source": "fallback",
        "updateMechanism": "self",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.source == "unknown"
    assert info.commit is None


def test_stamp_version_info_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    assert _stamp_version_info() is None


def test_get_version_info_unknown_when_no_stamp_and_no_git(monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.base_version == "unknown"
    assert info.derived_version == "unknown"
    assert info.distance is None
    assert info.commit is None
    assert info.source == "unknown"


def test_get_version_info_derives_identity_from_reachable_release_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": __import__("os").environ["PATH"]},
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    git("tag", "v0.21.4")
    git("tag", "v2026.9.21")
    (repo / "tracked").write_text("next\n", encoding="utf-8")
    git("commit", "-qam", "next")

    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    info = get_version_info()

    assert info.base_version == "0.21.4"
    assert info.derived_version == f"0.21.4+1.g{git('rev-parse', '--short=7', 'HEAD')}"
    assert info.distance == 1
    assert info.commit == git("rev-parse", "HEAD")
    assert info.source == "git"


def test_get_version_info_takes_the_version_a_calver_only_release_shipped(tmp_path, monkeypatch):
    """Releases tagged only vYYYY.M.D resolve to their pyproject version, not "unknown"."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": __import__("os").environ["PATH"]},
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "0.21.4"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "-qm", "release")
    git("tag", "v2026.9.21")
    (repo / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "0.0.0"\n', encoding="utf-8")
    git("commit", "-qam", "next")

    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    info = get_version_info()

    assert info.base_version == "0.21.4"
    assert info.distance == 1
    assert info.derived_version == f"0.21.4+1.g{git('rev-parse', '--short=7', 'HEAD')}"


def test_resolve_stamp_file_honors_install_root(tmp_path, monkeypatch):
    """Sealed installs (the Nix wrapper) point HERMES_INSTALL_ROOT at the stamp dir."""
    stamp = {"commit": "e" * 40, "source": "nix", "distribution": "nix", "updateMechanism": "external"}
    (tmp_path / "install-stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))

    assert _resolve_stamp_file() == tmp_path / "install-stamp.json"

    info = get_version_info()
    assert info.commit == "e" * 40
    assert info.source == "nix"
    assert info.distribution == "nix"


def test_resolve_stamp_file_install_root_without_stamp_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))
    assert _resolve_stamp_file() is None


def test_resolve_stamp_file_falls_back_to_code_root_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    stamp = {"commit": "f" * 40, "source": "docker", "updateMechanism": "external"}
    (tmp_path / "install-stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setattr("pm.paths.repo_root", lambda: tmp_path)

    assert _resolve_stamp_file() == tmp_path / "install-stamp.json"


def test_old_updater_version_stub_reads_the_same_stamp_as_version_info(tmp_path):
    """``hermes_cli.__version__`` exists only for shipped updaters that import it after the
    checkout swap (tests/compat/old_updater_surface.json). It must report the stamp's base
    version exactly as get_version_info() does, and the pre-stamp placeholder without one.
    A fresh interpreter, since the stub is evaluated when the package is imported."""
    import os
    import sys

    repo = Path(__file__).resolve().parents[2]
    probe = (
        f"import sys; sys.path.insert(0, {str(repo)!r}); import hermes_cli; "
        "from hermes_cli.version_info import get_version_info; "
        "print(hermes_cli.__version__, get_version_info().base_version)"
    )

    def read(install_root: Path) -> list[str]:
        env = {**os.environ, "HERMES_INSTALL_ROOT": str(install_root)}
        return subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
        ).stdout.split()

    stamped = tmp_path / "stamped"
    stamped.mkdir()
    stamp = {"commit": "c" * 40, "baseVersion": "9.8.7", "source": "ci", "updateMechanism": "external"}
    (stamped / "install-stamp.json").write_text(json.dumps(stamp))
    compat, identity = read(stamped)
    assert compat == identity == stamp["baseVersion"]

    unstamped = tmp_path / "unstamped"
    unstamped.mkdir()
    assert read(unstamped)[0] == "0.0.0"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="the blocking untracked scan uses a FIFO")
def test_a_timed_out_status_probe_never_strands_index_lock(tmp_path):
    """The update opens its receipt with this probe; killed by its 3 s timeout while refreshing
    the index, it used to leave .git/index.lock for the same run's merge to die on (#132089)."""
    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-qm", "a")
    (tmp_path / "junk").mkdir()
    os.mkfifo(tmp_path / "junk" / ".gitignore")  # the untracked scan blocks reading it
    (tmp_path / "junk" / "x").touch()
    (tmp_path / "a.txt").touch()  # stat-dirty: a locking status would refresh the index

    _git_version_info(tmp_path, include_untracked=True)

    assert not (tmp_path / ".git" / "index.lock").exists()


def test_local_stamp_skips_rev_parse_guard_when_head_differs(tmp_path, monkeypatch):
    """A source=local (or other non-git) stamp is accepted without comparing HEAD — the
    re-stamp path for local branch installs (YIE-66)."""
    stamp = {
        "commit": "a" * 40,
        "branch": "hermes-wake-speed",
        "baseVersion": "unknown",
        "displayVersion": "git.aaaaaaa",
        "source": "local",
        "updateMechanism": "self",
        "payload": "bootstrap",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr(
        "hermes_cli.version_info._run_git",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("git must not run for source=local")),
    )
    monkeypatch.setattr("hermes_cli.version_info._force_live_git", lambda: False)
    monkeypatch.setattr("hermes_cli.version_info._read_disk_cache", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._write_disk_cache", lambda info: None)

    info = get_version_info()

    assert info.commit == "a" * 40
    assert info.source == "local"
    assert info.branch == "hermes-wake-speed"


def test_git_stamp_rejected_when_live_head_differs(tmp_path, monkeypatch):
    stamp = {
        "commit": "a" * 40,
        "source": "git",
        "updateMechanism": "self",
        "payload": "bootstrap",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr("hermes_cli.version_info._run_git", lambda *a, **k: "b" * 40)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._force_live_git", lambda: False)
    monkeypatch.setattr("hermes_cli.version_info._read_disk_cache", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._write_disk_cache", lambda info: None)

    info = get_version_info()

    assert info.source == "unknown"
    assert info.commit is None


def test_disk_cache_used_only_when_stamp_absent(tmp_path, monkeypatch):
    """Stamp always wins over disk; disk only memoizes git/unknown after a stamp miss."""
    from hermes_cli import version_info as vi

    monkeypatch.setattr(vi, "_disk_cache_path", lambda: tmp_path / "version_info.json")
    monkeypatch.setattr(vi, "_force_live_git", lambda: False)
    monkeypatch.setattr(vi, "_resolve_repo_dir", lambda: None)

    # Seed disk with a git-shaped memo for commit B.
    vi._write_disk_cache(VersionInfo("1.0.0", "1.0.0", 0, "b" * 40, "main", "git"))

    stamp = {
        "commit": "a" * 40,
        "baseVersion": "2.0.0",
        "displayVersion": "2.0.0",
        "source": "local",
        "updateMechanism": "self",
        "payload": "bootstrap",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    monkeypatch.setattr(vi, "_resolve_stamp_file", lambda: stamp_file)

    stamped = get_version_info()
    assert stamped.commit == "a" * 40
    assert stamped.source == "local"

    _reset_version_info_cache()
    monkeypatch.setattr(vi, "_resolve_stamp_file", lambda: None)
    # After reset, disk was cleared; with no stamp/git → unknown (not stale B).
    assert get_version_info().source == "unknown"


def test_git_resolution_is_memoized_on_disk(tmp_path, monkeypatch):
    from hermes_cli import version_info as vi

    monkeypatch.setattr(vi, "_disk_cache_path", lambda: tmp_path / "version_info.json")
    monkeypatch.setattr(vi, "_force_live_git", lambda: False)
    monkeypatch.setattr(vi, "_resolve_stamp_file", lambda: None)
    monkeypatch.setattr(
        vi, "_git_version_info",
        lambda repo, **kw: VersionInfo("9.9.9", "9.9.9", 0, "d" * 40, "main", "git"),
    )
    monkeypatch.setattr(vi, "_resolve_repo_dir", lambda: tmp_path)

    first = get_version_info()
    assert first.commit == "d" * 40
    # Process cache clear only — leave disk so the next call proves the memo.
    vi._cached_version_info = None
    monkeypatch.setattr(
        vi, "_git_version_info",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("git must not run; disk should hit")),
    )
    second = get_version_info()
    assert second == first


def test_force_git_only_reads_version_mapping(tmp_path, monkeypatch):
    from hermes_cli import version_info as vi

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "other:\n  force_git: true\nversion:\n  force_git: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    assert vi._force_live_git() is False

    (home / "config.yaml").write_text("version:\n  force_git: true\n", encoding="utf-8")
    assert vi._force_live_git() is True
