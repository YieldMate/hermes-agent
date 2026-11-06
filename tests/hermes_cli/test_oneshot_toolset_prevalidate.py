"""YIE-68: -t unknown toolsets fail fast with a clear config error."""

from __future__ import annotations

from hermes_cli.oneshot import _validate_explicit_toolsets


def test_unknown_toolsets_fail_fast_naming_unresolved():
    valid, err = _validate_explicit_toolsets("creative,productivity")
    assert valid is None
    assert err is not None
    assert "toolset" in err.lower()
    assert "not found" in err.lower() or "not " in err.lower()
    assert "creative" in err and "productivity" in err


def test_known_builtin_toolsets_pass():
    valid, err = _validate_explicit_toolsets("terminal,file,web")
    assert err is None
    assert valid == ["terminal", "file", "web"]
