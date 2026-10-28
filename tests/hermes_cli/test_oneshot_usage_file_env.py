"""HERMES_USAGE_FILE is honored when --usage-file is not passed (Paperclip in-process)."""

from __future__ import annotations

import json


def test_run_oneshot_reads_hermes_usage_file_env(tmp_path, monkeypatch):
    usage_path = tmp_path / "usage.json"
    monkeypatch.setenv("HERMES_USAGE_FILE", str(usage_path))
    monkeypatch.delenv("HERMES_INPROCESS_CHAT", raising=False)

    captured = {}

    def fake_run_agent(prompt, **kwargs):
        captured["ledger"] = kwargs.get("ledger")
        return ("ok", {
            "final_response": "ok",
            "model": "primary-model",
            "provider": "openrouter",
            "input_tokens": 10,
            "output_tokens": 2,
            "estimated_cost_usd": 0.01,
            "completed": True,
            "session_id": "sess-1",
            "usage_by_model": [
                {
                    "model": "cheap-model",
                    "input_tokens": 4,
                    "output_tokens": 1,
                    "estimated_cost_usd": 0.002,
                },
                {
                    "model": "primary-model",
                    "input_tokens": 6,
                    "output_tokens": 1,
                    "estimated_cost_usd": 0.008,
                },
            ],
        })

    monkeypatch.setattr("hermes_cli.oneshot._run_agent", fake_run_agent)
    monkeypatch.setattr("hermes_cli.oneshot._validate_explicit_toolsets", lambda toolsets: (None, None))
    monkeypatch.setattr("hermes_cli.oneshot._normalize_toolsets", lambda toolsets: None)
    monkeypatch.setattr("hermes_cli.oneshot.declare_stateless_channel", lambda: None)
    monkeypatch.setattr("hermes_cli.oneshot._oneshot_exit_code", lambda response, result: 0)

    from hermes_cli.oneshot import run_oneshot

    rc = run_oneshot("hello")
    assert rc == 0
    assert captured["ledger"] is True
    report = json.loads(usage_path.read_text(encoding="utf-8"))
    assert report["model"] == "primary-model"
    assert report["usage_by_model"][0]["model"] == "cheap-model"
    assert report["failed"] is False
