"""When a provider omits usage (glm-5.3 / z-ai), Hermes synthesizes estimates (YIE-67)."""
import logging
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(
        api_key="k",
        base_url="https://api.z.ai/api/paas/v4",
        provider="zai",
        api_mode="chat_completions",
        model="glm-5.3",
        session_id="t-est",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )


def test_missing_usage_folds_local_estimate(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    a._cached_system_prompt = "You are a helpful assistant."
    a._current_streamed_assistant_text = "Hello world, this is a reply."
    try:
        from agent import turn_usage
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            outcome = turn_usage.record_response_usage(
                a,
                SimpleNamespace(usage=None),
                messages=[{"role": "user", "content": "hi there friend"}],
                api_call_count=1,
                api_duration=0.4,
                compression_attempts=0,
                max_compression_attempts=3,
            )
        assert outcome.compression_attempts == 0
        assert a.session_api_calls == 1
        assert a.session_input_tokens > 0
        assert a.session_output_tokens > 0
        assert a.session_cost_status == "estimated"
        assert a.session_cost_source == "local_estimate"
        assert isinstance(a._last_response_token_count, int) and a._last_response_token_count > 0
        assert a._last_response_token_count == a.session_output_tokens
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API call #"))
        assert "usage=estimated" in line
        assert "in=?" not in line
        # Compression must still treat the provider as usageless after the estimate fold.
        assert getattr(a.context_compressor, "_provider_omits_usage", False) is True
    finally:
        a.close()


def test_missing_usage_still_unavailable_when_estimate_empty(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    a._cached_system_prompt = ""
    a._current_streamed_assistant_text = ""
    try:
        from agent import turn_usage
        monkeypatch.setattr(turn_usage, "_estimate_missing_provider_usage", lambda *_a, **_k: None)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            turn_usage.record_response_usage(
                a,
                SimpleNamespace(usage=None),
                messages=[],
                api_call_count=1,
                api_duration=0.1,
                compression_attempts=0,
                max_compression_attempts=3,
            )
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API call #"))
        assert "usage=unavailable" in line
        assert a.session_input_tokens == 0
    finally:
        a.close()
