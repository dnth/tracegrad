import json
import subprocess
from typing import Any

import httpx
import pytest

from tracegrad import llm
from tracegrad.config import HarnessPreset, TracegradConfig
from tracegrad.llm import (
    ATTRIBUTION_TIER,
    SYNTHESIS_TIER,
    CommandBackend,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    OpenAIBackend,
    build_backend,
    parse_json_response,
    resolve_backend,
)

# --------------------------------------------------------------- parse_json_response


def test_parse_json_response_bare_json() -> None:
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_fenced_json_block() -> None:
    text = '```json\n{"a": 1}\n```'

    assert parse_json_response(text) == {"a": 1}


def test_parse_json_response_prose_wrapped_object() -> None:
    text = 'Sure, here is the result:\n{"a": 1}\nHope that helps.'

    assert parse_json_response(text) == {"a": 1}


def test_parse_json_response_prose_wrapped_array() -> None:
    text = "The values are [1, 2, 3] as requested."

    assert parse_json_response(text) == [1, 2, 3]


def test_parse_json_response_raises_llm_error_when_no_json() -> None:
    with pytest.raises(LLMError):
        parse_json_response("no json here at all")


# --------------------------------------------------------------------- OpenAIBackend


def _client_with_handler(handler: Any) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_openai_backend_omits_optional_params_when_unset() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = OpenAIBackend(api_key="key", client=_client_with_handler(handler))
    backend.complete("system", "user")

    body = captured["body"]
    assert "temperature" not in body
    assert "max_tokens" not in body
    assert "reasoning" not in body


def test_openai_backend_includes_optional_params_when_set() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = OpenAIBackend(
        api_key="key",
        temperature=0.2,
        max_tokens=100,
        reasoning_effort="medium",
        client=_client_with_handler(handler),
    )
    backend.complete("system", "user")

    body = captured["body"]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 100
    assert body["reasoning"] == {"effort": "medium"}


def test_openai_backend_raises_on_openrouter_error_in_200_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "upstream exploded"}})

    backend = OpenAIBackend(api_key="key", client=_client_with_handler(handler))

    with pytest.raises(LLMError, match="upstream exploded"):
        backend.complete("system", "user")


def test_openai_backend_429_honours_retry_after_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    sleeps: list[float] = []
    backend = OpenAIBackend(
        api_key="key", client=_client_with_handler(handler), sleep=sleeps.append
    )

    completion = backend.complete("system", "user")

    assert completion.text == "ok"
    assert sleeps == [3.0]
    assert calls["count"] == 2


def test_openai_backend_5xx_retries_then_raises_after_max_retries() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, text="server error")

    sleeps: list[float] = []
    backend = OpenAIBackend(
        api_key="key",
        client=_client_with_handler(handler),
        sleep=sleeps.append,
        max_retries=3,
    )

    with pytest.raises(LLMError):
        backend.complete("system", "user")

    assert calls["count"] == 3


def test_openai_backend_400_with_reasoning_retries_once_without_reasoning() -> None:
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_bodies.append(body)
        if "reasoning" in body:
            return httpx.Response(400, text="reasoning not supported")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = OpenAIBackend(
        api_key="key",
        reasoning_effort="high",
        client=_client_with_handler(handler),
    )

    completion = backend.complete("system", "user")

    assert completion.text == "ok"
    assert len(seen_bodies) == 2
    assert "reasoning" in seen_bodies[0]
    assert "reasoning" not in seen_bodies[1]


def test_openai_backend_missing_api_key_raises_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TRACEGRAD_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    backend = OpenAIBackend(api_key=None, client=_client_with_handler(lambda r: httpx.Response(200)))

    with pytest.raises(LLMUnavailable):
        backend.complete("system", "user")


def test_openai_backend_timeout_raises_llm_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    backend = OpenAIBackend(api_key="key", client=_client_with_handler(handler))

    with pytest.raises(LLMTimeout):
        backend.complete("system", "user")


def test_openai_backend_usage_tokens_land_on_completion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "model": "openai/gpt-4.1-mini",
                "usage": {"prompt_tokens": 12, "completion_tokens": 34},
            },
        )

    backend = OpenAIBackend(api_key="key", client=_client_with_handler(handler))
    completion = backend.complete("system", "user")

    assert completion.prompt_tokens == 12
    assert completion.completion_tokens == 34
    assert completion.model == "openai/gpt-4.1-mini"


# ------------------------------------------------------------------- CommandBackend


def _completed(stdout: str = "ok", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_command_backend_claude_preset_argv() -> None:
    backend = CommandBackend.claude(runner=lambda *a, **k: _completed())

    assert "--allowed-tools" in backend.command
    index = backend.command.index("--allowed-tools")
    assert backend.command[index + 1] == ""
    assert "--strict-mcp-config" in backend.command
    mcp_index = backend.command.index("--mcp-config")
    assert backend.command[mcp_index + 1] == "{}"
    settings_index = backend.command.index("--settings")
    assert backend.command[settings_index + 1] == "{}"


def test_command_backend_claude_passes_system_via_flag_not_stdin() -> None:
    captured: dict[str, Any] = {}

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["input"] = kwargs["input"]
        return _completed()

    backend = CommandBackend.claude(runner=runner)
    backend.complete("the system prompt", "the user prompt")

    argv = captured["argv"]
    assert "--append-system-prompt" in argv
    flag_index = argv.index("--append-system-prompt")
    assert argv[flag_index + 1] == "the system prompt"
    assert captured["input"] == "the user prompt"
    assert "the system prompt" not in captured["input"]


def test_command_backend_json_envelope_is_error_raises() -> None:
    envelope = json.dumps({"is_error": True, "result": "boom"})
    backend = CommandBackend(["cmd"], runner=lambda *a, **k: _completed(stdout=envelope))

    with pytest.raises(LLMError, match="boom"):
        backend.complete("system", "user")


def test_command_backend_plain_text_stdout_passes_through() -> None:
    backend = CommandBackend(["cmd"], runner=lambda *a, **k: _completed(stdout="plain text output"))

    completion = backend.complete("system", "user")

    assert completion.text == "plain text output"


def test_command_backend_json_envelope_with_result_string_unwraps() -> None:
    envelope = json.dumps({"is_error": False, "result": "the answer"})
    backend = CommandBackend(["cmd"], runner=lambda *a, **k: _completed(stdout=envelope))

    completion = backend.complete("system", "user")

    assert completion.text == "the answer"


def test_command_backend_nonzero_exit_raises_llm_error() -> None:
    backend = CommandBackend(
        ["cmd"], runner=lambda *a, **k: _completed(stdout="", returncode=1)
    )

    with pytest.raises(LLMError):
        backend.complete("system", "user")


def test_command_backend_file_not_found_becomes_llm_unavailable() -> None:
    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no such binary")

    backend = CommandBackend(["missing-binary"], runner=runner)

    with pytest.raises(LLMUnavailable):
        backend.complete("system", "user")


def test_command_backend_timeout_expired_becomes_llm_timeout() -> None:
    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="cmd", timeout=1.0)

    backend = CommandBackend(["cmd"], runner=runner)

    with pytest.raises(LLMTimeout):
        backend.complete("system", "user")


# --------------------------------------------------------- build_backend / resolve_backend


def test_build_backend_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACEGRAD_API_KEY", "test-key")
    preset = HarnessPreset(provider="openai", model="openai/gpt-4.1-mini")

    backend = build_backend(preset, tier=ATTRIBUTION_TIER)

    assert isinstance(backend, OpenAIBackend)
    assert backend.model == "openai/gpt-4.1-mini"


def test_build_backend_openai_is_unavailable_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TRACEGRAD_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(LLMUnavailable, match="no API key"):
        build_backend(HarnessPreset(provider="openai"), tier=ATTRIBUTION_TIER)


def test_attribution_defaults_to_temperature_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Attribution is a measurement; provider-default sampling would make the
    # rates it produces irreproducible.
    monkeypatch.setenv("TRACEGRAD_API_KEY", "test-key")

    attribution = build_backend(HarnessPreset(provider="openai"), tier=ATTRIBUTION_TIER)
    synthesis = build_backend(HarnessPreset(provider="openai"), tier=SYNTHESIS_TIER)
    overridden = build_backend(
        HarnessPreset(provider="openai", temperature=0.7), tier=ATTRIBUTION_TIER
    )

    assert attribution.temperature == 0.0
    assert synthesis.temperature is None
    assert overridden.temperature == 0.7


def test_a_preset_reasoning_effort_reaches_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACEGRAD_API_KEY", "test-key")

    backend = build_backend(
        HarnessPreset(provider="openai", reasoning_effort="high"), tier=ATTRIBUTION_TIER
    )

    assert backend.reasoning_effort == "high"


def test_build_backend_claude_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/bin/claude")
    preset = HarnessPreset(provider="claude")

    backend = build_backend(preset, tier=SYNTHESIS_TIER)

    assert isinstance(backend, CommandBackend)
    assert backend.name == "claude"


def test_build_backend_command_provider() -> None:
    preset = HarnessPreset(provider="command", command="my-harness --flag")

    backend = build_backend(preset, tier=SYNTHESIS_TIER)

    assert isinstance(backend, CommandBackend)
    assert backend.command == ("my-harness", "--flag")


def test_build_backend_disabled_preset_raises_llm_unavailable() -> None:
    preset = HarnessPreset(provider="openai", enabled=False)

    with pytest.raises(LLMUnavailable):
        build_backend(preset, tier=ATTRIBUTION_TIER)


def test_build_backend_unknown_provider_raises() -> None:
    preset = HarnessPreset(provider="unknown-provider")

    with pytest.raises(LLMUnavailable):
        build_backend(preset, tier=ATTRIBUTION_TIER)


def test_resolve_backend_uses_preset_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACEGRAD_API_KEY", "test-key")
    config = TracegradConfig(
        harness_presets={ATTRIBUTION_TIER: HarnessPreset(provider="openai")}
    )

    backend = resolve_backend(config, ATTRIBUTION_TIER)

    assert isinstance(backend, OpenAIBackend)


def test_an_unavailable_preset_falls_through_to_the_next_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TRACEGRAD_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/bin/claude")
    config = TracegradConfig(
        harness_presets={ATTRIBUTION_TIER: HarnessPreset(provider="openai")}
    )
    notices: list[str] = []

    backend = resolve_backend(config, ATTRIBUTION_TIER, on_fallback=notices.append)

    assert isinstance(backend, CommandBackend)
    # A fallback bills a different provider, so it is reported, never silent.
    assert notices and "openai is unavailable" in notices[0]
    assert "using claude instead" in notices[0]


def test_resolution_fails_loudly_when_no_backend_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TRACEGRAD_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)

    with pytest.raises(LLMUnavailable, match="no backend available"):
        resolve_backend(TracegradConfig(), ATTRIBUTION_TIER)


def test_resolve_backend_explicit_override_wins() -> None:
    config = TracegradConfig(
        harness_presets={ATTRIBUTION_TIER: HarnessPreset(provider="openai")}
    )
    override = object()

    backend = resolve_backend(config, ATTRIBUTION_TIER, override=override)  # type: ignore[arg-type]

    assert backend is override
