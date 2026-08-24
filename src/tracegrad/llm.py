"""The model layer — one of exactly two modules allowed to talk to a model.

Everything else in tracegrad is deterministic; ``attribute`` and ``synthesize``
are the only importers of this module, and a lint test enforces that.  Keeping
the boundary mechanical is what lets the rest of the pipeline be replayed,
cached, and reasoned about without a model in the loop.

Two backends ship:

* ``openai`` — any OpenAI-compatible chat completions endpoint, OpenRouter by
  default.  Optional parameters are sent only when explicitly set, because
  gateways reject unknown-but-present fields; reasoning-effort support is
  capability-flagged and retried once without it; an OpenRouter error delivered
  inside an HTTP 200 body is treated as the error it is.
* ``claude`` — a local coding-agent harness the user is already logged into,
  invoked in a deliberately isolated configuration: no tools, no MCP servers, no
  inherited settings.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import httpx

from .config import DEFAULT_ATTRIBUTION_TEMPERATURE, HarnessPreset, TracegradConfig
from .ports import Backend

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ATTRIBUTION_MODEL = "openai/gpt-4.1-mini"
DEFAULT_SYNTHESIS_TIMEOUT_SECONDS = 900.0
DEFAULT_ATTRIBUTION_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3
API_KEY_ENV_VARS = ("TRACEGRAD_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")

ATTRIBUTION_TIER = "attribution"
SYNTHESIS_TIER = "synthesis"


class LLMError(RuntimeError):
    """A backend failure that the caller must handle explicitly."""


class LLMTimeout(LLMError):
    """A request that exceeded its tier timeout.  Synthesis never retries these."""


class LLMUnavailable(LLMError):
    """A backend that cannot run here — missing key, missing binary, disabled."""


@dataclass(frozen=True)
class Completion:
    """One model response plus what it cost and where it came from."""

    text: str
    backend: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: Mapping[str, Any] | None = None

    def json(self) -> Any:
        """Parse the response as JSON, tolerating a fenced code block."""

        return parse_json_response(self.text)


def parse_json_response(text: str) -> Any:
    """Extract the JSON payload from a model response.

    Models fence JSON, prepend prose, or both.  This finds the outermost JSON
    value rather than failing on decoration, but never repairs malformed JSON —
    a broken response should surface as a broken response.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -len("```")]
        stripped = body.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for opening, closing in (("{", "}"), ("[", "]")):
        start = stripped.find(opening)
        end = stripped.rfind(closing)
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("model response did not contain JSON")


# ``Backend`` is imported from ``ports`` and re-exported here: the shape belongs
# to the deterministic side of the boundary, the implementations belong to this
# module.  Callers may import it from either.


@dataclass
class FakeBackend:
    """A scripted backend for tests and for ``--estimate`` dry runs."""

    responses: Sequence[str] = ()
    name: str = "fake"
    handler: Callable[[str, str], str] | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    _index: int = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        cacheable_prefix: str | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Completion:
        # Compose the prefix exactly as a real backend does, so a test double
        # sees the same system text the model would.
        full_system = f"{cacheable_prefix}\n\n{system}" if cacheable_prefix else system
        self.calls.append((full_system, user))
        if self.handler is not None:
            return Completion(self.handler(full_system, user), self.name)
        if not self.responses:
            raise LLMError("FakeBackend has no scripted responses left")
        text = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return Completion(text, self.name)


class OpenAIBackend:
    """An OpenAI-compatible chat-completions backend, OpenRouter by default."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_ATTRIBUTION_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        timeout: float = DEFAULT_ATTRIBUTION_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or _first_env(API_KEY_ENV_VARS)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self._supports_reasoning = reasoning_effort is not None

    @property
    def is_available(self) -> bool:
        """Whether this backend could actually run a request right now."""

        return bool(self.api_key)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _payload(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        *,
        with_reasoning: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Optional parameters are omitted entirely unless set: gateways reject
        # nulls, and a default temperature is not the same as no temperature.
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if with_reasoning and self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "tracegrad", "strict": True, "schema": dict(schema)},
            }
        return payload

    def complete(
        self,
        system: str,
        user: str,
        *,
        cacheable_prefix: str | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Completion:
        if not self.api_key:
            raise LLMUnavailable(
                "no API key: set one of " + ", ".join(API_KEY_ENV_VARS)
            )
        # The cacheable prefix leads the system message so prefix caching can hit.
        full_system = f"{cacheable_prefix}\n\n{system}" if cacheable_prefix else system
        request_timeout = timeout if timeout is not None else self.timeout
        attempt = 0
        with_reasoning = self._supports_reasoning
        while True:
            attempt += 1
            payload = self._payload(full_system, user, schema, with_reasoning=with_reasoning)
            try:
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeout(f"request timed out after {request_timeout}s") from exc
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise LLMError(f"request failed: {exc}") from exc
                self._sleep(_backoff(attempt))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"backend returned {response.status_code}: {response.text[:200]}"
                    )
                self._sleep(_retry_after(response) or _backoff(attempt))
                continue

            if response.status_code == 400 and with_reasoning:
                # The model does not accept a reasoning effort: retry once without.
                with_reasoning = False
                self._supports_reasoning = False
                attempt -= 1
                continue

            if response.status_code >= 400:
                raise LLMError(f"backend returned {response.status_code}: {response.text[:200]}")

            try:
                body = response.json()
            except ValueError as exc:
                raise LLMError("backend returned a non-JSON body") from exc

            # OpenRouter reports upstream failures inside an HTTP 200 body.
            error = body.get("error") if isinstance(body, dict) else None
            if error:
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise LLMError(f"backend reported an error: {message}")

            choices = body.get("choices") or []
            if not choices:
                raise LLMError("backend returned no choices")
            text = (choices[0].get("message") or {}).get("content") or ""
            usage = body.get("usage") or {}
            return Completion(
                text=text,
                backend=self.name,
                model=body.get("model", self.model),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                raw=body,
            )


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _backoff(attempt: int) -> float:
    return min(2.0 ** (attempt - 1), 8.0)


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


class CommandBackend:
    """A local coding-agent harness invoked as a subprocess.

    The ``claude`` preset runs the CLI in an isolated configuration on purpose:
    no tools, no MCP servers, no inherited settings file.  An analysis run must
    not be able to touch the repository it is analysing, and its output must not
    depend on whatever the user has configured locally.
    """

    name = "command"

    CLAUDE_ISOLATED_ARGS: tuple[str, ...] = (
        "claude",
        "-p",
        "--output-format",
        "json",
        "--allowed-tools",
        "",
        "--strict-mcp-config",
        # An empty object is rejected: the CLI validates the shape, and the
        # shape it wants is an mcpServers record that happens to be empty.
        "--mcp-config",
        '{"mcpServers":{}}',
        "--settings",
        "{}",
    )

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str = "command",
        env: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_SYNTHESIS_TIMEOUT_SECONDS,
        system_flag: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.command = tuple(command)
        self.name = name
        self.env = dict(env or {})
        self.timeout = timeout
        self.system_flag = system_flag
        self._runner = runner

    @classmethod
    def claude(cls, **kwargs: Any) -> "CommandBackend":
        """The isolated ``claude -p`` preset."""

        kwargs.setdefault("name", "claude")
        kwargs.setdefault("system_flag", "--append-system-prompt")
        return cls(cls.CLAUDE_ISOLATED_ARGS, **kwargs)

    def complete(
        self,
        system: str,
        user: str,
        *,
        cacheable_prefix: str | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Completion:
        full_system = f"{cacheable_prefix}\n\n{system}" if cacheable_prefix else system
        argv = list(self.command)
        if self.system_flag and full_system:
            argv += [self.system_flag, full_system]
            prompt = user
        else:
            prompt = f"{full_system}\n\n{user}" if full_system else user
        environment = {**os.environ, **self.env}
        try:
            completed = self._runner(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else self.timeout,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise LLMUnavailable(f"harness binary not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMTimeout(f"harness timed out: {argv[0]}") from exc

        if completed.returncode != 0:
            raise LLMError(
                f"harness exited {completed.returncode}: {(completed.stderr or '')[:200]}"
            )
        return Completion(self._extract(completed.stdout), self.name)

    def _extract(self, stdout: str) -> str:
        """Unwrap a harness JSON envelope, honoring its ``is_error`` flag."""

        text = stdout.strip()
        if not text:
            raise LLMError("harness produced no output")
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(envelope, dict):
            return text
        if envelope.get("is_error"):
            raise LLMError(f"harness reported an error: {str(envelope.get('result'))[:200]}")
        result = envelope.get("result")
        if isinstance(result, str):
            return result
        return text


def build_backend(
    preset: HarnessPreset,
    *,
    tier: str = ATTRIBUTION_TIER,
    client: httpx.Client | None = None,
) -> Backend:
    """Instantiate the backend one harness preset describes."""

    if not preset.enabled:
        raise LLMUnavailable(f"harness preset for {tier} is disabled")

    timeout = float(
        preset.timeoutSeconds
        if preset.timeoutSeconds is not None
        else (
            DEFAULT_SYNTHESIS_TIMEOUT_SECONDS
            if tier == SYNTHESIS_TIER
            else DEFAULT_ATTRIBUTION_TIMEOUT_SECONDS
        )
    )

    if preset.provider == "openai":
        # Attribution is a measurement: it samples at temperature 0 unless the
        # user deliberately overrides it, and the instrument records which.
        temperature = preset.temperature
        if temperature is None and tier == ATTRIBUTION_TIER:
            temperature = DEFAULT_ATTRIBUTION_TEMPERATURE
        backend = OpenAIBackend(
            model=preset.model or DEFAULT_ATTRIBUTION_MODEL,
            temperature=temperature,
            reasoning_effort=preset.reasoning_effort,
            timeout=timeout,
            client=client,
        )
        # Report unavailability now, not on the first call: the tier resolver
        # can only fall back to another provider if it learns here.
        if not backend.is_available:
            raise LLMUnavailable("no API key: set one of " + ", ".join(API_KEY_ENV_VARS))
        return backend
    if preset.provider == "claude":
        if shutil.which("claude") is None:
            raise LLMUnavailable("the claude CLI is not installed or not on PATH")
        return CommandBackend.claude(timeout=timeout, env=preset.env)
    if preset.provider == "command":
        if not preset.command:
            raise LLMUnavailable(f"harness preset for {tier} declares no command")
        argv = (
            shlex.split(preset.command)
            if isinstance(preset.command, str)
            else list(preset.command)
        )
        return CommandBackend(argv, name=preset.model or "command", env=preset.env, timeout=timeout)
    raise LLMUnavailable(f"unknown harness provider: {preset.provider}")


DEFAULT_TIER_PROVIDERS: dict[str, tuple[str, ...]] = {
    ATTRIBUTION_TIER: ("openai", "claude"),
    SYNTHESIS_TIER: ("claude", "openai"),
}
"""Per-tier provider preference, tried in order until one is available."""


def resolve_backend(
    config: TracegradConfig,
    tier: str,
    *,
    client: httpx.Client | None = None,
    override: Backend | None = None,
    on_fallback: Callable[[str], None] | None = None,
) -> Backend:
    """Resolve the backend for one tier: explicit override, preset, then defaults.

    The configured preset is tried first.  If it cannot run here — no API key,
    no harness binary — the tier falls through its preference order rather than
    failing the run, because a machine with a logged-in ``claude`` and no
    OpenRouter key should still work.  A fallback is never silent: ``on_fallback``
    is called with what happened, and the instrument fingerprint records the
    backend that actually ran, so a report cannot hide which model measured it.
    """

    if override is not None:
        return override

    preset = config.harness_presets.get(tier)
    attempts: list[HarnessPreset] = [preset] if preset is not None else []
    tried = {preset.provider} if preset is not None else set()
    for provider in DEFAULT_TIER_PROVIDERS.get(tier, ("openai",)):
        if provider not in tried:
            attempts.append(HarnessPreset(provider=provider))
            tried.add(provider)

    failures: list[str] = []
    for index, candidate in enumerate(attempts):
        try:
            backend = build_backend(candidate, tier=tier, client=client)
        except LLMUnavailable as exc:
            failures.append(f"{candidate.provider}: {exc}")
            continue
        if index and on_fallback is not None:
            on_fallback(
                f"{tier}: {attempts[0].provider} is unavailable "
                f"({failures[0]}); using {candidate.provider} instead"
            )
        return backend

    raise LLMUnavailable(f"no backend available for the {tier} tier — " + "; ".join(failures))
