from pathlib import Path

import pytest

from tracegrad.attribute import (
    AttributionCache,
    AttributionError,
    CoverageError,
    HealthSample,
    Instrument,
    attribute_batch,
    attribute_trace,
    build_instrument,
)
from tracegrad.distill import DistillConfig, distill_trace
from tracegrad.inventory import build_inventory
from tracegrad.llm import FakeBackend, LLMError
from tracegrad.schema import AttributionResult, Trace

PROMPT = "1. Always cite sources.\n2. Never speculate.\n"


def _inventory():
    return build_inventory(PROMPT)


def _trace(trace_id: str = "trace-1", output: str = "I made this up entirely.") -> Trace:
    return Trace(
        trace_id=trace_id,
        input="What is the capital of France?",
        output=output,
        judge={"score": 0.2, "rationale": "hallucinated, no citation"},
        prompt_hash="sha256:abc",
    )


def _distilled(trace: Trace):
    return distill_trace(trace, DistillConfig())


# --------------------------------------------------------------------- Instrument


def _base_instrument() -> Instrument:
    return Instrument(
        backend="openai",
        model="openai/gpt-4.1-mini",
        prompt_version=1,
        segmenter_version=2,
        normalizer_version=1,
        distill_config_hash="sha256:distill",
        inventory_hash="sha256:inventory",
    )


def test_instrument_fingerprint_stable_for_identical_instruments() -> None:
    a = _base_instrument()
    b = _base_instrument()

    assert a.fingerprint == b.fingerprint


@pytest.mark.parametrize(
    "field, value",
    [
        ("backend", "claude"),
        ("model", "other-model"),
        ("prompt_version", 2),
        ("segmenter_version", 3),
        ("normalizer_version", 2),
        ("distill_config_hash", "sha256:different"),
        ("inventory_hash", "sha256:different-inventory"),
    ],
)
def test_instrument_fingerprint_changes_with_any_component(field: str, value: object) -> None:
    from dataclasses import replace

    base = _base_instrument()
    changed = replace(base, **{field: value})

    assert base.fingerprint != changed.fingerprint


def test_instrument_cache_key_depends_on_instrument_and_trace() -> None:
    instrument = _base_instrument()
    distilled_one = _distilled(_trace("trace-1"))
    distilled_two = _distilled(_trace("trace-2", output="A different answer."))

    key_one = instrument.cache_key(distilled_one)
    key_two = instrument.cache_key(distilled_two)
    other_instrument_key = replace_instrument(instrument).cache_key(distilled_one)

    assert key_one != key_two
    assert key_one != other_instrument_key


def replace_instrument(instrument: Instrument) -> Instrument:
    from dataclasses import replace

    return replace(instrument, model="a-different-model")


# ----------------------------------------------------------------- AttributionCache


def test_attribution_cache_round_trip(tmp_path: Path) -> None:
    cache = AttributionCache(tmp_path)
    result = AttributionResult(
        trace_id="trace-1",
        violations=[
            {
                "instruction_id": "i-1",
                "theme_slug": "no-citation",
                "quote": "I made this up",
                "quote_source": "output",
            }
        ],
    )

    cache.put("cache:key-1", result)
    fetched = cache.get("cache:key-1")

    assert fetched == result


def test_attribution_cache_hit_avoids_second_backend_call(tmp_path: Path) -> None:
    inventory = _inventory()
    trace = _trace()
    distilled = (_distilled(trace),)
    response = (
        '{"violations": [{"instruction_id": "1", "theme_slug": "no-citation", '
        '"quote": "made this up", "quote_source": "output"}], "harmful": []}'
    )
    backend = FakeBackend(responses=[response])

    run_one = attribute_batch(
        distilled, inventory, backend, project_root=tmp_path, health_sample=0
    )
    assert len(backend.calls) == 1

    run_two = attribute_batch(
        distilled, inventory, backend, project_root=tmp_path, health_sample=0
    )

    assert len(backend.calls) == 1
    assert run_two.cache_hits == 1
    assert run_one.results[0] == run_two.results[0]


# --------------------------------------------------------------------- _parse_result


def test_attribute_trace_forces_violation_quote_source_to_output() -> None:
    inventory = _inventory()
    trace = _trace()
    distilled = _distilled(trace)
    response = (
        '{"violations": [{"instruction_id": "1", "theme_slug": "no-citation", '
        '"quote": "made this up", "quote_source": "distilled"}], "harmful": []}'
    )
    backend = FakeBackend(responses=[response])

    result = attribute_trace(distilled, inventory, backend)

    assert result.violations[0].quote_source.value == "output"


def test_attribute_trace_skips_entries_missing_quote_or_theme() -> None:
    inventory = _inventory()
    trace = _trace()
    distilled = _distilled(trace)
    response = (
        '{"violations": ['
        '{"instruction_id": "1", "theme_slug": "no-citation", "quote": "made this up", '
        '"quote_source": "output"},'
        '{"instruction_id": "1", "theme_slug": "missing-quote", "quote": "", "quote_source": "output"},'
        '{"instruction_id": "1", "quote": "no theme here", "quote_source": "output"}'
        '], "harmful": []}'
    )
    backend = FakeBackend(responses=[response])

    result = attribute_trace(distilled, inventory, backend)

    assert len(result.violations) == 1
    assert result.violations[0].theme_slug == "no-citation"


def test_attribute_trace_raises_on_non_object_response() -> None:
    inventory = _inventory()
    trace = _trace()
    distilled = _distilled(trace)
    backend = FakeBackend(responses=["[1, 2, 3]"])

    with pytest.raises(AttributionError):
        attribute_trace(distilled, inventory, backend)


# ------------------------------------------------------------------- vocabulary feed-forward


def test_theme_vocabulary_is_fed_forward_to_second_call() -> None:
    inventory = _inventory()
    trace_one = _trace("trace-1")
    trace_two = _trace("trace-2", output="Another fabricated answer.")
    distilled = (_distilled(trace_one), _distilled(trace_two))
    response_one = (
        '{"violations": [{"instruction_id": "1", "theme_slug": "no-citation", '
        '"quote": "made this up", "quote_source": "output"}], "harmful": []}'
    )
    response_two = '{"violations": [], "harmful": []}'
    backend = FakeBackend(responses=[response_one, response_two])

    attribute_batch(distilled, inventory, backend)

    second_call_user_message = backend.calls[1][1]
    assert "no-citation" in second_call_user_message


# ------------------------------------------------------------------------- attribute_batch


def test_attribute_batch_raises_coverage_error_below_floor() -> None:
    inventory = _inventory()
    traces = [_trace(f"trace-{i}") for i in range(5)]
    distilled = tuple(_distilled(trace) for trace in traces)
    # Only enough responses for 2 of 5 traces to succeed; the rest raise.
    backend = FakeBackend(handler=_raise_after(2))

    with pytest.raises(CoverageError):
        attribute_batch(distilled, inventory, backend, min_coverage=0.8)


def _raise_after(successes: int):
    state = {"count": 0}

    def handler(system: str, user: str) -> str:
        state["count"] += 1
        if state["count"] > successes:
            raise LLMError("simulated backend failure")
        return '{"violations": [], "harmful": []}'

    return handler


def test_attribute_batch_succeeds_at_coverage_floor() -> None:
    inventory = _inventory()
    traces = [_trace(f"trace-{i}") for i in range(5)]
    distilled = tuple(_distilled(trace) for trace in traces)
    # 4 of 5 succeed: coverage exactly 0.8, the floor.
    backend = FakeBackend(handler=_raise_after(4))

    run = attribute_batch(distilled, inventory, backend, min_coverage=0.8)

    assert run.coverage == pytest.approx(0.8)
    assert len(run.failures) == 1


# ------------------------------------------------------------------------- HealthSample


def test_health_sample_agreement_rate_none_with_no_sample() -> None:
    sample = HealthSample()

    assert sample.agreement_rate is None


def test_health_sample_agreement_rate_with_one_sample() -> None:
    sample = HealthSample(sampled=4, agreed=3)

    assert sample.agreement_rate == pytest.approx(0.75)


def test_build_instrument_reads_backend_and_inventory() -> None:
    inventory = _inventory()
    backend = FakeBackend(name="fake")

    instrument = build_instrument(backend, inventory, "sha256:distill-config")

    assert instrument.backend == "fake"
    assert instrument.segmenter_version == inventory.segmenter_version
    assert instrument.normalizer_version == inventory.normalizer_version
    assert instrument.distill_config_hash == "sha256:distill-config"


def test_concurrent_attribution_preserves_batch_order_and_results(tmp_path: Path) -> None:
    import json as _json

    traces = [
        _distilled(_trace(f"t-{index}", output=f"answer {index}")) for index in range(9)
    ]
    inventory = build_inventory("- Be concise.\n- Cite the doc.\n")

    def respond(system: str, user: str) -> str:
        return _json.dumps({"violations": [], "harmful": []})

    sequential = attribute_batch(
        traces, inventory, FakeBackend(handler=respond), health_sample=0, jobs=1
    )
    concurrent = attribute_batch(
        traces, inventory, FakeBackend(handler=respond), health_sample=0, jobs=4
    )

    assert [item.trace_id for item in concurrent.attributions] == [
        item.trace_id for item in sequential.attributions
    ]
    assert concurrent.coverage == 1.0


def test_the_instrument_records_the_sampling_it_measured_with() -> None:
    inventory = build_inventory("- Be concise.\n")

    class _Sampled(FakeBackend):
        temperature = 0.0
        reasoning_effort = None

    hot = _Sampled(responses=["{}"])
    hot.temperature = 0.7

    cold = build_instrument(_Sampled(responses=["{}"]), inventory, "config")
    warm = build_instrument(hot, inventory, "config")

    assert cold.temperature == 0.0
    assert warm.temperature == 0.7
    # Sampling changes the measurement, so it must change the cache key.
    assert cold.fingerprint != warm.fingerprint
