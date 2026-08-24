"""The determinism boundary, enforced mechanically rather than by convention."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tracegrad"
MODEL_AWARE_MODULES = frozenset({"attribute", "synthesize"})


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    return imported


def test_only_attribute_and_synthesize_import_llm() -> None:
    offenders = sorted(
        source.stem
        for source in PACKAGE.glob("*.py")
        if source.stem not in MODEL_AWARE_MODULES and "llm" in _imported_modules(source)
    )

    assert offenders == [], (
        "only attribute and synthesize may import llm; "
        f"these modules break the determinism boundary: {offenders}"
    )


def test_the_model_aware_modules_still_exist() -> None:
    # A boundary test that passes because the modules vanished is not a test.
    for module in MODEL_AWARE_MODULES:
        assert (PACKAGE / f"{module}.py").is_file()
        assert "llm" in _imported_modules(PACKAGE / f"{module}.py")


# Symbols that belong to the model layer.  Re-exporting one of these from
# attribute or synthesize would satisfy the import check while smuggling the
# model layer into a module that is supposed to be deterministic.
LLM_OWNED_SYMBOLS = frozenset(
    {"Backend", "Completion", "FakeBackend", "OpenAIBackend", "CommandBackend"}
)


def _symbols_imported_from(source: Path, modules: frozenset[str]) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[-1] in modules:
                symbols.update(alias.name for alias in node.names)
    return symbols


def test_llm_symbols_are_not_laundered_through_the_model_aware_modules() -> None:
    # The import check above only sees `llm` by name.  Without this, pipeline.py
    # could import Backend from .attribute and defeat the boundary while the
    # letter of the rule held.
    offenders: dict[str, set[str]] = {}
    for source in PACKAGE.glob("*.py"):
        if source.stem in MODEL_AWARE_MODULES:
            continue
        laundered = _symbols_imported_from(source, MODEL_AWARE_MODULES) & LLM_OWNED_SYMBOLS
        if laundered:
            offenders[source.stem] = laundered

    assert offenders == {}, (
        "these modules import model-layer symbols by way of attribute/synthesize, "
        f"which defeats the determinism boundary: {offenders}"
    )


def test_no_module_calls_out_to_the_network_directly() -> None:
    offenders = sorted(
        source.stem
        for source in PACKAGE.glob("*.py")
        if source.stem != "llm" and "httpx" in _imported_modules(source)
    )

    assert offenders == [], f"only llm may speak HTTP; offenders: {offenders}"
