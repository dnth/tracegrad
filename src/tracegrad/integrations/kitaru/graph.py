"""Root LLM node classification over Kitaru's session DAG.

A root LLM node is an ``llm_call`` with no ``subagent_call`` anywhere in its
ancestry, following ``parent_index`` **and** ``secondary_parent_indexes``.  A
node reachable from a subagent is not root even when one of its parents is.
See ADR 0005.

This module does not import the Kitaru SDK; nodes are duck-typed.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

LLM_CALL = "llm_call"
SUBAGENT_CALL = "subagent_call"
TOOL_CALL = "tool_call"


def node_type_value(node: Any) -> str:
    """Return the node type as a plain string, enum or not."""

    raw = getattr(node, "node_type", "")
    value = getattr(raw, "value", raw)
    return "" if value is None else str(value)


def node_index(node: Any) -> int:
    return int(node.index)


def parent_indexes(node: Any) -> tuple[int, ...]:
    """Every parent of ``node``, primary first, then secondary, de-duplicated."""

    parents: list[int] = []
    primary = getattr(node, "parent_index", None)
    if primary is not None:
        parents.append(int(primary))
    secondary = getattr(node, "secondary_parent_indexes", None) or ()
    for item in secondary:
        index = int(item)
        if index not in parents:
            parents.append(index)
    return tuple(parents)


def index_nodes(nodes: Sequence[Any]) -> dict[int, Any]:
    return {node_index(node): node for node in nodes}


def has_subagent_ancestor(node: Any, by_index: dict[int, Any]) -> bool:
    """Whether any ancestor of ``node`` is a ``subagent_call``."""

    seen: set[int] = set()
    stack = list(parent_indexes(node))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        ancestor = by_index.get(current)
        if ancestor is None:
            continue
        if node_type_value(ancestor) == SUBAGENT_CALL:
            return True
        stack.extend(parent_indexes(ancestor))
    return False


def is_root_llm_node(node: Any, by_index: dict[int, Any] | None = None) -> bool:
    """Whether ``node`` is an ``llm_call`` with no subagent in its ancestry."""

    if node_type_value(node) != LLM_CALL:
        return False
    table = by_index if by_index is not None else index_nodes((node,))
    return not has_subagent_ancestor(node, table)


def root_llm_nodes(nodes: Iterable[Any]) -> tuple[Any, ...]:
    """Root LLM nodes in session order (ascending index)."""

    materialised = tuple(nodes)
    by_index = index_nodes(materialised)
    roots = [node for node in materialised if is_root_llm_node(node, by_index)]
    return tuple(sorted(roots, key=node_index))


def llm_nodes(nodes: Iterable[Any]) -> tuple[Any, ...]:
    """Every LLM node, root or not, in session order."""

    return tuple(
        sorted(
            (node for node in nodes if node_type_value(node) == LLM_CALL),
            key=node_index,
        )
    )
