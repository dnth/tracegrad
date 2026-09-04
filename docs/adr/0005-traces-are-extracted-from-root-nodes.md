# 0005. Traces are extracted from root LLM nodes

## Status

Accepted

## Context

A Kitaru session is a DAG of nodes (`parent_index` and
`secondary_parent_indexes`). Node types include `llm_call`, `tool_call`,
`subagent_call`, and `span`. Subagents have their own instructions. Tool nodes
carry tool outputs, not the artifact's response.

`SessionResponse.inputs` / `.outputs` are typed `Any` with no session-level
selector. Stringifying them as a fallback would let a tool payload or a nested
object become `Trace.input` / `Trace.output`.

`Trace.input` is a `StrictStr`, so a multi-turn session cannot be represented
losslessly.

A node reachable from a subagent is not a root LLM node even when one of its
parents is a root LLM node. Following only `parent_index` would miss that.

## Decision

A **root LLM node** is an `llm_call` with no `subagent_call` anywhere in its
ancestry, following `parent_index` **and** `secondary_parent_indexes`.

- `input` is `input_text_selector` resolved on the **first** root LLM node.
- `output` is `output_text_selector` resolved on the **last** root LLM node.
- The system prompt is `system_prompt_selector` resolved against node inputs
  for each root LLM node, uniqueness-checked per ADR 0002.
- `meta.model` comes from `SessionNodeResponse.model` on the root LLM nodes.
- `trace_id` is `SessionResponse.id` (UUID). `number` is carried for display
  (`#4811`); the durable key stays the UUID.

Session-level inputs/outputs are never stringified as a fallback. Tool outputs
cannot become the final output, because only `llm_call` nodes are consulted. A
subagent's system prompt can never become the artifact.

Selector resolution failures drop by name (`input-unavailable`,
`output-unavailable`, `system-prompt-unavailable`). Multi-turn is lossy: a
session collapses to its first root input and last root output. Report it; do
not hide it.

`trace.meta["trajectory"]` is out of scope: `TraceMeta` is `extra="forbid"`
with one field. Kitaru remains the system of record for trajectories.

## Consequences

- Mapping is provider-agnostic: it reads Kitaru's normalized session graph.
- Tests must cover DAG reachability via a secondary parent, not only trees.
- Display reports may print `#4811` while every persisted key stays the UUID.
