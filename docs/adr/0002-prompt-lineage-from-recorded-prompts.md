# 0002. Prompt lineage comes from recorded system prompts

## Status

Accepted

## Context

tracegrad's ingest partitions a batch on `prompt_hash`. Rates are only
meaningful within one prompt version, so only the dominant partition survives.
The hash has to name the artifact under study.

Kitaru sessions do not carry a tracegrad prompt hash. They do carry per-node
`system_prompt_selector` pointers into node inputs. Those recorded prompts are
the only honest lineage: guessing a missing system prompt would attribute
failures to an artifact the session never ran.

`engine = "format"` manifests render differently per request. Hashing the
recorded (already rendered) prompts of a format template would give N partitions
of size one, and ingest would keep a batch of one. Reverse-templating those
recordings back onto a format template is a separate problem and is not solved
here.

## Decision

Hash the recorded system prompt extracted from root LLM nodes. The mapped batch
must be single-valued on that hash, which is what ingest already enforces.

This supports `engine = "none"` only. When the manifest declares
`engine = "format"`, refuse the Kitaru source with a named error rather than
proceeding into a batch that collapses.

Never infer or guess a missing system prompt. Zero unique recorded prompts is
`system-prompt-unavailable`. More than one unique recorded prompt on the root
LLM nodes of one session is `multiple-system-prompts`.

Reverse-templating for `format` prompts is deferred, not rejected.

## Consequences

- Kitaru-sourced runs with a `format` manifest fail closed.
- `prompt_hash` on a mapped `Trace` is `text_hash` of the extracted system
  prompt, so the rest of the pipeline does not learn Kitaru exists.
- Multi-prompt sessions drop at the source rather than poisoning the partition.
