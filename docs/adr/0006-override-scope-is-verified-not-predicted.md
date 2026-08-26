# 0006. Override scope is verified, not predicted

## Status

Accepted

## Context

`ReplayOverride.system_prompt` is a single unscoped string. The Kitaru adapters
disagree about where it lands:

- OpenAI Agents scopes it to the starting agent — `if data.agent is not
  starting_agent: return model_data`. Subagents keep their own instructions.
- LangGraph applies it in `_model_request` with no starting-agent check.
  **Every** model call gets the candidate, subagents included.

`AgentCapabilities` carries only tools, MCP servers, and skills. Whether an
adapter honours a system-prompt override, and where, is not exposed over the
wire. Preflight cannot predict it.

On a multi-agent LangGraph app, a naive submission moves two variables and
reports one: the candidate prompt *and* every subagent's prompt. A report that
says "the prompt caused this" would be a lie.

## Decision

Change exactly one variable: the system prompt, passed as
`ReplayOverride.system_prompt`. Hold fixed: baseline inputs, agent code and
version, model, model params, cohort, evaluator version, recorded tool history.

After each replay, fetch the result session's nodes and assert:

- every root LLM node carries the candidate prompt, and
- every non-root LLM node carries what its baseline counterpart carried.

A session failing either is `OVERRIDE_SCOPE_DIVERGENCE` and is reported
**incomparable** — not improved, not regressed.

A tool-history miss is `TOOL_HISTORY_MISS` and likewise incomparable.

## Consequences

- Headline improved/regressed counts never include diverged sessions.
- Adapter disagreement becomes a typed, per-session fact instead of a silent
  confounder.
- Preflight stays cheap; the assertion is the backstop.
