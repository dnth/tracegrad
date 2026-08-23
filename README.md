# tracegrad

**Evidence-gated system-prompt optimization from the traces you already have.**

Your LLM app runs. A judge scores each response. The traces pile up — and nothing
reads them. When the prompt gets edited, it's because someone remembered a failure
and guessed at a fix.

tracegrad closes that loop. Feed it a batch of traces with judge results, and it
proposes a small set of edits to your prompt template — each one backed by verbatim
quotes from real traces, capped at five per run, and gated by you. It never writes
without your approval.

> **Status: pre-release.** The design spec is complete and reviewed; implementation
> is in progress. See the [implementation plan](../../issues/1). The interface
> described below is the target, not yet the shipping tool.

## Why

- **Prompt edits are guesses.** tracegrad replaces "I think the tone instruction
  is the problem" with "this instruction was violated in 19% of traces — here are
  the quotes."
- **Prompts only grow.** Every instruction you add costs tokens on every request,
  forever, and dilutes the rest. tracegrad works under a token budget: at the
  ceiling, every addition must name the removal that pays for it.
- **One weird trace shouldn't rewrite your prompt.** A new instruction needs the
  same failure in at least two independent sessions or runs, tracked across
  batches in a ledger.
- **You stay in charge.** Analysis never writes. `tracegrad apply` shows each edit
  with its evidence; you accept or reject. Rejections are remembered.

## How it works

```mermaid
flowchart TD
    A[Your LLM app] -->|"traces + judge scores<br/>+ rationales (JSONL)"| B[distill<br/>deterministic reduction]
    P[Prompt template] --> B
    B --> C[attribute<br/>which instruction failed, which is missing<br/>quotes verified in code]
    C --> D[aggregate<br/>failure themes, counts, cross-run ledgers<br/>no model, no vibes]
    D --> E["synthesize<br/>≤5 proposed edits<br/>checked by mechanical gates"]
    E --> F{{"apply — human gate<br/>one card per edit: diff + evidence<br/>accept / reject"}}
    F -->|accepted edits| G[Updated prompt template]
    F -.->|rejections remembered| E
    G -->|you deploy| A
    A -->|"next batch: trend report<br/>rate before → after, with CIs"| F
```

Every quote is substring-verified in code against the stored trace — a model
cannot confabulate evidence past the gate. On the next batch, tracegrad shows you
how each accepted edit's failure theme moved, with confidence intervals, so you
know whether it worked before you trust it.

## What you need

- Your traces exported as JSONL: `{trace_id, input, output, judge: {score,
  rationale}, prompt_hash}` per line. One small export script from whatever eval
  stack you use — tracegrad never learns your stack.
- A judge that writes a **rationale**, not just a score. The rationale is the
  signal.
- A model to run the analysis. Two options, mixable per stage:
  - any OpenAI-compatible API (OpenRouter by default) with one env-var key, or
  - a coding-agent harness you're already logged into (`claude` supported
    out of the box; others are a config entry).

## Usage (target interface)

```sh
uv tool install tracegrad

cd my-app-evals
tracegrad init
tracegrad run --traces batch.jsonl --manifest manifest.json --estimate  # cost preview
tracegrad run --traces batch.jsonl --manifest manifest.json             # analyze, propose
tracegrad apply                                                         # review, accept/reject
tracegrad status                                                        # budget, trends, ledgers
```

## What tracegrad is not

- Not an autonomous loop. Cross-batch trends are advisory — statistics with error
  bars for you to read, never an automatic revert. (An opt-in A/B mode for
  trustworthy automatic verdicts is on the roadmap.)
- Not an eval harness. It doesn't run your app or host your judge.
- Not magic attribution. It optimizes against your judge; if your judge is wrong,
  tracegrad is confidently wrong with it. Freeze and version your judge.

## Prior art

The discipline layer — evidence gating, ledgers, edit caps, budget, human gate —
is adapted from [backpass](https://github.com/kunchenguid/backpass), which does
this for agent memory files. tracegrad applies it where an explicit judge signal
exists, and swaps transcript archaeology for measurable loss.

## License

MIT
