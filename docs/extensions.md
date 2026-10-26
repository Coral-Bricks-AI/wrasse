# Extending wrasse

The harness is small on purpose: the interesting parts are meant to be replaced
without editing the review loop. This page says which seams exist **today**,
and specifies the plugin registry that is the next milestone so contributions
can be written against it.

## Seams that exist today

### A provider (works now)

Add a `Provider` record in `src/wrasse/providers/__init__.py`. Endpoint, key
variable, default model, and whether it speaks the Responses API (`--agent`)
or only chat completions. File a PR with the record and a line in the README;
nothing else changes.

### A different model or endpoint (works now, no code)

`--provider`, `--model`, `--base-url`, and the matching environment variables.
Point `--base-url` at vLLM, Ollama, LiteLLM, a proxy, or a company gateway.

### The review prompt and the findings contract

`SYSTEM_PROMPT` and `build_user_prompt` live in `src/wrasse/core/contract.py`.
The output contract is a JSON array; `parse_findings` validates it strictly and
returns `None` (inconclusive) rather than `[]` (no bugs) when it cannot find
one. If you change the contract, change the parser and the tests together —
`tests/test_harness.py` is the specification.

### Post-parse filters

`drop_self_retracted`, `keep_path`, `select_findings`,
`split_already_raised`, `split_resolved_rederivations` — all pure functions
over `list[Finding]`, all unit-tested. This is the cheapest place to add
policy (severity caps, path rules, "never report X").

### A publishing target

`src/wrasse/platforms/publish.py` talks to GitHub through
`platforms/github.py` primitives (`gh_get`, `gh_post`, `gh_graphql`). A
different forge means a new module with the same three primitives plus
`post_review` / `post_status`; the orchestrator's posting section is the only
caller.

## The registry (v0.2 — designed, not built)

The goal: a contributor adds a capability in one file, without touching core.

```python
# examples/extensions/sentry_context.py
from wrasse import hook, ContextSection, Tool


@hook("context")
def sentry_errors(ctx) -> ContextSection:
    """Extra first-turn context: recent production errors for the touched files."""
    return ContextSection("Recent Sentry errors", fetch_sentry(ctx.repo, ctx.sha))


@hook("tool")
def jira_ticket() -> Tool:
    """A tool the model may call, next to read_file/grep."""
    return Tool(
        name="jira_ticket",
        description="Read one Jira issue by key.",
        parameters={"type": "object", "properties": {"key": {"type": "string"}},
                    "required": ["key"]},
        handler=lambda key, ctx: fetch_jira(key),
    )


@hook("finding_filter")
def no_secrets_in_titles(finding, ctx):
    """Deterministic post-parse filter, same pipeline as the built-ins."""
    return None if looks_like_secret(finding.title) else finding


@hook("publish")
def slack(review, ctx):
    """Extra sink, called after the platform post succeeds."""
    post_to_slack(review.summary, review.inline)
```

Loading: `entry_points(group="wrasse.extensions")` for installed packages,
`--extension path/to/file.py` for one file, and `.wrasse/extensions/*.py` in
the repository under review. Hooks are collected, then dispatched in
registration order.

Hook points mirror the run, so the mental model is "the same seven steps, with
a door at each":

| hook | when | typical use |
|---|---|---|
| `context` | before the first model turn | issue tracker, ADRs, ownership, incident history |
| `tool` | when the tool list is built | `blame`, `run_tests`, `semgrep`, `docs_search` |
| `finding_filter` | after parsing, before dedupe | semgrep/ruff findings merged in, severity remap |
| `publish` | after the platform post | Slack, Jira, CSV, a dashboard |
| `provider` | at startup | a model endpoint registered in code |

Two rules keep this from turning into a framework:

1. **Hooks receive and return plain data.** No mutable session object, no
   lifecycle to learn. A hook that raises is reported and skipped, never a
   failed review.
2. **Versioned surface.** Extensions declare `requires_wrasse = ">=0.2"`;
   anything not on this page is internal and may move.

## What to contribute

Ranked by what the project needs, from the issue tracker's perspective:

1. **Deterministic checks alongside the model** — semgrep, ruff, type errors,
   secret scanners. They are cheap, precise, and complementary: the model
   catches semantics, a linter catches facts. Findings merge into one review.
2. **More context hooks** — linked issue bodies, ADRs, ownership maps, recent
   incidents. Context is the difference between a plausible finding and a real
   one.
3. **Non-GitHub publishers** — GitLab, Bitbucket, Gitea, plain git hooks.
4. **Benchmarks** — a fixed set of PR heads with labelled outcomes, so model
   and prompt changes can be compared instead of argued about.
5. **Prompt-injection hardening** — see `docs/trust-model.md`.
