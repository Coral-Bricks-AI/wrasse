# wrasse

An AI code-review agent that reads the code before it comments.

The cleaner wrasse swims into a bigger fish's mouth and removes the parasites.
This one opens the pull request, reads the files around the diff, and posts only
bugs — as inline review comments, on the exact lines, with a commit status so
you can gate merges on it.

- **Reads before it reports.** A bounded agentic loop with tools: `read_file`,
  `list_dir`, `grep`, `file_history`. It can trace a changed value to where it
  is consumed instead of guessing from the diff.
- **Bugs only.** No style, no naming, no "consider extracting a helper". The
  prompt, the JSON contract, and a parser that drops retracted findings all
  enforce it.
- **Re-reviews without repeating itself.** Open threads are never re-posted,
  resolved ones are not re-derived, outdated ones are resolved with a reason.
- **Zero dependencies.** Standard library only. No install step, no lockfile,
  no build.
- **Small enough to read.** ~4k lines of harness and ~3k lines of tests. Every
  design decision is a comment at the point it matters.

## Add it to a repo

One workflow file and one secret:

```yaml
# .github/workflows/wrasse-review.yml
name: Wrasse review
on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
permissions:
  contents: read
  pull-requests: write
  statuses: write
jobs:
  review:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: Coral-Bricks-AI/wrasse@v1
        env:
          CORALBRICKS_API_KEY: ${{ secrets.CORALBRICKS_API_KEY }}
```

`wrasse init` writes that file for you and prints the next two steps.

Bring your own model instead:

```yaml
      - uses: Coral-Bricks-AI/wrasse@v1
        with:
          provider: openai
          model: gpt-4.1
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Any OpenAI-compatible endpoint works (`--base-url` points at a self-hosted vLLM,
Ollama, LiteLLM or your own gateway). `--agent` needs an endpoint with a
Responses API; the one-turn path works anywhere that speaks chat completions.

## Run it locally

```sh
pip install wrasse            # or: uvx wrasse
wrasse doctor                 # what provider/model/key did it resolve?
wrasse review --pr 123 --dry-run     # full review, posts nothing
wrasse review --pr 123               # for real
```

No GitHub? Review a diff file, no tokens, no network to GitHub:

```sh
wrasse review --diff-file change.diff --dry-run
```

`--diff-file` plus `--head-sha` replays a historical commit exactly as it was:
the tools read the pinned sha, so you can re-run a review of an old PR head and
compare what two models (or two versions of this harness) say about the same
code.

## Layout

| path | what |
|---|---|
| `src/wrasse/core/review.py` | the orchestrator: fetch → review → parse → post → tidy |
| `src/wrasse/core/agent.py` | the bounded tool loop and its budget caps |
| `src/wrasse/core/contract.py` | prompt, findings JSON contract, parser, dedupe |
| `src/wrasse/core/diff.py` | which lines a review is allowed to comment on |
| `src/wrasse/providers/` | the provider table and one-turn/turn-threaded transport |
| `src/wrasse/platforms/github.py` | PR diff, files, thread state — all GitHub reads |
| `src/wrasse/platforms/publish.py` | review, threads, commit status |
| `src/wrasse/identity.py` | marker, status context, bot logins |
| `tests/` | 207 tests, no network |

## Extending it

The seams are deliberate and documented in
[docs/extensions.md](docs/extensions.md): providers, tools, publish targets and
the review contract. Several are already open (a provider is a dataclass entry,
not a fork); the plugin registry is the next milestone, and its interface is
specified there so contributions can be written against it.

## Status

`0.1.0` — extracted from a harness that has reviewed every pull request of a
large private monorepo since October 2026, including its own changes. The
behaviour is unchanged by the extraction; the tests are the same tests.

Not yet done, and deliberately so: a plugin registry (v0.2), GitLab/Bitbucket
publishers, and a benchmark harness for comparing models on a fixed set of PR
heads (v0.3).

License: to be chosen (intended: Apache-2.0). Until then, all rights reserved.
