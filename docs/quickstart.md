# Quickstart

## In CI

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

`wrasse init` writes this file.

Then open a PR. You get:

- inline threads on the changed lines, each tagged `[high]`, `[medium]` or
  `[low]` with a concrete fix;
- findings that cannot be attached to a diff hunk listed in the review body
  instead of silently dropped;
- a `Wrasse Review` commit status: `success` = the review ran, `error` = the
  review itself broke. It is never `failure`, because that would read as "the
  PR's code failed".

To gate merges on it, require the `Wrasse Review` status check in branch
protection. An `error` status is not `success`, so a broken review blocks the
merge — which is the behaviour you want.

## Locally

```sh
pip install wrasse

export CORALBRICKS_API_KEY=...      # or put API_KEY=... in ~/.config/wrasse/api_key
wrasse doctor                       # resolved provider, endpoint, model, key source

wrasse review --pr 123 --dry-run    # does everything except post
wrasse review --pr 123              # posts
```

Dry-run prints the report JSON, the summary body, and each inline comment
exactly as it would have been posted.

## Without GitHub

```sh
wrasse review --diff-file change.diff --dry-run
```

No token, no GitHub calls. Repo instructions are read from the current
directory, so run it from the repository root you want conventions from.

Replay a historical commit precisely (tools read the pinned sha, context is
that commit's):

```sh
wrasse review --diff-file change.diff --head-sha <sha> --pr-hint 123 --dry-run --agent
```

This is how model and harness versions are compared: same diff, same sha, same
intent, different reviewer.

## Choosing the one-turn path

`--agent` (the default in the action) is the tool loop: it reads code before
reporting, costs a few turns, and needs a Responses API endpoint. Without it,
one call over the diff answers — cheaper, faster, and noticeably more prone to
findings that a look at the surrounding code would have refuted.

```sh
wrasse review --pr 123 --dry-run            # one-turn
wrasse review --pr 123 --dry-run --agent    # tool loop
```

Both share the same findings contract, parser, dedupe and posting, so their
output is directly comparable.
