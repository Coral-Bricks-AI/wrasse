# Contributing

Thanks for looking. This harness reviews pull requests for a living; the bar
for a change is the same one it applies to yours.

## Ground rules

- **No dependencies.** The standard library only. That is a feature: no install
  step, no supply chain, no version conflicts on a stock GitHub runner.
- **Tests are the specification.** `tests/test_harness.py` pins the findings
  parser, the dedupe rules, the posting shapes, and the retry policy.
  `tests/test_agent.py` pins the loop's turn sequencing and budget caps. A
  behaviour change without a test is an unreviewed change.
- **Explain the why in a comment.** The code is deliberately comment-heavy
  where a rule exists because something went wrong once: a false positive, a
  repeated finding, a truncated review. Keep that habit — a rule whose reason
  is lost gets deleted by the next person.
- **Small PRs.** One concern per pull request. The reviewer is a language
  model; a 40-file PR with three unrelated changes is where it does its worst.

## Run the tests

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

No network is used by the test suite. If you add a test that talks to a model
or to GitHub, it belongs behind an environment check and out of the default run.

## Try a change for real

```sh
export CORALBRICKS_API_KEY=...
wrasse review --pr <your pr> --dry-run          # one-turn
wrasse review --pr <your pr> --dry-run --agent  # tool loop, what CI runs
```

Dry-run posts nothing and prints the exact comments and summary a real run
would post. Diff a dry-run before and after your change when you touch the
prompt or the filters.

## Commits

`git commit -s` — sign-off is required (Developer Certificate of Origin,
https://developercertificate.org). It certifies you wrote it, or have the right
to submit it. No separate CLA, no paperwork.

Write the message for someone reading it a year from now: what was wrong, what
changed, why this way. If it fixes a bug, say what the bug looked like from the
outside — that is the part nobody can reconstruct later.

## Where to start

- `docs/extensions.md` has a ranked list and the plugin interface being built.
- Issues labelled `good first issue` are scoped to one file and one test.
- Anything in the "Not yet done" list in the README is fair game; open an issue
  first so we do not build the same thing twice.

## Code of conduct

Be decent. See `CODE_OF_CONDUCT.md`; report problems to the maintainers listed
there.
