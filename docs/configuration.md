# Configuration

Flags win over environment, environment wins over the provider's defaults.

## Provider and model

| flag | env | meaning |
|---|---|---|
| `--provider` | `WRASSE_PROVIDER` | `coral` (default) or `openai` |
| `--model` | `WRASSE_MODEL` | model name; empty means the provider's default |
| `--base-url` | `WRASSE_BASE_URL` | endpoint override (self-hosted, proxy, staging) |
| `--api-key-file` | — | `KEY=VALUE` file, default `~/.config/wrasse/api_key` |

Key resolution order: `WRASSE_API_KEY`, then the provider's own variable
(`CORALBRICKS_API_KEY`, `OPENAI_API_KEY`), then the key file.

`wrasse doctor` prints what all of that resolved to. Run it first when a review
fails for a reason you did not expect.

A provider is a record in `src/wrasse/providers/__init__.py`:

```python
Provider(
    name="acme",
    base_url="https://inference.acme.internal",
    api_key_env="ACME_KEY",
    default_model="acme-reviewer-1",
    supports_responses=True,   # needed for --agent
)
```

## Review behaviour

| flag | env | default | meaning |
|---|---|---|---|
| `--agent` | — | off (on in the action) | tool loop instead of one turn |
| `--max-tool-turns` | `WRASSE_MAX_TOOL_TURNS` | 10 (25 in the action) | turns that may call a tool; the forced final is one on top |
| `--agent-wall-timeout` | `WRASSE_WALL_TIMEOUT` | 600 (1200 in the action) | whole-review seconds |
| `--max-diff-lines` | — | 10000 | above this the review is skipped, not failed |
| `--max-findings` | — | 30 | rest go to the summary body |
| `--timeout` | — | 300 | per model call, seconds |

Budget exhaustion never fails a review: it forces a final answer with what the
model already has and the summary and status say `budget exhausted`. The same
is true for an answer cut off by the model's output limit (`answer truncated`).

## Posting

| flag | env | meaning |
|---|---|---|
| `--repo` | `GITHUB_REPOSITORY` | `owner/name` |
| `--token` | `GITHUB_TOKEN` | token for reads and posting |
| `--dry-run` | — | everything except posting |

`WRASSE_BOT_LOGINS` (comma separated) is the set of logins whose comments count
as "ours" for dedupe and thread resolution. Default
`github-actions,github-actions[bot]`; set it if you run as a GitHub App.

## Marker, status, version

Defined once in `src/wrasse/identity.py`:

- `STATUS_CONTEXT = "Wrasse Review"` — the commit status and the first line of
  every summary. A branch-protection rule targets this exact string.
- `MARKER = "wrasse"` — the tag in the hidden comment marker. Changing it
  orphans every thread posted before the change.
- `REVIEW_VERSION` — the package version, advertised in every summary so a
  review can always be attributed to the harness that produced it.

## Ignored paths

Findings on lockfiles, minified assets, vendored trees and bulk data are
dropped rather than reported, and the count appears in the summary. The lists
are `_SKIP_EXACT` / `_SKIP_SUFFIX` / `_SKIP_PREFIX` in
`src/wrasse/core/contract.py`.
