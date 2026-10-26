"""The review orchestrator: fetch, review, parse, post, tidy up.

One run:
  1. Fetch the PR diff, head sha, and repo context (CLAUDE.md/AGENTS.md) from
     the PR head, guarded against the head moving mid-fetch.
  2. Bail out gracefully if the diff is too large (cap) — success status with
     a "skipped" description, never a hard failure on big PRs.
  3. Ask the model for findings: one call through ``call_model`` by default,
     or the bounded agentic loop in ``core/agent.py`` with ``--agent``. Both
     send one ``x-session-id`` per PR head, so the turns of a review group
     together in the provider's logs.
  4. Parse findings (bugs only, JSON contract), drop ignored paths (locks,
     vendored, generated), and split them into inline (line inside a diff
     hunk) and overflow (listed in the summary body instead).
  5. Post a COMMENT review with one inline thread per finding, each carrying
     the hidden ``<!-- wrasse -->`` marker, plus a commit status on the head.
  6. Thread housekeeping: findings whose position already has an open thread
     are withheld (and listed in the summary), and the run resolves its own
     outdated threads plus newer same-position duplicates. A repeat event on
     an unchanged head skips the model call but still runs the housekeeping.

Exit code 0 = the review ran clean (including zero findings and skipped
caps); 1 = harness error (network, unparseable reply, posting). A harness
error still posts an error status when it can, so a broken run never reads
as a clean one.

API key resolution: $WRASSE_API_KEY (what the action sets from a secret)
wins; otherwise a key file in ``KEY=VALUE`` form, default
``~/.config/wrasse/api_key`` — outside the repo tree, so a committed file
can never leak it.

Usage:
  wrasse review --pr 741                      # full run against a real PR
  wrasse review --pr 741 --dry-run            # fetch + model call, post nothing
  wrasse review --diff-file x.diff --dry-run  # local: no GitHub I/O at all
  wrasse review --pr 741 --dry-run --agent    # agentic loop (tools live)
  wrasse review --diff-file x.diff --head-sha <sha> --dry-run --agent
      # ^ replay: tools read the repo at the pinned historical sha

Run the tests:  python -m pytest -q
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from wrasse.core import diff as difflines
from wrasse.platforms import github as fetch
from wrasse.core import contract as findings_mod
from wrasse.providers import resolve_provider
from wrasse.providers.base import GatewayError, call_model
from wrasse.core import agent as agent_mod
from wrasse.platforms.publish import (
    REVIEW_VERSION,
    STATUS_CONTEXT,
    comment_body,
    post_review,
    post_status,
    reply_in_thread,
    resolve_thread,
)

# Both model and endpoint default to the selected provider (see
# wrasse.providers); --model / --base-url / $WRASSE_MODEL / $WRASSE_BASE_URL
# override. A provider with no default model needs one supplied.
DEFAULT_KEY_FILE = "~/.config/wrasse/api_key"
# First-turn resolved-thread history cap (an earlier round pre-A/B): long-lived
# PRs accumulate dozens of resolved threads; the most recent ones are the
# relevant do-not-re-derive context, and the prompt must stay
# diff-dominated. Anything older is disclosed as omitted, never silently.
RESOLVED_HISTORY_MAX = 40


class HeadMovedDuringFetch(RuntimeError):
    """The PR head kept moving across fetch attempts.

    The sha in hand may already be superseded, so no error status is
    posted for it — the run that owns the new head (the synchronize event
    already queued for it) posts the authoritative status (an earlier review round).
    """


def read_key_file(path: str) -> str:
    """Parse ``KEY=VALUE`` lines from a key file (smoke-script convention)."""
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "API_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError(f"no API_KEY= line in {path}")


def resolve_api_key(args: argparse.Namespace) -> str:
    """$WRASSE_API_KEY, then the provider's own variable, then a key file."""
    if os.environ.get("WRASSE_API_KEY"):
        return os.environ["WRASSE_API_KEY"]
    provider_key = os.environ.get(args.provider.api_key_env)
    if provider_key:
        return provider_key
    return read_key_file(args.api_key_file)


def resolve_own_threads(token: str, repo: str, pr_number: int, open_threads: list) -> int:
    """Resolve this run's own stale threads (findings.plan_thread_resolution):
    outdated ones and newer same-position duplicates. Resolve FIRST, reply
    with the reason second (an earlier review round): a failed resolution must not leave
    an open thread that a retry would reply to again — a resolved thread
    drops out of the next run's plan, so retries can't stack duplicate
    replies; a failed reason-reply leaves a silent resolution, which is the
    lesser evil. Best effort per thread — cleanup never fails the review."""
    resolved = 0
    for t, note in findings_mod.plan_thread_resolution(open_threads):
        try:
            resolve_thread(token, t["thread_id"])
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            print(f"thread cleanup failed for {t['thread_id']}: {e}", file=sys.stderr)
            continue
        resolved += 1
        try:
            reply_in_thread(token, repo, pr_number, t["comment_id"], note)
        except Exception as e:  # noqa: BLE001 — best-effort reason note
            print(f"reason reply failed for {t['thread_id']}: {e}", file=sys.stderr)
    return resolved


def minimize_resolved_rounds(token: str, repo: str, pr_number: int) -> int:
    """Minimize this PR's own review summaries whose rounds are fully
    resolved (B3, the previous reviewer's own pattern — an earlier round). Runs AFTER thread cleanup so
    a round this run just resolved is caught in the same pass. Best
    effort: a failed minimize never fails the review, and a missed one is
    picked up by the next run.

    Refuses to guess on partial fetches (an earlier review round/BUG-4): when a
    thread or review fetch hits its pagination cap, an unseen OPEN thread
    would read as resolved and minimize a live round's summary — a
    skipped pass (the next run retries) is the only safe move."""
    try:
        summaries, summaries_complete = fetch.fetch_own_summaries(
            token, repo, pr_number)
        if not summaries:
            return 0
        if not summaries_complete:
            print("summary minimize skipped: a fetch hit its pagination cap; "
                  "partial thread/review state must not minimize a live "
                  "round's summary (an earlier review round)", file=sys.stderr)
            return 0
        open_threads, resolved_threads, threads_complete = (
            fetch.fetch_thread_state(token, repo, pr_number))
        if not threads_complete:
            print("summary minimize skipped: a fetch hit its pagination cap; "
                  "partial thread/review state must not minimize a live "
                  "round's summary (an earlier review round)", file=sys.stderr)
            return 0
        doomed = fetch.summaries_to_minimize(
            summaries, open_threads, resolved_threads)
        count = 0
        for s in doomed:
            try:
                if fetch.minimize_comment(token, s["node_id"]):
                    count += 1
            except Exception as e:  # noqa: BLE001 — best-effort tidy
                print(f"summary minimize failed for {s['node_id']}: {e}",
                      file=sys.stderr)
        return count
    except Exception as e:  # noqa: BLE001 — best-effort tidy
        print(f"summary minimize pass skipped: {e}", file=sys.stderr)
        return 0


def build_summary(
    model: str,
    reply,
    inline: list,
    overflow: list,
    dropped_paths: int,
    dropped_cap: int,
    dropped_dupes: list | None = None,
    skipped_note: str | None = None,
    agent: bool = False,
    dropped_retracted: int = 0,
    withheld_resolved: list | None = None,
) -> str:
    """Human-readable review body. inline and overflow are disjoint by
    construction (single-pass partition in main), so total is their sum —
    never counted twice. dropped_dupes lists withheld findings explicitly
    (an earlier review round): a review that posts nothing new must not read as clean,
    and a distinct second bug at a withheld position must stay visible.
    withheld_resolved does the same for re-derivations of resolved
    threads (an earlier round, an earlier round): visible to a human, never a new thread.
    ``agent`` only changes the header (an earlier review round): an agentic v2 review
    must not call itself a one-turn v1 test in its own summary."""
    dupes = dropped_dupes or []
    reso = withheld_resolved or []
    total = len(inline) + len(overflow)
    if agent:
        head = (
            f"**Wrasse Review** — agentic review (v{REVIEW_VERSION}) "
            f"via `{model}` "
            "(agentic: it read the code around this diff before reporting)."
        )
    else:
        head = (
            f"**Wrasse Review** — one-turn review (v{REVIEW_VERSION}) "
            f"via `{model}` "
            "(one-turn: diff only, no code reading — findings are less certain)."
        )
    lines = [head, ""]
    if skipped_note:
        lines += [skipped_note, ""]
    if total == 0 and (dupes or reso):
        lines.append("No NEW bugs in this diff. Withheld finding(s):")
        if dupes:
            lines += [
                "",
                f"{len(dupes)} at position(s) that already have an open "
                "the review thread:",
                *[f"- `{f.path}:{f.line}"
                  + (f"-{f.end_line}" if f.end_line is not None else "")
                  + f"` **[{f.severity}]** {f.title}" for f in dupes],
            ]
        if reso:
            lines += [
                "",
                f"{len(reso)} matching thread(s) already raised and "
                "resolved on this PR (possible re-derivation; a genuine "
                "reintroduction would still be listed here):",
                *[f"- `{f.path}:{f.line}"
                  + (f"-{f.end_line}" if f.end_line is not None else "")
                  + f"` **[{f.severity}]** {f.title}" for f in reso],
            ]
        lines.append("")
    elif total == 0:
        lines.append("No bugs found in this diff.")
    elif not inline:
        lines.append(f"Found **{total}** bug(s), none commentable inline (listed below):")
    elif overflow:
        lines.append(
            f"Found **{total}** bug(s): {len(inline)} inline below, "
            f"{len(overflow)} outside the diff hunks (comments cannot attach there):"
        )
    else:
        lines.append(f"Found **{total}** bug(s): {len(inline)} inline below.")
    for f in overflow:
        where = f"{f.path}:{f.line}"
        if f.end_line is not None:
            where += f"-{f.end_line}"
        bullet = f"- `{where}` **[{f.severity}]** {f.title}"
        if f.detail:
            bullet += f" — {f.detail[:300]}"
        lines.append(bullet)
    if dropped_paths:
        lines.append(
            f"- ({dropped_paths} finding(s) on ignored paths — locks, datasets, "
            "vendored — were dropped.)"
        )
    if dropped_retracted:
        # Visible, not silent (same rule as dropped_paths): a review that
        # dropped self-retracted entries must say how many, so a human can
        # audit the guard's decisions against the model's raw output.
        lines.append(
            f"- ({dropped_retracted} finding(s) dropped — their own text "
            "concluded 'no bug' ('not reporting', 'tests pass', 'intended', "
            "…).)"
        )
    if dropped_cap:
        lines.append(
            f"- ({dropped_cap} further finding(s) beyond the --max-findings cap "
            "were not reported.)"
        )
    if dupes and total:
        lines += [
            "",
            f"{len(dupes)} finding(s) withheld — that position already has "
            "an open the review thread:",
            *[f"- `{f.path}:{f.line}` **[{f.severity}]** {f.title}" for f in dupes],
        ]
    if reso and total:
        # Only when findings posted (an earlier review round analysis): the total==0
        # branch above already lists reso under its own heading — an
        # unconditional block here would print every entry twice.
        lines += [
            "",
            f"{len(reso)} finding(s) withheld — position matches a thread "
            "already raised and resolved on this PR (possible "
            "re-derivation; a genuine reintroduction would still be listed "
            "here):",
            *[f"- `{f.path}:{f.line}` **[{f.severity}]** {f.title}" for f in reso],
        ]
    lines += [
        "",
        f"Model call: {reply.elapsed_s:.1f}s, {reply.prompt_tokens}+{reply.completion_tokens} tokens.",
    ]
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pr", type=int, help="PR number to review")
    ap.add_argument("--diff-file", help="Local diff file instead of a PR (testing)")
    ap.add_argument("--head-sha", help=(
        "Pin tools (and the local session id) to this sha — replay mode for "
        "the v2 A/B: --diff-file + --head-sha reviews a historical head "
        "without posting. Requires --diff-file; never valid with --pr, whose "
        "head is fetched live."))
    ap.add_argument("--pr-hint", type=int, help=(
        "Replay mode: the PR number the historical head belongs to, for "
        "the PR-intent section of the prompt (an earlier round). Requires "
        "--diff-file; never valid with --pr, whose intent is fetched live."))
    ap.add_argument("--agent", action="store_true", help=(
        "v2 agentic loop (an earlier round): bounded /v1/responses turns with "
        "runner-side tools, instead of the v1 one-turn chat call. LIVE in "
        "the workflows since the shadow-mode flip (deepseek-v4.1-flash-fast, "
        "non-gating); v1 remains available for one-turn comparisons."))
    ap.add_argument("--max-tool-turns", type=int,
                    default=int(os.environ.get("WRASSE_MAX_TOOL_TURNS", "10")),
                    help="Agent mode: model turns that may produce tool calls "
                         "(the forced final answer is one turn on top; tunable)")
    ap.add_argument("--agent-wall-timeout", type=float,
                    default=float(os.environ.get("WRASSE_WALL_TIMEOUT", "600")),
                    help="Agent mode: whole-review wall-clock budget in seconds (tunable)")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--dry-run", action="store_true", help="Do everything except posting to GitHub")
    ap.add_argument("--provider", default=os.environ.get("WRASSE_PROVIDER", "coral"),
                    help="Model provider: coral (default) or openai; see "
                         "wrasse/providers/__init__.py to add one")
    ap.add_argument("--model", default=os.environ.get("WRASSE_MODEL") or None,
                    help="Model name (default: the provider's)")
    ap.add_argument("--base-url", default=os.environ.get("WRASSE_BASE_URL") or None,
                    help="Model endpoint base URL (default: the provider's)")
    ap.add_argument("--api-key-file", default=DEFAULT_KEY_FILE,
                    help=f"Key file with an API_KEY= line (default: {DEFAULT_KEY_FILE}; "
                         "the dedicated review key, not a shared platform key)")
    # 10000 (2026-10-14; was 3000): a 10k-line diff is ~120k tokens, well inside the
    # model's context, and 3000 skipped ordinary large PRs (an earlier round at 3135 lines).
    ap.add_argument("--max-diff-lines", type=int, default=10000,
                    help="Skip the review (gracefully) above this many diff lines")
    ap.add_argument("--max-findings", type=int, default=30,
                    help="Report at most this many findings; the rest go in the summary")
    # 300 s, up from 180: a diff near the 10000-line cap is a ~120k-token prompt.
    ap.add_argument("--timeout", type=float, default=300.0, help="Model call timeout (s)")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token")
    args = ap.parse_args(argv)

    # Provider resolution before any other check: it supplies the defaults the
    # rest of the run is validated against.
    args.provider = resolve_provider(args.provider)
    args.model = args.model or args.provider.default_model
    if not args.model:
        ap.error(f"provider {args.provider.name!r} has no default model — "
                 "pass --model or set $WRASSE_MODEL")
    args.base_url = args.base_url or args.provider.base_url

    if not args.pr and not args.diff_file:
        ap.error("one of --pr or --diff-file is required")
    if args.pr and not args.token:
        ap.error("--pr (or $GITHUB_TOKEN) requires a GitHub token")
    if args.diff_file and args.pr:
        ap.error("--diff-file and --pr are mutually exclusive")
    if args.diff_file and not args.dry_run:
        # Without this, a bare --diff-file run would reach post_review with
        # pr_number=None and fail on a confusing /pulls/None 404 (an earlier review round).
        ap.error("--diff-file is local-only: it posts nothing, so it requires --dry-run")
    if args.head_sha and not args.diff_file:
        ap.error("--head-sha is replay mode and requires --diff-file "
                 "(--pr fetches its own live head)")
    if args.pr_hint and not args.diff_file:
        ap.error("--pr-hint is replay mode and requires --diff-file")
    if args.pr_hint and args.pr:
        ap.error("--pr-hint is for replay mode; --pr is the live review")
    if args.max_tool_turns < 1:
        ap.error("--max-tool-turns must be >= 1 — a review that never shows "
                 "the model the diff is not a review")
    if args.agent_wall_timeout <= 0:
        ap.error("--agent-wall-timeout must be > 0")
    if args.head_sha and not args.token:
        # Not fatal (an earlier review round): the run continues, but say so —
        # silent degradation is how a replay A/B ends up measuring reviews
        # whose tools answered errors and whose context came from the wrong
        # checkout.
        print("warning: --head-sha without a GitHub token — agent tools are "
              "disabled (per-call errors) and context falls back to the "
              "local checkout; pass --token/$GITHUB_TOKEN for a faithful "
              "replay", file=sys.stderr)

    api_key: str | None = None
    head_sha = args.head_sha or "local"
    pr_number = args.pr
    pr_intent: dict | None = None  # the PR's own title/body (an earlier round)
    open_threads: list = []  # own open review threads (dedupe + resolve)
    already_raised: list = []  # prompt lines for open-thread positions
    resolved_history: list = []  # own resolved threads (first-turn history)
        # Local (--diff-file, no token) runs read repo context from the CWD:
    # run them from the repository root being reviewed.
    repo_root = Path.cwd()

    try:
        if args.diff_file:
            diff = Path(args.diff_file).read_text(encoding="utf-8")
            if args.head_sha and args.token:
                # Replay mode: context must come from the PINNED sha, not
                # the current checkout — tools read the pinned commit, and
                # a historical head must be reviewed under the guidance
                # that was live then, not under today's (an earlier review round).
                context = fetch.fetch_repo_context(args.repo, args.head_sha, args.token)
            else:
                # Repo context from the repo root, not the CWD — running from a
                # subdirectory must not silently yield an empty context.
                context = "\n\n".join(
                    f"### {p}\n\n{(repo_root / p).read_text(encoding='utf-8')[:20_000]}"
                    for p in ("CLAUDE.md", "AGENTS.md") if (repo_root / p).exists()
                )
            if args.pr_hint:
                # Replay's PR intent (an earlier round). GitHub serves the
                # description as it stands NOW (later edits invisible) —
                # disclosed in the replay condition, and a fetch failure
                # degrades to no-intent: a review without it still ran.
                try:
                    pr_intent = fetch.fetch_pr_intent(args.repo, args.pr_hint, args.token)
                except fetch.GitHubError as e:
                    print(f"PR-intent fetch failed, reviewing without it: {e}",
                          file=sys.stderr)
        else:
            # Head-race guard (an earlier review round): meta and diff are two separate
            # reads, so a push in between would review one commit while
            # pinning comments/status to another. Re-check the head after the
            # diff read and retry a bounded number of times; give up with a
            # failure rather than post a mismatched review.
            for attempt in range(3):
                meta = fetch.fetch_pr_meta(args.repo, args.pr, args.token)
                head_sha = meta["head_sha"]
                diff = fetch.fetch_diff(args.repo, args.pr, args.token)
                if fetch.fetch_pr_meta(args.repo, args.pr, args.token)["head_sha"] == head_sha:
                    break
                print(f"head moved during fetch (attempt {attempt + 1}); retrying")
            else:
                raise HeadMovedDuringFetch("PR head kept moving; review superseded, not posting")
            # Context by head sha, not branch name: the sha is immutable, so a
            # same-named branch on the base repo can never supply the context.
            context = fetch.fetch_repo_context(args.repo, head_sha, args.token)
            # PR intent (an earlier round): the author's stated purpose is the
            # cheapest false-positive guard there is — a diff reviewed blind
            # manufactures findings on intended behavior. Same degradation
            # rule as replay: no-intent beats no-review.
            try:
                pr_intent = fetch.fetch_pr_intent(args.repo, args.pr, args.token)
            except fetch.GitHubError as e:
                print(f"PR-intent fetch failed, reviewing without it: {e}",
                      file=sys.stderr)

            # Duplicate-run guard: label churn (or any repeat event) must not
            # re-post the same review on the same head. The summary body
            # carries a head marker (platforms/publish.py), so this check is cheap.
            # --dry-run never posts, so it skips the guard and always
            # exercises the model (an earlier review round).
            if not args.dry_run and fetch.head_already_reviewed(
                args.token, args.repo, args.pr, head_sha,
                require_agent=args.agent,
            ):
                # Repair only when the head is not already green: repeat
                # events must not pile up duplicate success rows (an earlier review round).
                # A failed lookup posts rather than skips — a duplicate row
                # beats an unrepaired red status. LATEST status, not any
                # historical success: an older success must not mask a
                # newer error on the same head (an earlier review round). 'pending'
                # means another run on this head is in flight — its owner
                # will post; a stomp here would race it.
                try:
                    latest_state = fetch.latest_status_state(
                        args.token, args.repo, head_sha, STATUS_CONTEXT
                    )
                except fetch.GitHubError as lookup_err:
                    print(f"status lookup failed, posting anyway: {lookup_err}",
                          file=sys.stderr)
                    latest_state = None
                if latest_state != "success" and latest_state != "pending":
                    try:
                        post_status(
                            args.token, args.repo, head_sha, "success",
                            "review already posted on this head (status repaired)",
                        )
                    except Exception as repair_err:  # noqa: BLE001 — best-effort repair
                        print(f"status repair failed: {repair_err}", file=sys.stderr)
                # Repeat event, unchanged head: no model call (same diff,
                # same findings), but the thread housekeeping still runs —
                # resolving stale threads needs no model call, and an
                # unchanged head must not block the cleanup (an earlier round
                # 2026-10-15). --dry-run never reached this branch.
                try:
                    repeat_threads = fetch.fetch_open_own_threads(
                        args.token, args.repo, args.pr)
                except fetch.GitHubError as e:
                    print(f"open-thread fetch failed, cleanup skipped: {e}",
                          file=sys.stderr)
                    repeat_threads = []
                resolved = resolve_own_threads(
                    args.token, args.repo, args.pr, repeat_threads)
                minimized = minimize_resolved_rounds(args.token, args.repo, args.pr)
                print(json.dumps({"pr": args.pr, "head": head_sha,
                                  "skipped": "head already reviewed",
                                  "resolved_threads": resolved,
                                  **({"minimized_summaries": minimized}
                                     if minimized else {})}))
                return 0

            # Own open threads, for dedupe + resolve (an earlier round 2026-10-15).
            # A fetch failure degrades to no dedupe rather than failing the
            # review: a missed dedupe posts one duplicate, a failed review
            # posts nothing.
            if args.pr:
                try:
                    open_threads = fetch.fetch_open_own_threads(
                        args.token, args.repo, args.pr)
                except fetch.GitHubError as e:
                    print(f"open-thread fetch failed, running without dedupe: {e}",
                          file=sys.stderr)
            already_raised = [
                f"- `{findings_mod.format_thread_position(t)}` — {t['title']}"
                for t in open_threads
                if not t["outdated"]
                and (t["line"] is not None or t.get("marker_start") is not None
                     or t.get("original_line") is not None)
            ]

            # Resolved-thread history (an earlier round pre-A/B): the first turn
            # must know what was already raised AND addressed, or the
            # reviewer re-derives the same shapes every round (an earlier round: two
            # rounds, 17 findings, zero new, mostly repeats of resolved
            # threads). Same degradation rule as the open-thread fetch: a
            # missed history line costs a duplicate finding, a failed
            # review posts nothing. Prompt context only — the hard dedupe
            # (split_already_raised) still sees open threads only.
            try:
                resolved_history = fetch.fetch_resolved_own_threads(
                    args.token, args.repo, args.pr)
            except fetch.GitHubError as e:
                print(f"resolved-thread fetch failed, running without "
                      f"history: {e}", file=sys.stderr)
                resolved_history = []

        diff_lines = len(diff.splitlines())
        if diff_lines > args.max_diff_lines:
            note = f"Skipped: diff is {diff_lines} lines (cap {args.max_diff_lines})."
            print(note)
            # Idempotent skip (an earlier review round round 6): without this check
            # every event on an unchanged oversize head re-fetched the whole
            # diff and re-posted the status forever. Guards: --diff-file runs
            # never reach here with posting rights (head stays "local", and
            # a bare --diff-file is rejected by argparse), but defend anyway.
            # A failed lookup posts rather than fails the run — the skip
            # must stay graceful even when the status read errors (an earlier review round).
            # LATEST status, not any historical success (an earlier review round): an
            # error posted after an earlier skip-success still gets repaired;
            # 'pending' is left for the in-flight run that owns it.
            if not args.dry_run and args.token and head_sha != "local":
                try:
                    latest_state = fetch.latest_status_state(
                        args.token, args.repo, head_sha, STATUS_CONTEXT
                    )
                except fetch.GitHubError as lookup_err:
                    print(f"status lookup failed, posting anyway: {lookup_err}",
                          file=sys.stderr)
                    latest_state = None
                if latest_state != "success" and latest_state != "pending":
                    post_status(args.token, args.repo, head_sha, "success", note)
            # Cleanup runs even on the oversize skip (an earlier review round): resolving
            # stale threads needs no model call, so a big diff must not leave
            # outdated/duplicate threads open. No-op when the plan is empty.
            if not args.dry_run and args.token and args.pr:
                try:
                    big_threads = fetch.fetch_open_own_threads(
                        args.token, args.repo, args.pr)
                except fetch.GitHubError as e:
                    print(f"open-thread fetch failed, cleanup skipped: {e}",
                          file=sys.stderr)
                    big_threads = []
                resolve_own_threads(args.token, args.repo, args.pr, big_threads)
                minimize_resolved_rounds(args.token, args.repo, args.pr)
            return 0

        # Key resolution lives INSIDE the try block so a missing key file
        # fails through the handler (error status + clean message) instead
        # of an uncaught traceback (an earlier review round).
        api_key = resolve_api_key(args)

        # Resolved-history prompt lines: most recent first, bounded, and a
        # disclosed omission note when older threads fall outside the cap —
        # the model must know the list is a truncation, not the whole past.
        resolved_history.sort(key=lambda t: t["created_at"], reverse=True)
        already_resolved = [
            f"- `{findings_mod.format_thread_position(t)}` — {t['title']}"
            + (f" (resolved: {t['resolution']})" if t.get("resolution") else "")
            for t in resolved_history[:RESOLVED_HISTORY_MAX]
        ]
        omitted_history = len(resolved_history) - len(already_resolved)
        if omitted_history > 0:
            already_resolved.append(
                f"- ({omitted_history} older resolved thread(s) omitted — "
                "the list above is the most recent history, not all of it.)"
            )

        session_id = f"review-pr{pr_number or 'local'}-{head_sha[:10]}"
        if args.agent:
            # v2 (an earlier round): bounded agentic loop on /v1/responses; the
            # findings contract, dedupe, caps, and posting below are shared
            # with v1 — only the transport differs.
            reply = agent_mod.run_agent_review(
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                session_id=session_id,
                repo=args.repo,
                head_sha=head_sha,
                token=args.token,
                diff=diff,
                context=context,
                already_raised=already_raised,
                already_resolved=already_resolved,
                pr_intent=pr_intent,
                max_tool_turns=args.max_tool_turns,
                wall_timeout_s=args.agent_wall_timeout,
                per_turn_timeout_s=args.timeout,
            )
        else:
            reply = call_model(
                api_key=api_key,
                model=args.model,
                system=findings_mod.SYSTEM_PROMPT,
                user=findings_mod.build_user_prompt(
                    diff, context, already_raised, already_resolved, pr_intent),
                base_url=args.base_url,
                session_id=session_id,
                timeout_s=args.timeout,
            )

        hunks = difflines.parse_hunks(diff)
        parsed_all = findings_mod.parse_findings(reply.content)
        if parsed_all is None:
            # an earlier review round: unparseable model output is NOT "no bugs" — the
            # status must say the review is inconclusive, not clean.
            # Raw reply to the job log first (an earlier review round follow-up):
            # truncation and off-contract replies were indistinguishable
            # after the fact, and the A/B scores exactly this per-head
            # failure mode. Head+tail, bounded — the tail is where a
            # truncated JSON block dies. session_id joins the dump to
            # inference_requests.
            raw = reply.content or ""
            print(
                f"unparseable model output (session {session_id}, "
                f"{len(raw)} chars):\n--- raw head ---\n{raw[:1500]}\n"
                f"--- raw tail ---\n{raw[-1500:]}\n",
                file=sys.stderr,
            )
            raise RuntimeError("could not parse model output (truncated or off-contract reply)")
        # Self-retraction guard (an earlier round pre-A/B, an earlier review round/an earlier round): drop
        # findings whose own text concludes "no bug" — the model reported
        # them and retracted them in the same breath, and until now the
        # post path ran anyway, inflating the false-positive count the A/B
        # pre-registered as a pass bar. Shared parse section, so v1 and v2
        # both shed the artifact. Dropped ones are counted, never silent.
        parsed_all, dropped_retracted = findings_mod.drop_self_retracted(parsed_all)
        # Normalize b/<path> header-form paths onto real diff files — only
        # when the stripped form matches, so a real a/ or b/ directory
        # survives (an earlier review round). Before path filtering, so prefixed
        # generated/dataset paths are filtered correctly too. A collision
        # (the repo really contains BOTH x.py and b/x.py) is ambiguous:
        # demote to the summary rather than guess (an earlier review round). Re-dedup
        # and rekey afterwards, so x.py and b/x.py for one bug collapse
        # into a single thread (an earlier review round).
        ambiguous: list = []
        resolved: list = []
        dropped_ambiguous = 0
        for f in parsed_all:
            path, is_ambiguous = difflines.resolve_diff_path(f.path, hunks)
            if is_ambiguous:
                # Conservative path filtering for ambiguous findings: drop
                # if EITHER candidate path is a filtered path (an earlier review round) —
                # a lock/dataset finding must not reach the summary by
                # exploiting the collision escape hatch.
                raw_ok = findings_mod.keep_path(f.path)
                stripped_ok = (
                    not f.path.startswith(("a/", "b/"))
                    or findings_mod.keep_path(f.path[2:])
                )
                if raw_ok and stripped_ok:
                    ambiguous.append(f)
                else:
                    dropped_ambiguous += 1
                continue
            f.path = path
            resolved.append(f)
        deduped = findings_mod.dedupe_findings(resolved)
        kept = [f for f in deduped if findings_mod.keep_path(f.path)]
        dropped_paths = len(deduped) - len(kept) + dropped_ambiguous
        # Clamp spans to the hunk BEFORE the dedupe splits (an earlier review round
        # BUG-3): the splits test the span the comment will actually
        # post — a raw end_line reaching into another hunk must not
        # withhold a finding there that the posted comment can't cover.
        # Out-of-hunk STARTS keep their raw span: they never post inline
        # (summary only), and erasing their end_line would corrupt the
        # summary + JSON payload spans (an earlier review round) — so the
        # assignment only lands when the clamp produced a value.
        for f in kept:
            if f.end_line is None:
                continue
            clamped = difflines.clamp_end_to_hunk(
                hunks, f.path, f.line, f.end_line)
            if clamped is not None:
                f.end_line = clamped
        # Re-review dedupe (an earlier round 2026-10-15): drop findings whose
        # (path, line) already has an open thread, BEFORE the cap so a
        # duplicate never spends a slot a new bug could have used. The
        # withheld ones go to the summary (an earlier review round) — visible, not silent.
        fresh, withheld_dupes = findings_mod.split_already_raised(kept, open_threads)
        dropped_dupes = len(withheld_dupes)
        # Resolved-thread re-derivations (an earlier round, an earlier round data):
        # hard filter, same as the open-thread dedupe — prose alone does
        # not stop a one-turn model from re-posting an answered position.
        fresh, withheld_resolved = findings_mod.split_resolved_rederivations(
            fresh, resolved_history)
        # Ambiguous findings share the same selection pipeline — severity
        # ranking and the cap apply to them too (an earlier review round) — merged in
        # ORIGINAL MODEL ORDER, not bucket-concatenated: at the cap, an
        # earlier-emitted ambiguous bug must not lose its slot to a later
        # resolved one of equal severity (an earlier review round).
        order = {id(f): i for i, f in enumerate(parsed_all)}
        combined = sorted(fresh + ambiguous, key=lambda f: order[id(f)])
        all_keep, over_cap = findings_mod.select_findings(combined, args.max_findings)
        # Single-pass partition: no membership test, no double count (the previous reviewer
        # an earlier round). Ambiguous findings always route to the summary — their
        # inline placement is unknowable, never guessed.
        amb_ids = {id(f) for f in ambiguous}
        inline: list = []
        overflow: list = []
        for f in all_keep:
            if id(f) in amb_ids or not difflines.line_in_hunks(hunks, f.path, f.line):
                overflow.append(f)
            else:
                inline.append(f)

        summary = build_summary(
            args.model, reply, inline, overflow, dropped_paths, over_cap,
            withheld_dupes, agent=args.agent,
            dropped_retracted=len(dropped_retracted),
            withheld_resolved=withheld_resolved,
        )
        if args.agent:
            # Agent telemetry on the review body: turns and tool calls are
            # what makes an agentic review reproducible from the summary
            # alone; a budget-exhausted run must never read as a full one,
            # and neither may a token-limit-truncated answer (an earlier review round).
            summary += (
                f"\n\nAgentic loop: {reply.turns} model turn(s), "
                f"{reply.tool_calls} tool call(s)"
                + (" — budget exhausted, review cut short" if reply.budget_exhausted else "")
                + (" — final answer truncated by the model's output-token limit, "
                   "findings may be incomplete" if reply.final_truncated else "")
                + "."
            )

        report = {
            "pr": pr_number,
            "head": head_sha,
            "model": args.model,
            "session_id": session_id,
            # an earlier review round: replays pair a pinned diff with the PR intent as
            # it stands NOW — later description edits shift review criteria.
            # Recording the title actually used makes each row's criterion
            # auditable when the arms are compared.
            "pr_intent_title": (pr_intent or {}).get("title") or None,
            "diff_lines": diff_lines,
            "inline": len(inline),
            "overflow": len(overflow),
            "dropped_paths": dropped_paths,
            "dropped_cap": over_cap,
            "dropped_dupes": dropped_dupes,
            "dropped_resolved": len(withheld_resolved),
            "dropped_retracted": len(dropped_retracted),
            "ambiguous": len(ambiguous),
            # Findings, not just counts (replay A/B, an earlier round): the ground-truth
            # join — "did a later commit on that PR fix it?" — needs
            # path/line/severity per finding; counts alone cannot be scored.
            # Exactly the list a posted review would carry (inline +
            # overflow, post-cap), detail bounded like the summary's.
            "findings": [
                {"path": f.path, "line": f.line,
                 **({"end_line": f.end_line} if f.end_line is not None else {}),
                 "severity": f.severity,
                 "title": f.title, "detail": f.detail[:300]}
                for f in inline + overflow
            ],
            "open_bot_threads": len(open_threads),
            "resolved_history_threads": len(resolved_history),
            "elapsed_s": round(reply.elapsed_s, 2),
            "tokens": [reply.prompt_tokens, reply.completion_tokens],
            "dry_run": args.dry_run,
        }
        if args.agent:
            report["agent"] = {
                "turns": reply.turns,
                "tool_calls": reply.tool_calls,
                "tool_counts": reply.tool_counts,
                "budget_exhausted": reply.budget_exhausted,
                "final_truncated": reply.final_truncated,
                "max_tool_turns": args.max_tool_turns,
                "wall_timeout_s": args.agent_wall_timeout,
            }
        print(json.dumps(report, indent=2))

        if args.dry_run:
            print("\n=== summary body (not posted) ===")
            print(summary)
            for f in inline:
                print(f"\n=== inline comment {f.path}:{f.line} (not posted) ===")
                print(comment_body(f, head_sha))
            print(f"\n=== status (not posted): {STATUS_CONTEXT} = success ===")
            return 0

        post_review(args.token, args.repo, pr_number, head_sha, summary, inline)
        # The status description must not read "no bugs found" when findings
        # were only withheld as duplicates (an earlier review round) — a reader of the
        # checks list would conclude the diff is clean when it is not.
        total_found = len(inline) + len(overflow)
        if total_found:
            desc = f"{total_found} new bug(s) found"
            if dropped_dupes:
                desc += f"; {dropped_dupes} already-open not re-posted"
            if withheld_resolved:
                desc += f"; {len(withheld_resolved)} resolved-thread re-derivation(s) not re-posted"
        elif dropped_dupes or withheld_resolved:
            desc = ("no new bugs "
                    f"({dropped_dupes} already-open, "
                    f"{len(withheld_resolved)} resolved re-derivation(s))")
        else:
            desc = "no bugs found"
        if args.agent and reply.budget_exhausted:
            # A cut-short review is still success (it ran clean and posted
            # what it had), but the checks list must not read it as full.
            desc += "; budget exhausted"
        if args.agent and reply.final_truncated:
            # Same rule for a token-limit-truncated answer (an earlier review round):
            # success, but visibly incomplete.
            desc += "; answer truncated"
        post_status(args.token, args.repo, head_sha, "success", desc)
        # Housekeeping after posting: resolving stale threads must not be
        # able to lose the fresh review above.
        resolved = resolve_own_threads(args.token, args.repo, pr_number, open_threads)
        if resolved:
            print(json.dumps({"pr": pr_number, "head": head_sha,
                              "resolved_threads": resolved}))
        minimized = minimize_resolved_rounds(args.token, args.repo, pr_number)
        if minimized:
            print(json.dumps({"pr": pr_number, "head": head_sha,
                              "minimized_summaries": minimized}))
        return 0

    except (GatewayError, fetch.GitHubError, RuntimeError, OSError) as e:
        print(f"review failed: {e}", file=sys.stderr)
        if isinstance(e, HeadMovedDuringFetch):
            # No status: the sha may be superseded and its real run is
            # already queued — a red status here would land on a stale head.
            return 1
        if not args.dry_run and args.token and head_sha != "local":
            try:
                # State "error", not "failure" (2026-10-17, the an earlier round 502):
                # a harness/service failure is never a verdict on the PR's
                # code — the review died before judging it. "error" reads
                # in the checks list as infrastructure, and the automation
                # gate (at flip: status must be "success") is unaffected:
                # error is not success, so a blip still blocks a merge.
                post_status(
                    args.token, args.repo, head_sha, "error",
                    f"harness error, not a code verdict — re-run /wrasse: "
                    f"{str(e)[:80]}",
                )
            except Exception as status_err:  # noqa: BLE001 — best-effort status
                print(f"also failed to post error status: {status_err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
