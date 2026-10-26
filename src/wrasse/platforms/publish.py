"""Posting to GitHub: the review, its inline threads, and the commit status.

This module always posts; ``--dry-run`` is decided by the caller, which simply
never calls these functions. The hidden marker at the end of each inline
comment is the machine-readable contract that re-reviews parse back:

    <!-- wrasse id=<fid> severity=<sev> kind=bug sha=<head-sha> -->

The same marker is what makes thread housekeeping possible: a later run can
tell its own threads from a human's, match them to a position, and resolve
the ones this diff superseded.

Status state semantics: ``success`` means the review ran clean (findings or
not, including skipped-because-too-large); ``error`` means the harness or the
provider broke. It is never ``failure`` — that would read in the checks list
as "the PR's code failed" when only the review died. If you gate merges on
this status, error still blocks, which is what you want.
"""

from __future__ import annotations

from wrasse.core.contract import Finding
from wrasse.identity import MARKER, REVIEW_VERSION, STATUS_CONTEXT
from wrasse.platforms.github import GitHubError, gh_graphql, gh_post

# The version every review advertises in its summary comes from the
# package version (wrasse.identity.REVIEW_VERSION), so a release cannot forget
# to bump what the summaries claim.


def comment_body(finding: Finding, head_sha: str) -> str:
    """Inline comment text: severity + bug, then the hidden marker."""
    lines = [f"**[{finding.severity}] {finding.title}**", ""]
    if finding.detail:
        lines += [finding.detail, ""]
    marker = (
        f"<!-- {MARKER} id={finding.fid} severity={finding.severity} "
        f"kind=bug sha={head_sha[:10]}"
    )
    if finding.end_line is not None and finding.end_line > finding.line:
        # Both endpoints: GitHub records a range comment's `line` as the
        # range END, so the START exists only here (an earlier review round) —
        # without it a re-derivation at the original start line escapes
        # dedupe.
        marker += f" start_line={finding.line} end_line={finding.end_line}"
    lines.append(marker + " -->")
    return "\n".join(lines)


def _comment_position(f: Finding) -> dict:
    """GitHub review-comment position payload for a finding. A multi-line
    finding (end_line > line) posts as a range comment: start_line..line,
    both on the RIGHT side — GitHub's multi-line anchor. Single-line
    findings omit start_line (a start_line == line payload is a 422)."""
    pos = {
        "path": f.path,
        "line": f.line if f.end_line is None else f.end_line,
        "side": "RIGHT",
    }
    if f.end_line is not None and f.end_line > f.line:
        pos["start_line"] = f.line
        pos["start_side"] = "RIGHT"
    return pos


def post_review(
    token: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    summary: str,
    findings: list[Finding],
) -> dict:
    """Post a COMMENT review on the PR head with one inline thread per finding.

    The summary body ends with a head marker so repeat events on an unchanged
    head can be detected and skipped (fetch.head_already_reviewed).

    A 422 means GitHub rejected a comment position (line outside the diff —
    should not happen, difflines routes those to the summary). Degrade by
    posting the summary alone — with every dropped finding spelled out in the
    body, so no bug report is lost (an earlier review round) — rather than losing the
    whole review.
    """
    payload = {
        "commit_id": head_sha,
        "body": summary.rstrip() + f"\n\n<!-- {MARKER} head={head_sha[:10]} "
        f"v={REVIEW_VERSION} -->",
        "event": "COMMENT",
        "comments": [
            {**_comment_position(f), "body": comment_body(f, head_sha)}
            for f in findings
        ],
    }
    try:
        return gh_post(f"/repos/{repo}/pulls/{pr_number}/reviews", token, payload)
    except GitHubError as e:
        if e.code != 422 or not payload["comments"]:
            raise
        dropped = "\n".join(
            f"- `{f.path}:{f.line}` **[{f.severity}]** {f.title} — {f.detail[:300]}"
            for f in findings
        )
        fallback_body = (
            payload["body"]
            + "\n\n(Inline comment threads could not be posted: GitHub rejected a "
            "comment position. The findings, preserved:)\n\n" + dropped
        )
        return gh_post(
            f"/repos/{repo}/pulls/{pr_number}/reviews",
            token,
            {**payload, "comments": [], "body": fallback_body},
        )


def reply_in_thread(token: str, repo: str, pr_number: int, comment_id: int, body: str) -> dict:
    """Reply inside an existing review thread (REST in_reply_to — the reply
    endpoint takes the parent comment's REST id, not a GraphQL thread id)."""
    return gh_post(
        f"/repos/{repo}/pulls/{pr_number}/comments",
        token,
        {"body": body, "in_reply_to": comment_id},
    )


def resolve_thread(token: str, thread_id: str) -> dict:
    """Resolve a review thread via GraphQL. Field is ``threadId`` — not
    ``threadID`` — and it wants the GraphQL node id (fetch_open_own_threads
    supplies it), which is unrelated to the REST comment id."""
    return gh_graphql(
        "mutation($t: ID!) { resolveReviewThread(input: {threadId: $t}) "
        "{ thread { isResolved } } }",
        token,
        {"t": thread_id},
    )


def post_status(
    token: str,
    repo: str,
    sha: str,
    state: str,
    description: str,
    target_url: str | None = None,
) -> dict:
    """Post the `Wrasse Review` commit status on the PR head.

    state: success (review ran clean, findings or not) | error (harness or
    service failure — a GatewayError, a parse failure, a posting failure:
    anything where the review died before judging the code). Never
    "failure": the review never posts a code verdict as a status — bugs
    are threads, and "failure" would read in the checks list like the PR's
    code failed CI when only the harness did (the 2026-10-17 an earlier round 502:
    gateway queue-hop exhaustion, one transient blip, red X on the head).
    "error" is also correct for the future gate: the automation requires
    "success", so an error state still blocks a merge.
    """
    payload = {
        "state": state,
        "context": STATUS_CONTEXT,
        "description": description[:140],
    }
    if target_url:
        payload["target_url"] = target_url
    return gh_post(f"/repos/{repo}/statuses/{sha}", token, payload)
