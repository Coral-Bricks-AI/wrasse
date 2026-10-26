"""GitHub-side I/O: the PR diff, the files around it, and thread state.

All GitHub API calls for a review live here, so the provider-side compute is
the model call itself. The runner fetches the diff, the repository's agent
instructions (AGENTS.md / CLAUDE.md), and — through the head-pinned file API —
whatever the tools ask for. The provider never sees repo code beyond what is
in a prompt.

Every read is pinned to the immutable head sha rather than a branch name: a
branch can move between two API calls, and reviews must describe one commit.

Thread state (open, resolved, outdated) exists only through GraphQL — REST
sees a flat comment list — so the dedupe and the resolve pass both read it
there.

Stdlib only (urllib): the harness has no dependencies and needs no install
step on a stock runner.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from wrasse.identity import BOT_LOGINS, MARKER, SUMMARY_PREFIX

_API = "https://api.github.com"

# Files whose contents are given to the model as repo context, mirroring what
# the previous reviewer ingests (an earlier round v1). From the PR head, not main.
CONTEXT_PATHS = ("CLAUDE.md", "AGENTS.md")
_CONTEXT_MAX_CHARS = 20_000  # keep the prompt diff-dominated, not doc-dominated


class GitHubError(RuntimeError):
    def __init__(self, code: int, path: str, body: str):
        super().__init__(f"GitHub API {code} on {path}: {body[:300]}")
        self.code = code


def gh_get(path: str, token: str, accept: str = "application/vnd.github+json") -> bytes:
    req = urllib.request.Request(
        f"{_API}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": accept},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise GitHubError(e.code, path, e.read().decode("utf-8", "replace")) from None
    except urllib.error.URLError as e:
        raise GitHubError(0, path, str(e.reason)) from None


def gh_post(path: str, token: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{_API}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise GitHubError(e.code, path, e.read().decode("utf-8", "replace")) from None
    except urllib.error.URLError as e:
        raise GitHubError(0, path, str(e.reason)) from None


def fetch_pr_meta(repo: str, pr_number: int, token: str) -> dict:
    """Head sha + head ref for the PR (the review always reviews the head)."""
    payload = json.loads(gh_get(f"/repos/{repo}/pulls/{pr_number}", token))
    return {
        "head_sha": payload["head"]["sha"],
        "head_ref": payload["head"]["ref"],
    }


def fetch_pr_intent(repo: str, pr_number: int, token: str) -> dict:
    """The PR's own statement of intent: title + body. The review prompt
    carries it (an earlier round) so intended behavior is not reported as a bug.

    NOTE for replay use: GitHub returns the description as it stands NOW,
    not as it stood at review time — later edits are invisible. Descriptions
    are usually stable; the replay condition discloses the divergence."""
    payload = json.loads(gh_get(f"/repos/{repo}/pulls/{pr_number}", token))
    return {"title": payload.get("title") or "", "body": payload.get("body") or ""}


def fetch_diff(repo: str, pr_number: int, token: str) -> str:
    return gh_get(
        f"/repos/{repo}/pulls/{pr_number}", token, accept="application/vnd.github.v3.diff"
    ).decode("utf-8", "replace")


def fetch_file(repo: str, ref: str, path: str, token: str) -> str | None:
    """File contents at ``ref``, or None if absent (404). Small files only."""
    try:
        payload = json.loads(gh_get(f"/repos/{repo}/contents/{path}?ref={ref}", token))
    except GitHubError as e:
        if e.code == 404:
            return None
        raise
    if not isinstance(payload.get("content"), str):
        # GitHub omits content for files >1MB: distinguish that from a 404 with
        # a note instead of silently degrading the context (an earlier review round).
        size = payload.get("size")
        if isinstance(size, int):
            print(f"note: {path} at {ref} is {size} bytes (>1MB); skipping as context",
                  file=sys.stderr)
        return None  # directory, or >1MB
    return base64.b64decode(payload["content"]).decode("utf-8", "replace")


def fetch_dir_listing(repo: str, ref: str, path: str, token: str) -> list[dict] | None:
    """Directory entries at ``ref`` ({name, type} dicts), or None if the
    path is absent or not a directory. Direct children only, non-recursive
    — the agent's list_dir/grep tools are bounded by design (an earlier round
    v2: fetch-and-match, not repo-wide scans). GitHub caps a contents-API
    listing at 1000 entries; a larger directory comes back truncated and
    that truncation is surfaced by the caller, not silently dropped."""
    p = f"/repos/{repo}/contents/{path}?ref={ref}" if path else f"/repos/{repo}/contents/?ref={ref}"
    try:
        payload = json.loads(gh_get(p, token))
    except GitHubError as e:
        if e.code == 404:
            return None
        raise
    if not isinstance(payload, list):
        return None  # a file, not a directory
    return [
        {"name": e.get("name") or "", "type": e.get("type") or ""}
        for e in payload
        if isinstance(e, dict)
    ]


def fetch_tree_paths(repo: str, ref: str, token: str) -> tuple[list[str], bool]:
    """(all blob paths at ``ref``, GitHub's truncated flag), in ONE recursive
    git-trees call. Lets grep scope a whole directory subtree without the
    per-directory contents walk a search otherwise needs (an earlier round:
    find-usages greps measured 15+ calls on one head, probing directories
    one at a time). Sorted so scoping is deterministic; the truncated flag
    (>100k entries / >7MB responses) is returned for the caller to
    disclose, never silently dropped."""
    payload = json.loads(gh_get(f"/repos/{repo}/git/trees/{ref}?recursive=1", token))
    paths = sorted(
        e["path"] for e in payload.get("tree", [])
        if isinstance(e, dict) and e.get("type") == "blob"
    )
    return paths, bool(payload.get("truncated"))


def latest_status_state(token: str, repo: str, sha: str, context: str):
    """The state of ``sha``'s MOST RECENT status in ``context``, or None.

    Idempotence for the repair/skip paths: what matters is the LATEST
    status, not whether ANY historical success exists (an earlier review round: an
    older success must not mask a newer error — a head whose latest
    status is 'error' still needs repair even if an earlier run on it
    succeeded). 'pending' means another run on this head is in flight;
    callers leave the status alone rather than stomp it.
    """
    payload = json.loads(gh_get(f"/repos/{repo}/commits/{sha}/statuses?per_page=100", token))
    matches = [
        s for s in payload
        if isinstance(s, dict) and s.get("context") == context
    ]
    if not matches:
        return None
    latest = max(matches, key=lambda s: s.get("updated_at") or "")
    return latest.get("state")


def head_already_reviewed(token: str, repo: str, pr_number: int, head_sha: str,
                          *, require_agent: bool = False) -> bool:
    """True if this bot already posted a review pinned to this head.

    The summary body carries ``<!-- wrasse head=<sha[:10]> -->``
    (platforms/publish.py), so repeat events on an unchanged head — label churn is the
    common one — skip instead of re-posting duplicate threads (an earlier review round).
    Push a commit (new head) to force a re-review.

    The commit_id fallback matters once reviews can be MINIMIZED (the previous reviewer's
    own summaries are, and ours follow via minimize_resolved_summaries):
    a minimized review's REST body is replaced by GitHub's "Newer findings
    are available below" placeholder, which drops the marker — but the
    review object and its commit_id survive.

    ``require_agent`` (the v2 flip, an earlier review round): a v1 one-turn review on
    the head does not count — heads reviewed before the flip get their v2
    review instead of being locked out of it forever. The summary body
    names the mode ("agentic review (vX.Y)"), so the guard matches the
    header's mode word — anchored to the body's first line, immune to
    finding prose — when asked to. A MINIMIZED agentic review counts as
    not reviewed in agent mode (its mode line died with the body): one
    redundant v2 re-review is cheaper than a head that can never be
    v2-reviewed.
    """
    marker = f"head={head_sha[:10]}"
    # The head marker gained a version field (REVIEW_VERSION, an earlier round): new
    # summaries end the marker with " v=<version> -->", legacy ones with
    # " -->". Either pins the head — the 10-char sha followed by
    # whitespace is unambiguous.
    for page in range(1, 21):  # hard stop at 2000 reviews
        payload = json.loads(gh_get(
            f"/repos/{repo}/pulls/{pr_number}/reviews?per_page=100&page={page}", token))
        for rev in payload:
            if not isinstance(rev, dict):
                continue
            user = (rev.get("user") or {}).get("login")
            if user not in BOT_LOGINS:
                continue
            body = rev.get("body") or ""
            if not (marker + " -->" in body or marker + " v=" in body
                    or rev.get("commit_id") == head_sha):
                continue
            # Mode word, never version digits: the summary's FIRST LINE is
            # the header ("**Wrasse Review** — agentic review (vX.Y) via …"),
            # so an anchored startswith is immune to a one-turn review
            # whose FINDINGS prose happens to mention "agentic review"
            # (an earlier review round) — and to any version spelling.
            if not require_agent or body.startswith(
                    f"{SUMMARY_PREFIX} — agentic review"):
                return True
        if len(payload) < 100:
            return False
    return False


def fetch_path_history(repo: str, path: str, token: str, at_ref: str,
                       per_page: int = 10) -> list[dict]:
    """Recent commits that touched ``path`` **reachable from ``at_ref``**
    ({sha, date, message}), newest first, bounded. The reviewer's
    when/why-did-this-line-appear tool (an earlier round): long-standing code is
    presumptively intended behavior — a diff-introduced bug is the thing to
    report — and the commit that introduced a line is the cheapest way to
    tell them apart.

    ``at_ref`` pins the history to the reviewed commit (an earlier review round):
    without it the commits API walks the DEFAULT branch, which for a
    replayed head shows later commits and drops branch-only ones — history
    of a different revision than the one under review. The path is
    urlencoded because repo paths contain spaces and ampersands
    (an earlier review round: "docs/Release Notes.md" is a real path here)."""
    query = urllib.parse.urlencode({"path": path, "sha": at_ref, "per_page": per_page})
    payload = json.loads(gh_get(f"/repos/{repo}/commits?{query}", token))
    if not isinstance(payload, list):
        return []
    return [
        {"sha": c.get("sha") or "",
         "date": ((c.get("commit") or {}).get("author") or {}).get("date") or "",
         "message": (c.get("commit") or {}).get("message") or ""}
        for c in payload if isinstance(c, dict)
    ]


def fetch_repo_context(repo: str, head_ref: str, token: str) -> str:
    """Concatenate CLAUDE.md/AGENTS.md at the PR head, truncated per file."""
    chunks: list[str] = []
    for path in CONTEXT_PATHS:
        text = fetch_file(repo, head_ref, path, token)
        if text:
            chunks.append(f"### {path}\n\n{text[:_CONTEXT_MAX_CHARS]}")
    return "\n\n".join(chunks)


def gh_graphql(query: str, token: str, variables: dict | None = None) -> dict:
    """POST one GraphQL request. Review-thread state (isResolved/isOutdated)
    and the thread id that resolveReviewThread needs only exist via GraphQL —
    REST sees flat comments. Errors reuse GitHubError like the REST helpers;
    GraphQL reports failures as HTTP 200 with an "errors" key, so that key
    is checked too, or a query typo would look like an empty result."""
    result = gh_post("/graphql", token, {"query": query, "variables": variables or {}})
    if isinstance(result, dict) and result.get("errors"):
        raise GitHubError(200, "/graphql", json.dumps(result["errors"]))
    return result


def _own_thread_record(node: dict, *, resolved: bool) -> dict | None:
    """One own-marker thread record from a GraphQL reviewThread node, or
    None when it is not one of ours (shared shape for the open-thread and
    resolved-history fetchers). Open records keep line exactly as the
    GitHub comment gave it (None when outdated, as before); resolved
    records fall back to originalLine so the history names the position
    the finding was raised at."""
    comments = node.get("comments", {}).get("nodes", [])
    if not comments:
        return None
    first = comments[0]
    # GraphQL author logins lack the "[bot]" suffix REST has — a
    # filter on the REST name silently skips every thread.
    if (first.get("author") or {}).get("login") not in BOT_LOGINS:
        return None
    if f"<!-- {MARKER} " not in (first.get("body") or ""):
        return None
    title = next(
        (ln for ln in (first.get("body") or "").splitlines()
         if ln.strip() and not ln.lstrip().startswith("<!--")),
        "",
    )
    # Marker fid (an earlier review round): the hidden marker carries the finding's
    # identity (sha1 of path+line+title at raise time) — the only reliable
    # same-claim key. A re-derivation drifts phrasing and can move a line,
    # but the fid pins what was actually raised.
    fid = ""
    m = re.search(rf"<!-- {MARKER} id=(\w+)", first.get("body") or "")
    if m:
        fid = m[1]
    # Marker end_line (an earlier round adjacent-line survivor): the hidden marker
    # spans multi-line findings, so thread spans come from what WE wrote,
    # not just GitHub's line fields. Old markers predate the field —
    # their span is the single raised line.
    # Marker span (an earlier round adjacent-line survivor, an earlier review round): the
    # marker is the only same-revision record of a multi-line finding's
    # START — GitHub's `line` on a range comment is the range END, so
    # without marker_start a re-flag at the original start line would
    # post a duplicate thread. Old markers predate the fields — their
    # span is the single raised position.
    marker_start = marker_end = None
    body = first.get("body") or ""
    m = re.search(rf"<!-- {MARKER} [^>]*\bstart_line=(\d+)", body)
    if m:
        marker_start = int(m[1])
    m = re.search(rf"<!-- {MARKER} [^>]*\bend_line=(\d+)", body)
    if m:
        marker_end = int(m[1])
    # Marker sha: the head the finding was raised on — groups a review
    # round's threads under the summary that posted them, so a summary
    # whose round's threads are all resolved can be minimized (the previous reviewer does
    # the same to its own reviews).
    sha = ""
    m = re.search(rf"<!-- {MARKER} [^>]*\bsha=(\w+)", body)
    if m:
        sha = m[1]
    rec = {
        "thread_id": node["id"],
        "comment_id": first["databaseId"],
        "path": first.get("path") or "",
        "line": first.get("line"),
        "start_line": first.get("startLine"),
        "original_line": first.get("originalLine"),
        "original_start_line": first.get("originalStartLine"),
        "marker_start": marker_start,
        "marker_end": marker_end,
        "sha": sha,
        "outdated": bool(node.get("isOutdated")),
        "created_at": first.get("createdAt") or "",
        "title": title.strip().strip("*").strip(),
        "fid": fid,
    }
    if resolved:
        # Resolved-and-outdated threads have line=None; the history entry
        # must still name the position it was raised at.
        if rec["line"] is None:
            rec["line"] = first.get("originalLine")
        # The resolution a reader sees on the thread: the first content
        # line of the thread's TRUE last comment (lastComment connection —
        # comments[-1] of the first-10 truncates long threads, an earlier review round
        # r1). Marker lines are skipped like in the title extraction;
        # bounded to one line. A thread resolved with no reply at all has
        # no explanation to quote — resolution stays empty rather than
        # echoing the finding's own body back as if it were a resolution.
        last_nodes = (node.get("lastComment") or {}).get("nodes") or []
        if not last_nodes:
            # lastComment missing from the response: fall back to the
            # first-10 connection ONLY when the connection itself says
            # it is complete (pageInfo.hasNextPage false). A node count
            # cannot tell a truncated prefix from an exactly-10-comment
            # thread (an earlier review round); quoting a possibly-truncated
            # [-1] would fabricate a resolution — the exact thing the
            # alias exists to prevent.
            conn = node.get("comments") or {}
            # Positive assertion (an earlier review round, round on 24349e391):
            # a MISSING pageInfo must read as "unknown completeness", not
            # "complete" — a partial payload that omits it would otherwise
            # fabricate a resolution from a possibly-truncated prefix.
            page = conn.get("pageInfo")
            complete = isinstance(page, dict) and not page.get("hasNextPage")
            last_nodes = comments if complete else []
        last = last_nodes[-1] if last_nodes else None
        if last is not None and last.get("databaseId") != first.get("databaseId"):
            resolution = next(
                (ln for ln in (last.get("body") or "").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("<!--")),
                "",
            )
            rec["resolution"] = resolution.strip().strip("*").strip()[:200]
        else:
            rec["resolution"] = ""
    return rec


def _fetch_own_threads(
    token: str, repo: str, pr_number: int, *, resolved: bool
) -> tuple[list[dict], bool]:
    """The PR's wrasse marker threads filtered by resolution state,
    via GraphQL (thread state lives there). Shared query body for the two
    public fetchers below; hard stop at 200 threads, same bound as before.

    Returns ``(records, complete)``: ``complete`` is False when the hard
    stop cut the walk short — the caller decides what partial data may
    do. The dedupe/history consumers keep the old behavior (a missed
    thread costs one duplicate finding, which the next run cleans up);
    the summary-minimization pass must NOT guess on partial data — an
    unseen OPEN thread would read as resolved and minimize a live round's
    summary (an earlier review round)."""
    owner, _, name = repo.partition("/")
    # The comments connection fetches pageInfo because a full page of 10
    # nodes may be the WHOLE thread (an earlier review round): its hasNextPage —
    # not a node count — is the completeness signal the resolution
    # fallback consumes in _own_thread_record.
    query = """query($owner: String!, $name: String!, $n: Int!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $n) {
          reviewThreads(first: 50, after: $cursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              isResolved
              isOutdated
              comments(first: 10) {
                nodes {
                  databaseId
                  author { login }
                  body
                  path
                  line
                  startLine
                  originalLine
                  originalStartLine
                  createdAt
                }
                pageInfo { hasNextPage }
              }
              # The TRUE last comment (an earlier review round): comments(first: 10)
              # truncates long threads, and the resolving reply — the one
              # a reader sees as the thread's conclusion — is the one a
              # >10-comment thread would cut off. An aliased last: 1
              # connection fetches it in the same request; the first-10
              # connection still supplies marker, title and position.
              lastComment: comments(last: 1) {
                nodes {
                  databaseId
                  author { login }
                  body
                  path
                  line
                  originalLine
                  createdAt
                }
              }
            }
          }
        }
      }
    }"""
    out: list[dict] = []
    cursor = None
    complete = True
    for _ in range(4):  # hard stop at 200 threads
        data = gh_graphql(query, token, {
            "owner": owner, "name": name, "n": pr_number, "cursor": cursor})
        threads = (data.get("data", {}).get("repository", {})
                   .get("pullRequest", {}).get("reviewThreads", {}))
        for node in threads.get("nodes", []):
            if bool(node.get("isResolved")) != resolved:
                continue
            rec = _own_thread_record(node, resolved=resolved)
            if rec is not None:
                out.append(rec)
        page = threads.get("pageInfo", {})
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
    else:
        complete = False  # the hard stop cut the walk short
    return out, complete


def fetch_open_own_threads(token: str, repo: str, pr_number: int) -> list[dict]:
    """The PR's OPEN review threads authored by this bot whose first comment
    carries the wrasse marker (dedupe + resolve on re-review, an earlier round 2026-10-15).

    Each record: thread_id (GraphQL node id, for resolveReviewThread),
    comment_id (REST id, for in_reply_to), path, line (None when outdated —
    use original_line then), outdated, created_at, title (first non-marker
    line of the body)."""
    return _fetch_own_threads(token, repo, pr_number, resolved=False)[0]


def fetch_resolved_own_threads(token: str, repo: str, pr_number: int) -> list[dict]:
    """The PR's RESOLVED wrasse marker threads, for the first-turn
    history (an earlier round pre-A/B): a reviewer that cannot see what it
    already raised and addressed re-derives the same shapes every round —
    on an earlier round the last two v1 rounds posted 17 findings, zero new, mostly
    exact repeats of threads already resolved. Records match the open-thread
    shape plus ``resolution`` (first content line of the thread's last
    comment). Prompt context only — the hard dedupe
    (findings.split_already_raised) still sees OPEN threads only, so a
    genuinely re-introduced bug at a resolved position stays reportable."""
    return _fetch_own_threads(token, repo, pr_number, resolved=True)[0]


def fetch_thread_state(
    token: str, repo: str, pr_number: int
) -> tuple[list[dict], list[dict], bool]:
    """(open_threads, resolved_threads, complete) in one place, for the
    caller that must not guess on partial thread data: the summary-
    minimization pass (an earlier review round). The dedupe/history paths keep
    the older single-list fetchers — their failure mode is a duplicate
    finding, not a wrongly minimized live round."""
    open_threads, open_complete = _fetch_own_threads(
        token, repo, pr_number, resolved=False)
    resolved_threads, resolved_complete = _fetch_own_threads(
        token, repo, pr_number, resolved=True)
    return open_threads, resolved_threads, open_complete and resolved_complete


def fetch_own_summaries(
    token: str, repo: str, pr_number: int
) -> tuple[list[dict], bool]:
    """The PR's review objects authored by this bot, for minimization.

    Each record: node_id (GraphQL id, for minimizeComment), sha (the
    head marker `<!-- wrasse head=<sha[:10]> -->`, "" when the body
    has none — e.g. already minimized, where GitHub replaces the body
    with its placeholder), minimized (bool).

    Returns ``(records, complete)``: paginated to the end of the reviews
    connection with a hard stop at 500 (an earlier review round — the first
    100-review page left later summaries permanently unminimized on
    long-lived PRs), and ``complete`` False at the hard stop so the
    minimize pass can refuse to guess instead of silently skipping the
    tail.
    """
    owner, _, name = repo.partition("/")
    query = """query($owner: String!, $name: String!, $n: Int!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $n) {
          reviews(first: 100, after: $cursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              state
              isMinimized
              body
              author { login }
            }
          }
        }
      }
    }"""
    out: list[dict] = []
    cursor = None
    complete = True
    for _ in range(5):  # hard stop at 500 reviews
        data = gh_graphql(query, token, {
            "owner": owner, "name": name, "n": pr_number, "cursor": cursor})
        reviews = (data.get("data", {}).get("repository", {})
                   .get("pullRequest", {}).get("reviews", {}))
        for node in reviews.get("nodes", []) or []:
            if (node.get("author") or {}).get("login") not in BOT_LOGINS:
                continue  # GraphQL logins lack the [bot] suffix (see above)
            body = node.get("body") or ""
            m = re.search(rf"<!-- {MARKER} head=(\w+)", body)
            out.append({
                "node_id": node["id"],
                "sha": m[1] if m else "",
                "minimized": bool(node.get("isMinimized")),
                # The mode word, anchored to the header (same immunity rule
                # as head_already_reviewed): an agentic summary IS the
                # head's agent-mode guard record and must never minimize.
                "agent": body.startswith(f"{SUMMARY_PREFIX} — agentic review"),
            })
        page = reviews.get("pageInfo", {})
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
    else:
        complete = False  # the hard stop cut the walk short
    return out, complete


def minimize_comment(token: str, node_id: str) -> bool:
    """Minimize one node (PullRequestReview or comment) as RESOLVED.

    the previous reviewer does this to its own review summaries once their threads are
    resolved (an earlier round reviews are isMinimized=true, reason="resolved");
    minimizeComment accepts PullRequestReview nodes, and the workflow token
    has write access. Returns True when GitHub reports it minimized."""
    mutation = """mutation($id: ID!) {
      minimizeComment(input: {subjectId: $id, classifier: RESOLVED}) {
        minimizedComment { isMinimized }
      }
    }"""
    data = gh_graphql(mutation, token, {"id": node_id})
    node = (data.get("data", {}).get("minimizeComment", {})
            .get("minimizedComment") or {})
    return bool(node.get("isMinimized"))


def summaries_to_minimize(
    summaries: list[dict],
    open_threads: list[dict],
    resolved_threads: list[dict],
) -> list[dict]:
    """Which own review summaries should be minimized.

    A summary goes when its round raised at least one thread and EVERY one
    is resolved: the round is over — its summary is noise, and minimizing
    shrinks the PR's visible review wall (the previous reviewer's own pattern: an earlier round
    reviews are isMinimized=true, reason="resolved"). Rules, conservative:

    - A summary whose round has any OPEN thread stays: live conversation.
    - A clean-review summary (round raised NO threads at all) stays: it
      IS the head's reviewed-record marker, the cheapest signal
      head_already_reviewed has.
    - Already-minimized summaries are not re-asked for.
    - An AGENTIC summary never minimizes (first live v2 review, an earlier round):
      it is the head's agent-mode guard record — minimizing replaces its
      body with GitHub's placeholder, the mode word dies with it, and
      head_already_reviewed(require_agent=True) would count the head as
      never v2-reviewed — an unbounded re-review loop, one full agentic
      review per repeat event. One-turn summaries minimize freely; when
      the mode becomes persistable outside the body (e.g. in the commit
      status), agentic summaries can join them.
    """
    open_shas = {t.get("sha") for t in open_threads if t.get("sha")}
    raised_shas = open_shas | {
        t.get("sha") for t in resolved_threads if t.get("sha")}
    return [
        s for s in summaries
        if not s["minimized"] and s["sha"] and not s.get("agent")
        and s["sha"] in raised_shas and s["sha"] not in open_shas
    ]
