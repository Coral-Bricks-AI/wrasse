"""Findings contract: prompt, parser, validation (an earlier round v1).

Bugs only. The system prompt fixes the scope, the parser drops anything that
does not carry the required fields, and every accepted finding gets a stable
id so later tooling can resolve/track threads by parsing the hidden marker in
the posted comment (see platforms/publish.py).

Output contract given to the model: a single ```json fenced block containing
a JSON array of objects:

    {"path": str, "line": int, "end_line": int (optional), "severity":
     "high"|"medium"|"low", "title": str, "detail": str}

``line`` is 1-based in the file AFTER the change (new side), so it maps
straight onto the diff hunks parsed in core/diff.py. ``end_line``, when
the model emits it, spans the changed lines the bug covers (>= line) —
thread dedupe matches on range overlap, not just the exact start line,
so the same block flagged at neighboring lines dedupes (an earlier round adjacent
survivor).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from dataclasses import dataclass, field

SYSTEM_PROMPT = """\
You are a meticulous code reviewer. You review code changes (diffs) and report
ONLY actual bugs: logic errors, off-by-one, inverted conditions, wrong
variable, unhandled error paths, race conditions, resource leaks, security
issues, data corruption, crashes.

Hard rules:
- Report bugs introduced or exposed by this diff, not pre-existing issues the
  diff does not touch.
- Trace every changed value to where it is assigned and to every place it
  is consumed: read the surrounding code, not just the changed lines,
  before concluding.
- Before dismissing a suspected issue, identify the factual claims your
  dismissal depends on and verify each by reading the code. If the
  verification shows a claim false, report the finding rather than dismiss
  it. If the evidence you need could not be read at all (a read failed or
  came back truncated), do not report the suspicion either — an unverified
  guess is not a finding, in either direction.
- A change that breaks a documented workflow in the repository's AGENTS.md
  or CLAUDE.md (supplied as repository context below) is a finding — quote
  the rule it violates.
- Do NOT report style, naming, formatting, missing tests, documentation,
  performance micro-optimizations, or refactoring suggestions. Bugs only.
- If you are not confident something is a real bug, do not report it.
- Every finding must be concrete: what breaks, under what input or state.
- Before emitting your findings, re-walk the diff hunk by hunk and confirm
  every changed line was considered — do not stop at the first finding.
- Each array entry must be a bug you are affirmatively reporting. NEVER
  include an entry whose own reasoning retracts it — no "this is actually
  correct", "no bug", or "not reporting" entries. Finish your analysis
  first, then emit only the findings that survived it.

Output format — exactly one ```json fenced block, a JSON array, no prose
before or after it:

```json
[
  {
    "path": "<file path as it appears in the diff>",
    "line": <1-based line number in the file AFTER the change>,
    "end_line": <optional: last line of the buggy span, same file, >= line;
                 the changed lines the bug spans when it covers a block.
                 Omit it for a single-line finding>,
    "severity": "high" | "medium" | "low",
    "title": "<one-line summary>",
    "detail": "<why this is a bug, what breaks, and the concrete fix>"
  }
]
```

Severity scale: "high" = data loss or corruption, security exposure, a
crash, or user-visible breakage on a common path; "medium" = wrong
behavior on an edge case or under specific states; "low" = latent
defect, hard to trigger, or limited blast radius. When unsure between
two levels, pick the lower one.

If the diff introduces no bugs, output an empty array: []
"""

# Path filters: findings on generated / vendored / bulk-data paths are dropped
# rather than reported — they are noise, and commenting on them trains people
# to ignore the bot (graceful degradation, an earlier round v1).
_SKIP_EXACT = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "uv.lock", "cargo.lock", "go.sum", "flake.lock", "composer.lock",
    "Gemfile.lock",
}
_SKIP_SUFFIX = (".lock", ".min.js", ".min.css", ".map")
_SKIP_PREFIX = ("datasets/", "node_modules/", ".venv/", "graphify-out/")

_SEVERITY = {"high", "medium", "low"}
_SEVERITY_ALIASES = {"critical": "high", "blocker": "high", "info": "low", "minor": "low"}
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}  # for cap truncation + dedup preference
# PR description cap (an earlier round): the intent section must stay a guard, not
# a second diff — the prompt stays diff-dominated.
PR_INTENT_MAX_CHARS = 4000
_FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

# Self-retraction verdicts (an earlier review round/an earlier round, an earlier round pre-A/B): the model
# sometimes emits an array entry whose own text concludes the "finding" is
# not one — bodies ending "No bug.", "…but tests pass", "Not reporting.",
# "(Not reporting.)", "no bug here." — and the post path ran anyway (on
# an earlier round, 15 of one round's 25 findings; on an earlier round, 10+ across rounds). Nothing
# dropped them, which inflates the false-positive count the A/B
# pre-registered as a pass bar ("no more false positives per PR") with a
# known harness artifact.
#
# The guard anchors on the VERDICT POSITION, not a whole-text grep, because
# a false drop of a real finding is worse than the disease (an earlier round) —
# an earlier round has a real finding whose body says "…still matches. No bug there.
# The actual defect: …", so a retraction phrase ANYWHERE but the final
# verdict segment proves nothing. A phrase counts only when it sits in the
# LAST segment of the text with at most a short residue after it: "No bug
# here." retracts ("here" is filler); "The assertion is inverted; tests
# pass even when the handler is broken." does not (the clause after
# "tests pass" is the point of the finding).
#
# Two phrase classes (an earlier review round, round on 538c2d774). STRONG phrases are
# unambiguous withdrawals ("no bug", "not reporting", "false positive")
# and keep the end-position + filler-residue rule. WEAK phrases ("is/as
# intended", "tests pass", "by design") read as verdicts only inside a
# verdict FRAME, because real findings end with them as CONCLUSIONS TOO:
# "handle() returns a str where an int is intended.", "…so tests pass."
# after a stated cause. A weak phrase counts only when the final segment
# OPENS with the verdict ("the behavior is intended", "this is by
# design", "works as intended") or the phrase is preceded by a contrast
# "but" ("…, but tests pass") — never by "so" (consequence) or by the
# finding's own subject mid-sentence.
_STRONG_RETRACTION_RE = re.compile(
    r"\b(?:no bug|not a bug|isn'?t a bug|no defect|not a defect|no issue"
    r"|not an issue|not reporting|no actual bug|not a real bug"
    r"|false positive)\b",
    re.I,
)
_WEAK_RETRACTION_RE = re.compile(
    r"\b(?:by design|working as intended|works as intended|as intended"
    r"|is intended|are intended|tests pass)\b",
    re.I,
)
# The frame a weak phrase must sit in, at the START of the final segment:
# an optional subject ("this", "the behavior", …), an optional copula,
# then the verdict — or the contrast form "but tests pass".
_WEAK_VERDICT_START_RE = re.compile(
    r"^(?:(?:this|that|it|the code|the behavior|they) )?"
    r"(?:(?:is |are |was )?(?:intended|by design)"
    r"|(?:is |are |was )?(?:works |working )?as intended"
    r"|tests pass)"
    r"|^but tests pass\b",
    re.I,
)
# Sentence/segment terminators. An em-dash and a colon separate too: the
# observed retraction shapes append their verdict after one ("— no bug",
# "Consistent. Not a bug.").
_SEGMENT_SPLIT_RE = re.compile(r"(?<=[.!?;:\u2014])\s+")
# Not a raw string: the \uXXXX escapes must stay escapes. Only `\]` was
# invalid (SyntaxWarning on the 39070307732 run) — `\\]` is the same two
# characters Python already passed through.
_TRAILING_PUNCT = " \t\n\r.!?;:,)\\]}'\"\u2014\u2019\u201d"
# What may follow a retraction phrase in the final segment and still make
# it a verdict (the previous reviewer + an earlier review round/r2, an earlier review round): pure
# deictic filler, or an "as a <bug-noun>" qualifier naming what is not
# being reported. A LENGTH bound was tried first and is wrong — "No bug in
# tests." (the previous reviewer) and "…tests pass, prod breaks." (the review) are real
# findings whose short tails are the point, not filler; a residue that
# SAYS anything keeps the finding. The "as a" qualifier is similarly
# narrowed: "as a side effect while prod crashes" and "as a fallback but
# it double-charges the user" are consequences, not verdicts — only an
# explicit bug-noun counts (an earlier review round). Conservative by construction:
# unknown residue = keep.
_VERDICT_RESIDUE_RE = re.compile(
    r"^(?:here|there|now|at all|this one|in this (?:file|case)"
    r"|as an? (?:(?:definite|real|actual|separate|new|potential|possible) )?"
    r"(?:bug|issue|defect|finding|problem))?$",
    re.I,
)


@dataclass
class Finding:
    path: str
    line: int
    severity: str
    title: str
    detail: str = ""
    end_line: int | None = None  # last line of the span; None = single line
    fid: str = field(init=False)

    def __post_init__(self) -> None:
        self.end_line = self._norm_end_line(self.end_line, self.line)
        self.fid = self._compute_fid()

    @staticmethod
    def _norm_end_line(end_line: int | None, line: int | None = None) -> int | None:
        # None stays None (single line); anything below `line` collapses to
        # a single line — an inverted span is a contract slip, not a range.
        if end_line is None:
            return None
        if line is not None and end_line < line:
            return None
        return int(end_line)

    def span(self) -> tuple[int, int]:
        """Inclusive (start, end) line span — dedupe matches on overlap."""
        return (self.line, self.end_line if self.end_line is not None else self.line)

    def _compute_fid(self) -> str:
        # Stable across runs so the same bug on the same head maps to the
        # same thread marker; not a function of detail prose (which drifts).
        return hashlib.sha1(
            f"{self.path}\n{self.line}\n{self.title.strip().lower()}".encode("utf-8")
        ).hexdigest()[:10]

    def rekey(self) -> None:
        """Recompute fid after path normalization (an earlier review round): `x.py` and
        header-form `b/x.py` for the same bug must share one marker id, or
        the review posts two threads for one bug."""
        self.fid = self._compute_fid()


def _claim_similarity(a: str, b: str) -> float:
    """Difflib ratio on normalized claim text: case-folded, punctuation
    and marker decoration stripped. A re-derivation rephrases; a different
    bug at the same coordinate states a different claim."""
    def norm(s: str) -> str:
        s = s.strip().strip("*").strip().lower()
        return "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


# Above this, a coordinate-matching finding is the same claim rephrased
# (withheld); below it, a different claim at a reused line — POSTED, not
# swallowed (an earlier review round: six distinct claims lived at one line on an earlier round, so
# coordinates alone cannot decide).
REPHRASE_RATIO = 0.75


def thread_spans(t: dict) -> list[tuple[int, int]]:
    """Inclusive (start, end) line spans a thread record covers.

    Two anchors, never merged (an earlier review round: min/max across them
    fabricates a span covering every line a moved thread skipped past):

    - The raise-time span: the marker the thread was posted with
      (start_line..end_line), else GitHub's original coordinates
      (original_start_line..original_line) — the same revision the
      finding was raised on. For pre-span single-line markers it is the
      single raised position.
    - The thread's CURRENT span from GitHub's live coordinates —
      (start_line..line) for a range comment, (line..line) otherwise.
      Without the live START, a moved range's interior lines would be
      uncovered and an interior re-flag would post a duplicate thread
      (an earlier review round BUG-1).

    A record with no line at all has no spans."""
    spans: list[tuple[int, int]] = []
    # Raise-time span: marker first, else original coordinates.
    start, end = t.get("marker_start"), t.get("marker_end")
    if start is None:
        start = t.get("original_start_line")
        end = t.get("original_line")
    if start is None:
        start = end = t.get("original_line")  # pre-span marker, unmoved
    if start is None:
        start = end = t.get("line")  # last resort: current position
    if start is not None:
        start = int(start)
        end = int(end) if end is not None and int(end) >= start else start
        spans.append((start, end))
    # Current span: GitHub's live coordinates for the same comment.
    cur_start, cur = t.get("start_line"), t.get("line")
    if cur is not None:
        lo = int(cur_start) if cur_start is not None else int(cur)
        cur_span = (min(lo, int(cur)), int(cur))
        if not any(s <= cur_span[0] and cur_span[1] <= e for s, e in spans):
            spans.append(cur_span)
    return spans


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _thread_covers(t: dict, span: tuple[int, int]) -> bool:
    return any(_spans_overlap(ts, span) for ts in thread_spans(t))


def format_thread_position(t: dict) -> str:
    """One-line position rendering of a thread record for the prompt
    lists (already-raised, resolved history): the raise-time span, plus
    the thread's CURRENT line when it differs — the dedupe
    (thread_spans) withholds on BOTH anchors, so the prompt must list
    both or the model gets re-derivations silently withheld at a
    position it was never told about (an earlier review round)."""
    path = t.get("path") or "?"
    start = t.get("marker_start")
    if start is None:
        start = t.get("original_start_line")
    if start is None:
        start = t.get("original_line", t.get("line"))
    out = f"{path}:{start}"
    if t.get("marker_end") and int(t["marker_end"]) != start:
        out += f"-{t['marker_end']}"
    cur = t.get("line")
    if cur is not None and start is not None and int(cur) != int(start):
        out += f" (now at {path}:{int(cur)})"
    return out


def split_resolved_rederivations(
    findings: list[Finding],
    resolved_history: list[dict] | None,
) -> tuple[list[Finding], list[Finding]]:
    """(fresh, withheld) — findings that re-derive a thread already raised
    AND resolved on this PR, two-tier (an earlier round, an earlier round data + an earlier review round's reused-line concern):

    Tier 1, exact identity: the resolved thread's marker fid (sha1 of
    path+line+title at raise time) equals the finding's fid — the same
    claim verbatim. Withheld.
    Tier 2, span + rephrase: the finding's (line..end_line) span
    overlaps the resolved thread's span (raised-at, current, or marker
    end_line) AND the title similarity is at or above REPHRASE_RATIO —
    the same claim reworded (an earlier round round-4 repeats). Withheld.
    Everything else passes — including a different claim at a reused
    coordinate, which becomes a real thread: a reused line must not
    swallow an unrelated bug, and a reintroduction must stay actionable.

    Withheld findings are counted and listed in the review summary (a
    human still sees them), never posted as new threads — so a
    re-derivation cannot block a merge after the gate flip. Replay mode
    has no thread state (empty history): a live-review guard, a no-op for
    the A/B rows."""
    by_fid: set[str] = set()
    threads: list[dict] = []
    for t in resolved_history or []:
        if t.get("fid"):
            by_fid.add(t["fid"])
        if thread_spans(t):
            threads.append(t)
    fresh, withheld = [], []
    for f in findings:
        if f.fid in by_fid:
            withheld.append(f)
            continue
        p = f.path.strip()
        if any((t.get("path") or "").strip() == p
               and _thread_covers(t, f.span())
               and _claim_similarity(f.title, t.get("title") or "") >= REPHRASE_RATIO
               for t in threads):
            withheld.append(f)
        else:
            fresh.append(f)
    return fresh, withheld


def keep_path(path: str) -> bool:
    """False for generated / vendored / bulk-data paths (see _SKIP_* lists)."""
    if path in _SKIP_EXACT:
        return False
    if path.endswith(_SKIP_SUFFIX):
        return False
    return not path.startswith(_SKIP_PREFIX)


def _retraction_at_verdict(text: str) -> bool:
    """True when the text's own conclusion retracts the finding.

    A retraction phrase counts only in the LAST segment of the text, and
    only when what follows it is verdict filler (_VERDICT_RESIDUE_RE) —
    the position where the model states its conclusion. Everywhere else
    the phrase is reasoning, not conclusion, and an earlier round has a real finding
    whose body literally contains "No bug there. The actual defect: …",
    so a broader grep would drop real findings. A residue that says
    anything ("in tests", "it crashes", "prod breaks") keeps the finding:
    unknown residue is preserved, not guessed at (the previous reviewer + an earlier review round).
    """
    if not text or not text.strip():
        return False
    segments = [s for s in _SEGMENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if not segments:
        return False
    last = segments[-1]
    for m in _STRONG_RETRACTION_RE.finditer(last):
        residue = last[m.end():].strip(_TRAILING_PUNCT)
        if _VERDICT_RESIDUE_RE.match(residue):
            return True
    for m in _WEAK_RETRACTION_RE.finditer(last):
        residue = last[m.end():].strip(_TRAILING_PUNCT)
        if residue:
            continue  # a weak phrase with a tail is the finding, not a verdict
        before = last[:m.start()].rstrip(" ,;").lower()
        if before.endswith("but") or _WEAK_VERDICT_START_RE.match(last.lstrip()):
            return True
    return False


def drop_self_retracted(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """(kept, dropped) — partition out findings whose own title or detail
    retracts them (_retraction_at_verdict). Parse-time enforcement of the
    prompt rule that was prose-only until now: "NEVER include an entry
    whose own reasoning retracts it". Shared by v1 and v2 (both post
    through parse_findings + this guard in the CLI), so neither arm's
    false-positive count carries the artifact into the A/B.
    """
    kept: list[Finding] = []
    dropped: list[Finding] = []
    for f in findings:
        if _retraction_at_verdict(f.title) or _retraction_at_verdict(f.detail):
            dropped.append(f)
        else:
            kept.append(f)
    return kept, dropped


def _coerce_finding(entry: object) -> Finding | None:
    """Validate one parsed array entry; None if it is not a usable finding."""
    if not isinstance(entry, dict):
        return None
    path = entry.get("path")
    line = entry.get("line")
    title = entry.get("title")
    if not isinstance(path, str) or not path.strip():
        return None
    if isinstance(line, bool) or not isinstance(line, (int, float)) or line < 1:
        return None
    if not isinstance(title, str) or not title.strip():
        return None
    severity = str(entry.get("severity") or "").strip().lower()
    severity = _SEVERITY_ALIASES.get(severity, severity)
    if severity not in _SEVERITY:
        severity = "medium"
    detail = entry.get("detail")
    if not isinstance(detail, str):
        detail = ""
    end_line = entry.get("end_line")
    if isinstance(end_line, bool) or not isinstance(end_line, (int, float)) \
            or (isinstance(end_line, float) and not math.isfinite(end_line)):
        # absent, garbage, or non-finite float (1e999 -> inf): single-line.
        # isfinite applies to FLOATS only: a huge integer is finite, and
        # math.isfinite converts its argument to float — on a large int
        # that conversion itself raises OverflowError (an earlier review round).
        end_line = None
    else:
        end_line = int(end_line)
    f = Finding(
        path=path.strip(),
        line=int(line),
        severity=severity,
        title=title.strip(),
        detail=detail.strip(),
        end_line=end_line,
    )
    return f


def _candidate_arrays(text: str):
    """JSON array candidates, best first: fenced blocks, then one bare array.

    Fenced blocks are tried last-first (the final answer). For the lenient
    fallback — the model dropped the fence — scan every '[' and try
    raw_decode from it, keeping the LAST span that decodes to a NON-EMPTY
    LIST OF DICTS: prose brackets like a citation "[1]" or a markdown
    "[link]" must not shadow the real findings array (an earlier review round),
    and must not be mistaken for one either — a reply whose only bracket
    list is a citation is inconclusive (None), not "no bugs" ([]).
    """
    for m in reversed(_FENCED_RE.findall(text)):
        yield m
    decoder = json.JSONDecoder()
    best_dicts = None
    for match in list(re.finditer(r"\[", text))[:50]:
        try:
            obj, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and obj and all(isinstance(e, dict) for e in obj):
            best_dicts = text[match.start(): match.start() + end]  # later spans win
    yield best_dicts


def parse_findings(text: str) -> list[Finding] | None:
    """Parse model output into validated, deduplicated findings.

    Return value distinguishes the two failure-like cases (an earlier review round:
    a truncated reply must not masquerade as a clean review):

    - ``[]``  — the model answered the contract with an empty array: a
      valid "no bugs" reply. This includes an *entire* bare ``[]`` answer
      (fence dropped, nothing but whitespace around it); an ``[]`` embedded
      in other prose is still inconclusive — only a reply that says nothing
      but "empty array" is unambiguous (an earlier review round).
    - ``None`` — no decodable findings array at all (truncation, garbage,
      empty content): the caller must treat the review as inconclusive,
      not as "No bugs found".

    If the model returns multiple fenced blocks (it should not), the last
    one wins: it is the final answer.
    """
    if text.strip() == "[]":
        return []  # entire bare empty-array answer: valid "no bugs" (an earlier review round)
    for candidate in _candidate_arrays(text):
        if not candidate:
            continue
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        findings: list[Finding] = []
        seen: set[tuple[str, int, str]] = set()
        for entry in parsed:
            f = _coerce_finding(entry)
            if f is None:
                continue
            key = (f.path, f.line, f.title.lower())
            if key in seen:
                continue
            seen.add(key)
            findings.append(f)
        return findings  # empty list = valid "no bugs" answer, not a parse miss
    return None


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Deduplicate on (path, line, title) — for findings whose paths are
    already normalized — preferring the HIGHEST severity on collision
    (an earlier review round): the same bug emitted as `x.py` low and header-form
    `b/x.py` high must keep the high, not whichever came first. The
    FIRST-occurring finding is kept and upgraded in place, so cap
    tie-breaking follows the bug's first mention, not the winning copy's
    position (an earlier review round). Ids are recomputed after dedup."""
    by_key: dict[tuple[str, int, str], Finding] = {}
    for f in findings:
        key = (f.path, f.line, f.title.lower())
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = f
        elif SEVERITY_RANK[f.severity] < SEVERITY_RANK[cur.severity]:
            cur.severity = f.severity  # first position survives, best severity wins
    out = list(by_key.values())
    for f in out:
        f.rekey()
    return out


def split_already_raised(
    findings: list[Finding], open_threads: list[dict]
) -> tuple[list[Finding], list[Finding]]:
    """(fresh, withheld) — partition findings against the bot's OPEN threads.

    A finding whose (line..end_line) span overlaps an open, non-outdated
    thread's span is already raised on this PR: withhold it rather than
    post a second thread for the same position (an earlier round 2026-10-15:
    dedupe on re-review) — and hand it back so the summary can list it,
    keeping the suppression reviewable (an earlier review round: a distinct second bug
    at that position must stay visible to a human). Outdated threads
    don't block — their line changed, and they get resolved instead
    (plan_thread_resolution). Spans overlap, not exact-match: a block
    bug re-flagged inside the range it was raised on dedupes (an earlier round
    adjacent-line survivor), while a finding on a disjoint span survives,
    by design — suppressing it would guess.
    """
    raised: list[dict] = []
    for t in open_threads:
        if not t["outdated"] and thread_spans(t):
            raised.append(t)
    withheld, fresh = [], []
    for f in findings:
        p = f.path.strip()
        if any((t.get("path") or "").strip() == p and _thread_covers(t, f.span())
               for t in raised):
            withheld.append(f)
        else:
            fresh.append(f)
    return fresh, withheld


def plan_thread_resolution(open_threads: list[dict]) -> list[tuple[dict, str]]:
    """Own threads this run should resolve, as (thread, note) pairs — the
    note is posted as a reply first so the resolution is not silent.

    Two kinds (an earlier round 2026-10-15, "resolve your own threads whose lines
    changed"):
    - outdated threads (GitHub says the commented line changed): superseded;
      the re-review re-flags the finding at its new position if it survives.
    - newer duplicates at the exact same (path, line) as an older open
      thread: one thread per position, oldest kept.
    Threads whose finding no longer matches anything stay open — resolving
    those would claim a fix nobody verified."""
    out: list[tuple[dict, str]] = []
    by_line: dict[tuple[str, int], list[dict]] = {}
    for t in open_threads:
        if t["outdated"]:
            out.append((t, (
                "Superseded: the commented line changed in a newer head. "
                "The review re-flags this finding at its new position if it "
                "still applies; if not, this thread stays as history.")))
        elif t["line"] is not None:
            by_line.setdefault((t["path"], t["line"]), []).append(t)
    for group in by_line.values():
        if len(group) > 1:
            group.sort(key=lambda t: t["created_at"])  # ISO-8601 sorts lexically
            for t in group[1:]:
                out.append((t, (
                    "Duplicate: this position already has an open Wrasse "
                    "Review thread — the oldest one stays, this copy is "
                    "resolved.")))
    return out


def select_findings(findings: list, max_findings: int) -> tuple[list, int]:
    """Sort by severity (high first, stable within a severity) and truncate
    to the cap. Truncating by model output order instead dropped a high
    that happened to come after max_findings lows (an earlier review round) — the
    cap must cost the least important findings, not the tail.

    Returns (selected, over_cap).
    """
    ranked = sorted(findings, key=lambda f: SEVERITY_RANK.get(f.severity, 1))
    return ranked[:max_findings], max(0, len(ranked) - max_findings)


def build_user_prompt(
    diff: str, repo_context: str, already_raised: list[str] | None = None,
    already_resolved: list[str] | None = None,
    pr_intent: dict | None = None,
    repo_map: str | None = None,
) -> str:
    """Assemble the one-turn review prompt: contract, PR intent, repo
    context, repo layout, already-raised findings, resolved-thread
    history, diff.

    ``pr_intent`` ({title, body}) is the PR's own statement of what it is
    for (an earlier round): the previous reviewer ingests it, and reviewing a diff blind to
    stated intent manufactures false positives on intended behavior — the
    an earlier round shape (10 findings, 9 false). The body is truncated at
    PR_INTENT_MAX_CHARS; the section is omitted entirely when there is
    no PR.

    ``repo_map`` is a bounded two-level layout (agent.py ToolBox.repo_map,
    an earlier round): first-turn orientation for free — the tree is fetched for
    recursive grep anyway. Omitted when empty."""
    parts = [
        "Review the following code change for bugs, following the output",
        "format from your instructions exactly.",
    ]
    if pr_intent and ((pr_intent.get("title") or "").strip()
                      or (pr_intent.get("body") or "").strip()):
        body = (pr_intent.get("body") or "(no description)").strip()
        if len(body) > PR_INTENT_MAX_CHARS:
            body = body[:PR_INTENT_MAX_CHARS] + "\n…(description truncated)"
        parts += [
            "",
            "## The author's stated intent for this change",
            "",
            f"**Title:** {pr_intent.get('title') or '(none)'}",
            "",
            body,
            "",
            "(Intent ends. Use it to tell intended behavior from bugs — do",
            "NOT report the change's stated purpose as a finding, and do not",
            "assume intent the description does not state.)",
        ]
    if repo_context.strip():
        parts += [
            "",
            "## Repository context",
            "",
            repo_context.strip(),
            "",
            "(Context ends. It describes repo conventions — do not report its",
            "contents as findings.)",
        ]
    if repo_map and repo_map.strip():
        parts += [
            "",
            "## Repository layout (top two levels)",
            "",
            repo_map.strip(),
            "",
            "(Layout ends. Use these real paths to scope searches — a grep",
            "scoped to the right directory costs one call instead of a walk.)",
        ]
    if already_raised:
        parts += [
            "",
            "## Findings already raised on this PR — do NOT re-report them",
            "",
            *already_raised,
            "",
            "Each position above already has an open review thread — the",
            "line shown, plus its whole span when a range like `12-18` is",
            "listed. Do not include these in your output, and do not",
            "re-derive the same bug at an adjacent line to get around it.",
            "Report only NEW bugs at other positions.",
        ]
    if already_resolved:
        parts += [
            "",
            "## Findings previously raised on this PR and already resolved",
            "",
            *already_resolved,
            "",
            "Each line above was raised in an earlier review round and the",
            "thread was resolved — fixed, or explained as not a bug (the",
            "resolution says which). Do not re-derive or re-report the same",
            "claim without checking the current diff: re-litigating a",
            "resolved thread is a false positive. If this diff genuinely",
            "reintroduces a resolved bug, report it — a finding at the same",
            "position is listed in the review summary for a human to see,",
            "not posted as a new thread. Otherwise report only bugs",
            "not in the list above.",
        ]
    parts += [
        "",
        "## The diff to review",
        "",
        "```diff",
        diff,
        "```",
        "",
        "Report only bugs introduced by the diff above. Line numbers are",
        "1-based in the file AFTER the change. Output the ```json block now.",
    ]
    return "\n".join(parts)
