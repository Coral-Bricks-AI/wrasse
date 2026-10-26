"""The bounded agentic review loop: the model reads the code before it reports.

A loop over POST /v1/responses with ``previous_response_id`` threading — the
fit for a multi-step tool agent: reasoning is carried across turns by the
provider, so each request carries only the new input. Not chat completions,
and not background polling: tool execution is turn-by-turn anyway, and poll
cadence would only add latency.

Division of labor: tools execute runner-side through the platform API (never a
checkout), pinned to the immutable head sha — never a branch name — so the
provider-side compute stays the model turns. One ``x-session-id`` per PR-head
review, sent on every turn, so the turns of one review group together and
per-turn latency is attributable.

Budget caps: ``--max-tool-turns`` counts model turns that may still produce
tool calls; the forced final answer (no tools) is one turn ON TOP of the cap,
so a review makes at most max_tool_turns + 1 model calls, plus a wall-clock
deadline. Exhaustion never fails the review — it forces a final report with
what the model already has, and flags ``budget_exhausted`` so the summary and
status say so. A large PR must degrade to a cut-short review, never to an
unbounded loop, and a cut-short review must not read as a clean one.

Findings contract: shared with the one-turn path (``SYSTEM_PROMPT``, the JSON
block, the markers and severity scale). The agent addendum only tells the
model it may look before it reports. Open- and resolved-thread history go into
the first turn's prompt, which with the hard dedupe in the orchestrator is
what stops a re-review from re-raising the same finding on every push.

The transport is injectable (``transport=``) and the tools are injectable
(``tools_exec=``), so the loop is unit-testable with canned /v1/responses
payloads — no network in the tests.
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

from wrasse.core import diff as difflines
from wrasse.platforms import github as fetch
from wrasse.core import contract as findings_mod
from wrasse.providers.base import (
    call_responses,
    extract_function_calls,
    extract_text,
)

# Tool bounds — the fetch-and-match contract (an earlier round v2). Each cap is a
# CLI/env knob away from the model prompt: they exist so one review can
# never pull the whole repo or loop unbounded, and each truncation is
# REPORTED to the model in the tool output, never silent.
READ_MAX_CHARS = 120_000        # one read_file answer, ~30k tokens
LIST_MAX_ENTRIES = 500          # one list_dir answer
GREP_MAX_FILES = 200            # files one grep may scan
LISTING_CAP = 1000              # GitHub's contents-API listing cap (no flag)
FILE_HISTORY_MAX = 10           # commits one file_history call may return
GREP_MAX_BYTES = 2_000_000      # total content one grep may scan
GREP_MAX_MATCHES = 50           # matches one grep may return
GREP_MAX_LINE_CHARS = 400       # shown per match line; search still sees the full line
GREP_SCAN_TIME_S = 30.0         # one grep's total scan-time backstop
PER_FILE_SCAN_S = 10.0          # one file's worker gets at most this much of it
FILE_CACHE_MAX_BYTES = 4_000_000  # per-review file cache; > this degrades
CACHEABLE_FILE_BYTES = 512_000   # bigger files are fetched per-use, never cached
OVERSIZE_FETCH_MAX_BYTES = 16_000_000  # per-review budget for those refetches
FORCED_FINAL_GRACE_S = 120.0    # the after-deadline final answer's window

AGENT_ADDENDUM = """\
You have tools to inspect the repository AT THE EXACT COMMIT UNDER REVIEW
before you report. Use them to verify what the diff suggests — read the
code around a change, check a caller, list a directory — instead of
guessing. The tools read only the reviewed commit.

- read_file(path, offset): contents of one file (truncated if very large;
  the truncation note names the next line — pass it as offset to read on).
- file_history(path): the recent commits that touched one file, newest
  first, as "sha date message" — use it to tell long-standing intended
  behavior from a bug this diff introduced.
- list_dir(path): entries of one directory; path "" is the repo root;
  direct children only, truncated if huge.
- grep(pattern, path, context_lines): regular-expression search, one match
  per line, as "path:line: text". context_lines (0-3) echoes that many
  surrounding lines around each match, indented — use it when you would
  otherwise follow the grep with a read_file to see the neighborhood.
  Without path it searches the files the diff touches;
  with a file path it searches that file; with a directory path it
  searches that directory RECURSIVELY — one call at a parent directory
  covers its whole subtree, so never walk subdirectories one call at a
  time. If a search reports it was truncated (too many files or bytes),
  narrow the path to a more specific subdirectory and search again.
  Results and the set of files scanned are bounded — a truncation
  note means you saw a prefix.

Call tools sparingly — a handful of targeted lookups, not a sweep. When
you are confident, or have looked enough, answer with the required output
format from your instructions and nothing else: your final message is
parsed as the findings array, so it must be exactly the ```json block."""

FORCED_FINAL_INPUT = (
    "Budget exhausted. Report your findings now, in the required output "
    "format (exactly one ```json fenced block), based on what you have "
    "already seen. If you saw no bugs, output []."
)

# Per-turn budget disclosure (an earlier round, replay A/B): without it the model
# has no idea a budget exists until "Budget exhausted" arrives, so on
# exploration-heavy heads it greps until the cap and the forced final loses
# findings to the discontinuity — measured on the an earlier round/an earlier round/an earlier round heads, 3/3
# runs self-terminated cleanly with the note (one at turn 7 of 25) where
# without it 100% exhausted the cap. The model treats the cap as its
# planning horizon; tell it the horizon.
BUDGET_NOTE = (
    "NOTE: {remaining} of {max_tool_turns} tool turn(s) remaining. When you "
    "have enough context, stop calling tools and answer now with the "
    "required findings JSON block."
)


class _CacheFull(RuntimeError):
    """The per-review file cache is full (bounded reads, bounded cache).
    Distinct from a missing file on purpose: reporting it as 'not found'
    would send the model a false negative it acts on (an earlier review round)."""


def _timed_scan(rx, text: str, path: str, start_count: int, budget_s: float,
                context_lines: int = 0):
    """Search every line of ``text`` in a KILLABLE worker process under a
    wall-clock budget (an earlier review round security): a single runaway backtracking
    search — a shape _risky_pattern cannot see, e.g. ``(a|a)+`` — must not
    hold the review hostage, and full-line search gives such a pattern
    real fuel.

    A worker PROCESS, not a thread, and the difference is load-bearing:
    CPython's sre engine holds the GIL for an entire match, so an
    abandoned daemon thread keeps the GIL forever and freezes the whole
    review process with it — observed directly while testing the thread
    version (the faulthandler dump showed the main thread stuck in
    Thread.start() waiting for a GIL the worker would never release). A
    process is killed outright on timeout.

    The worker STREAMS its progress (an earlier review round): every match is
    flushed the moment it is found and line counts checkpoint every 1000
    lines, so a scan killed mid-file keeps the matches it already found
    — a timed-out file must not report fewer findings than it had.

    Returns (matches, lines_searched, timed_out, capped). On timeout the
    caller keeps whatever arrived, discloses the incomplete file, and
    moves on to the next one while the total budget lasts. Wall time
    burned by KILLED workers is bounded by ceil(GREP_SCAN_TIME_S /
    PER_FILE_SCAN_S) — every spawn, kill and drain is wall time the
    caller re-charges against its budget before spawning the next
    worker, so the only unaccounted window is the ONE call in flight
    when the budget expires; total wall time is therefore bounded by
    GREP_SCAN_TIME_S plus one call's full cost (slice + spawn + drain).
    A worker that exits without a done record (crash, not kill) can mark
    a file incomplete near-instantly, so the COUNT of incomplete files
    can exceed the kill ceiling; each costs only its own spawn, bounded
    by the file cap (an earlier review round). ``lines_searched`` is the last
    checkpoint when a scan is killed — approximate by design, and the
    incomplete-file note says so. POSIX only (select(2) on the worker's
    pipe); the review runner is ubuntu-latest."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _SCAN_WORKER_SRC],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    # Feed stdin from a daemon writer: a payload bigger than the pipe
    # buffer would block the parent against a worker that never drains it
    # (killed before reading), and the parent must stay free to kill.
    payload = json.dumps({
        "pattern": rx.pattern, "flags": rx.flags, "text": text,
        "path": path, "start": start_count,
        "max_matches": GREP_MAX_MATCHES,
        "max_line_chars": GREP_MAX_LINE_CHARS,
        "context": context_lines,
    }).encode("utf-8")

    def feed():
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # worker died first — the kill path handles the verdict

    threading.Thread(target=feed, daemon=True).start()

    fd = proc.stdout.fileno()
    buf = b""
    deadline = time.monotonic() + max(budget_s, 0.01)
    timed_out = False
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            timed_out = True
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break  # EOF — the worker finished
        buf += chunk
    if timed_out:
        proc.kill()
    proc.wait()
    if timed_out:
        # Whatever the worker flushed before the kill is still in the pipe.
        try:
            buf += proc.stdout.read()
        except OSError:
            pass

    matches: list[str] = []
    lines = 0
    capped = False
    saw_done = False
    match_count = 0  # real matches only — context echoes must not trip caps
    for event_line in buf.decode("utf-8", "replace").splitlines():
        try:
            ev = json.loads(event_line)
        except json.JSONDecodeError:
            continue  # a line cut by the kill — ignore it, the note covers it
        if ev.get("t") == "m":
            matches.append(ev["v"])
            match_count += 1
            # A match proves its line was searched — carrying the count on
            # the event keeps a killed small-file scan from reporting 0
            # lines and dropping the file out of the searched count
            # (the review + an earlier review round).
            lines = max(lines, int(ev.get("l") or 0))
        elif ev.get("t") == "c":
            # Context echo (an earlier round): surrounding lines, indented so the
            # model can tell them from matches at a glance. Display-only.
            matches.append("  " + ev["v"])
        elif ev.get("t") == "p":
            lines = max(lines, int(ev["l"]))
        elif ev.get("t") == "done":
            lines = max(lines, int(ev["l"]))
            capped = bool(ev["capped"])
            saw_done = True
    if not saw_done and not timed_out:
        # EOF without a done record: the worker died early (crash, not
        # kill) — an inconclusive scan, never a clean one (an earlier review round).
        timed_out = True
    return matches, match_count, lines, timed_out, capped


# Runs in the worker process: read one JSON request off stdin, search each
# line (full line — the display cap is applied to what is REPORTED, not to
# the fuel), and STREAM the results — every match flushed as found
# (a killed scan must not lose what it already found), line counts
# checkpointed every 1000 lines, one done record at the end (flushed
# explicitly like every other record: pipe semantics guarantee data
# written before close is readable before EOF, but relying on the
# interpreter's exit-time flush for the protocol's only completion marker
# makes correctness depend on shutdown behavior — an earlier review round).
# Plain stdlib, no imports from this package: the worker must be killable
# and independent.
_SCAN_WORKER_SRC = r'''
import json, re, sys
from collections import deque
req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
rx = re.compile(req["pattern"], req["flags"])
out = sys.stdout.buffer
ctx = max(0, min(int(req.get("context", 0) or 0), 3))
recent = deque(maxlen=ctx) if ctx else None  # (lineno, shown) of last ctx lines
emitted = 0    # highest line echoed as match or context — dedups overlaps
trail_to = 0   # echo plain lines up to here after a match (context-after)
capped = False # the match cap tripped: finish the frozen window, then stop
lines = 0
matches = 0

def echo(kind, lineno, text):
    out.write((json.dumps(
        {"t": kind, "v": req["path"] + ":" + str(lineno) + ": " + text,
         "l": lineno}) + "\n").encode("utf-8"))

for lineno, line in enumerate(req["text"].splitlines(), 1):
    lines = lineno  # the line being examined counts as searched
    shown = line
    if len(line) > req["max_line_chars"]:
        shown = line[:req["max_line_chars"]] + "…[line truncated for display]"
    if not capped and rx.search(line):
        matches += 1
        if recent is not None:  # context-before, deduped against earlier echoes
            for ln, tx in recent:
                if ln > emitted:
                    echo("c", ln, tx.strip())
                    emitted = ln
        echo("m", lineno, shown.strip())
        emitted = lineno
        trail_to = lineno + ctx if ctx else 0
        out.flush()
        if matches + req["start"] >= req["max_matches"]:
            # Cap hit (an earlier review round): stop counting, but do NOT exit before
            # the last match's trailing window — the caller asked for that
            # context and a cut mid-window reads as a broken scan. The
            # window is FROZEN here: a post-cap matching line must not
            # re-enter this branch and chain the scan past the cap
            # (an earlier review round).
            if ctx:
                capped = True
            else:
                out.write((json.dumps({"t": "done", "l": lines, "capped": True}) + "\n").encode("utf-8"))
                out.flush()
                sys.exit(0)
    elif ctx and lineno <= trail_to and lineno > emitted:
        echo("c", lineno, shown.strip())  # context-after
        emitted = lineno
    if recent is not None:
        recent.append((lineno, shown))
    if lineno % 1000 == 0:
        out.write((json.dumps({"t": "p", "l": lines}) + "\n").encode("utf-8"))
        out.flush()
    if capped and lineno >= trail_to:
        # The cap's trailing window has closed (an earlier review round):
        # trail_to is FROZEN at cap time — a post-cap matching line must
        # not re-enter the match branch, bump the window, and chain the
        # scan past the cap in a dense-match file.
        out.write((json.dumps({"t": "done", "l": lines, "capped": True}) + "\n").encode("utf-8"))
        out.flush()
        sys.exit(0)
out.write((json.dumps({"t": "done", "l": lines, "capped": capped}) + "\n").encode("utf-8"))
out.flush()
'''


def _risky_pattern(src: str) -> bool:
    """True for the classic ReDoS signature: a quantified group that itself
    contains a quantifier — ``(a+)+``, ``(a*)*b`` … Nested quantifiers make
    backtracking exponential. Best-effort defense-in-depth (an earlier review round),
    not a complete matcher — it pairs with the per-line length cap and the
    per-line scan-time check; a false positive only costs the model a
    'simplify the pattern' error, a false negative is bounded by those
    caps. Escapes are skipped so a literal ``\\*`` never counts, and group
    prefixes (``(?:``, ``(?=``, ``(?P<name>`` …) are consumed whole so
    their ``?`` is not mistaken for a quantifier — ``(?:foo|bar)+`` is a
    safe pattern, not a risky one (an earlier review round).

    Known hole (an earlier review round, security): AMBIGUOUS ALTERNATION under a
    quantifier — ``(a|a)+`` — backtracks exponentially without any nested
    quantifier, so this heuristic misses it. Detecting it requires branch
    equivalence, which stdlib ``re`` gives no hook for; the remaining
    layers bound it: the killable worker-process budget (_timed_scan), the
    per-file scan-time cap, and the review job's 15-minute workflow
    timeout failing the status visibly. Blast radius: one wasted CI job
    on a private repo."""
    chars = list(src)
    stack: list[bool] = []  # per open group: does it contain a quantifier?
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch == "\\":
            i += 2  # skip the escaped character
            continue
        if ch == "(":
            stack.append(False)
            if i + 1 < len(chars) and chars[i + 1] == "?":
                i = _skip_group_prefix(chars, i + 1)  # i now at the prefix end
        elif ch == ")":
            if not stack:
                return False  # unbalanced — re.compile reports the real error
            has_quant = stack.pop()
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            # nxt must be non-empty: '' in "*+?{" is True in Python, which
            # flagged every pattern ENDING in a quantified group — (a+),
            # (a+)(b+) — as risky (caught while verifying an earlier round r4).
            if has_quant and nxt and nxt in "*+?{":
                return True
        elif ch in "*+?" or (ch == "{" and _is_interval(chars, i)):
            if stack:
                stack[-1] = True
        i += 1
    return False


def _skip_group_prefix(chars: list[str], i: int) -> int:
    """With ``i`` at the ``?`` of a group prefix, return the index of its
    last character: ``(?:``, ``(?=``, ``(?!`` take 2, ``(?<=`` / ``(?<!``
    take 3, ``(?P<name>`` runs through the ``>``. Inline flags like
    ``(?i)`` fall through the same way — nothing in a prefix is a
    quantifier. Called only when chars[i] == '?'."""
    if i + 1 < len(chars) and chars[i + 1] in ":=!":
        return i + 1
    if i + 2 < len(chars) and chars[i + 1] == "<" and chars[i + 2] in "=!":
        return i + 2
    if i + 1 < len(chars) and chars[i + 1] == "P" and i + 2 < len(chars) \
            and chars[i + 2] == "<":
        try:
            return chars.index(">", i + 3)
        except ValueError:
            return len(chars) - 1
    return i  # bare "(?" — re.compile reports the real error


def _is_interval(chars: list[str], i: int) -> bool:
    """True when ``{`` at ``i`` starts a ``{m,n}`` quantifier (and not a
    literal brace, which re rejects anyway — but the check is cheap)."""
    j = i + 1
    digits = 0
    while j < len(chars) and (chars[j].isdigit() or chars[j] == ","):
        if chars[j].isdigit():
            digits += 1
        j += 1
    return digits > 0 and j < len(chars) and chars[j] == "}"

# Flat Responses-API tool shape (no nested "function" wrapper) —
# the gateway's published Responses-API tool example. strict=True so
# argument drift fails at the API, not in dispatch.
TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read one file's contents at the reviewed commit. Large files "
            "truncate; the truncation note names the next line — call "
            "again with that offset to read further."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative file path."},
                "offset": {"type": "integer", "description": "1-based line to start reading from (the value the truncation note names)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "file_history",
        "description": (
            "List the recent commits that touched one file (sha, date, "
            "message), newest first. Use it to tell long-standing "
            "intended behavior from a bug this diff introduced."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative file path."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_dir",
        "description": "List one directory's direct entries at the reviewed commit.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path; \"\" is the repo root."},
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "grep",
        "description": (
            "Regex-search file contents at the reviewed commit. Without path, "
            "searches the files the diff touches; with a file path searches "
            "that file; with a directory path searches that directory "
            "recursively (every file under it, bounded). context_lines (0-3) "
            "echoes that many surrounding lines around each match, indented "
            "— often saves a follow-up read_file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {"type": "string", "description": "Optional file or directory to scope the search."},
                "context_lines": {"type": "integer", "description": "Surrounding lines to echo around each match (0-3)."},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass(frozen=True)
class AgentReply:
    """What review.py needs from the loop. The first four fields are the
    ModelReply shape (elapsed, tokens), so build_summary treats both
    transports alike; the rest is the agent's own telemetry."""

    content: str
    elapsed_s: float          # sum of model-turn latencies (not wall clock)
    prompt_tokens: int
    completion_tokens: int
    turns: int                # model calls made (<= max_tool_turns + 1)
    tool_calls: int           # function_call items executed
    budget_exhausted: bool
    final_truncated: bool = False  # final turn hit the output-token limit
    # Per-tool call counts (an earlier round): the replay benchmark's rows must
    # show WHICH tools a review used, or "is it exploring smartly" needs
    # manual inference from gateway logs on every analysis pass.
    tool_counts: dict = field(default_factory=dict)


class ToolBox:
    """Runner-side tool execution via the GitHub API at the immutable head
    sha. Every method returns a string for the model — errors included: a
    tool failure must answer the turn (the loop continues), never raise
    (which would fail the whole review)."""

    def __init__(self, repo: str, head_sha: str | None, token: str | None, diff: str):
        self.repo = repo
        self.head_sha = head_sha or ""
        self.token = token or ""
        self.diff_paths = sorted(set(difflines.parse_hunks(diff)))
        self._cache: dict[str, str | None] = {}
        self._cache_bytes = 0
        self._oversize_bytes = 0  # uncached per-use refetches, against their own budget
        # Recursive-tree cache (one trees-API call per review) and the
        # exact-duplicate guard's set of already-answered calls.
        self._tree: list[str] | None = None
        self._tree_truncated = False
        self._dispatched: set[str] = set()
        # Local --diff-file mode without --head-sha: tools degrade
        # per-call rather than the agent mode being unusable offline.
        # The message names what is ACTUALLY missing (an earlier review round):
        # a replay run with a sha but no token used to claim the sha was
        # missing, sending the model a wrong diagnosis.
        if not self.head_sha or self.head_sha == "local":
            self._unavailable_reason = (
                "local diff without --head-sha — there is no commit for "
                "the tools to read")
        elif not self.token:
            self._unavailable_reason = (
                "no GitHub token — pass one to let the tools read the "
                "pinned commit")
        else:
            self._unavailable_reason = ""
        self.enabled = not self._unavailable_reason

    def _unavailable(self, what: str) -> str:
        return f"error: {what} unavailable — {self._unavailable_reason}."

    def _file_text(self, path: str) -> str | None:
        """Cached file content at the head sha; None = absent/oversize.
        A full cache raises _CacheFull — NOT None, which the callers
        report to the model as 'not found', a false negative it would act
        on (an earlier review round). Files over CACHEABLE_FILE_BYTES are
        returned but NOT cached (an earlier review round): one grep's byte-cap skips
        could otherwise pin several multi-MB blobs in the shared 4MB
        cache and starve later targeted reads for the whole review. Their
        per-use refetches are bounded by OVERSIZE_FETCH_MAX_BYTES (Wrasse
        Review an earlier round): uncached reads still count against a per-review
        budget, so the _CacheFull signal stays meaningful instead of
        being silently unexhaustible."""
        if path in self._cache:
            return self._cache[path]
        if self._cache_bytes >= FILE_CACHE_MAX_BYTES:
            raise _CacheFull()
        text = fetch.fetch_file(self.repo, self.head_sha, path, self.token)
        if text is not None and len(text) <= CACHEABLE_FILE_BYTES:
            self._cache_bytes += len(text)
            self._cache[path] = text
        elif text is not None:
            # The check is necessarily POST-fetch (a file's size is only
            # known once fetched), so the budget can overshoot by one
            # file — bounded: fetch_file refuses files over 1MB (GitHub
            # omits their content), so the overshoot is at most ~1MB on
            # top of OVERSIZE_FETCH_MAX_BYTES (an earlier review round).
            self._oversize_bytes += len(text)
            if self._oversize_bytes > OVERSIZE_FETCH_MAX_BYTES:
                raise _CacheFull()
        return text

    def read_file(self, path, offset=None) -> str:
        if not self.enabled:
            return self._unavailable("read_file")
        if not isinstance(path, str) or not path.strip():
            return "error: read_file needs a non-empty string path."
        if offset is not None and (not isinstance(offset, int) or offset < 1):
            return "error: offset must be a positive 1-based line number (or omitted)."
        path = path.strip()
        start_line = offset or 1
        try:
            text = self._file_text(path)
        except _CacheFull:
            return ("error: per-review file cache is full — be more "
                    "targeted instead of reading another file.")
        except fetch.GitHubError as e:
            return f"error: GitHub API failure reading {path}: {e}"
        except Exception as e:  # noqa: BLE001 — a tool failure must answer the turn
            return f"error: could not read {path}: {e}"
        if text is None:
            return f"error: {path} not found at this commit (or over 1MB, or a directory)."
        if start_line > 1:
            # Offset reads (an earlier round): a file truncated mid-read used to
            # be a dead end — the model could never see the rest. Lines,
            # not chars: grep reports line numbers, so the model speaks
            # lines.
            lines = text.splitlines()
            if start_line > len(lines):
                return f"error: {path} has {len(lines)} line(s); offset {start_line} is past the end."
            text = "\n".join(lines[start_line - 1:])
        if len(text) > READ_MAX_CHARS:
            # The note is the recovery instruction (an earlier round), and it
            # must be exact (an earlier review round): landing ON a newline means the
            # next character starts the following line — count it as
            # consumed and the note skips a whole unread line. And when
            # truncation falls inside the FINAL line, there is no next
            # line to resume at — advertising an offset there is a dead
            # end, so the note says the remainder is unreachable instead.
            shown = text[:READ_MAX_CHARS]
            newlines = shown.count("\n")
            last_line = start_line + len(text.splitlines()) - 1
            next_line = (start_line + newlines if shown.endswith("\n")
                         else start_line + newlines + 1)
            if next_line > last_line:
                return shown + (
                    f"\n\n[truncated at {READ_MAX_CHARS} characters inside "
                    f"line {last_line} — the line exceeds the read window "
                    "and its remainder cannot be read with offsets]")
            if shown.endswith("\n"):
                return shown + (
                    f"\n\n[truncated at {READ_MAX_CHARS} characters — the "
                    f"file continues at line {next_line}; call read_file "
                    f"again with offset={next_line} to read further]")
            return shown + (
                f"\n\n[truncated at {READ_MAX_CHARS} characters inside line "
                f"{next_line - 1} — the rest of that line is beyond the "
                f"read window and is skipped; the file continues at line "
                f"{next_line}; call read_file again with offset={next_line} "
                "to read further]")
        return text

    def file_history(self, path) -> str:
        """Recent commits touching one path (an earlier round). When/why a line
        appeared separates long-standing intended behavior from a bug this
        diff introduced — the model's cheapest false-positive check after
        the PR description. Bounded to FILE_HISTORY_MAX commits."""
        if not self.enabled:
            return self._unavailable("file_history")
        if not isinstance(path, str) or not path.strip():
            return "error: file_history needs a non-empty string path."
        path = path.strip()
        try:
            commits = fetch.fetch_path_history(
                self.repo, path, self.token,
                at_ref=self.head_sha, per_page=FILE_HISTORY_MAX)
        except fetch.GitHubError as e:
            return f"error: GitHub API failure fetching history for {path}: {e}"
        except Exception as e:  # noqa: BLE001
            return f"error: could not fetch history for {path}: {e}"
        if not commits:
            return f"no commits found touching {path} at or below the reviewed commit."
        out = []
        for c in commits:
            first_line = (c["message"].splitlines() or [""])[0][:100]
            out.append(f"{c['sha'][:10]} {c['date'][:10]} {first_line}")
        return "\n".join(out)

    def list_dir(self, path) -> str:
        if not self.enabled:
            return self._unavailable("list_dir")
        path = (path or "").strip()
        try:
            entries = fetch.fetch_dir_listing(self.repo, self.head_sha, path, self.token)
        except fetch.GitHubError as e:
            return f"error: GitHub API failure listing {path or '<root>'}: {e}"
        except Exception as e:  # noqa: BLE001
            return f"error: could not list {path or '<root>'}: {e}"
        if entries is None:
            return f"error: {path or '<root>'} not found at this commit, or not a directory."
        out = [f"{e['name']}{'/' if e['type'] == 'dir' else ''}" for e in entries]
        note = ""
        if len(out) > LIST_MAX_ENTRIES:
            out = out[:LIST_MAX_ENTRIES]
            note = f"\n[truncated at {LIST_MAX_ENTRIES} entries]"
        return "\n".join(out) + note if out else "(directory is empty)"

    def grep(self, pattern, path=None, context_lines=None) -> str:
        if not self.enabled:
            return self._unavailable("grep")
        if not isinstance(pattern, str) or not pattern:
            return "error: grep needs a non-empty string pattern."
        if context_lines is None:
            context_lines = 0
        if (not isinstance(context_lines, int) or isinstance(context_lines, bool)
                or context_lines < 0 or context_lines > 3):
            return ("error: context_lines must be an integer between 0 and 3.")
        if _risky_pattern(pattern):
            # Best-effort ReDoS guard (an earlier review round): the per-line length cap
            # and the scan-time check below bound what slips past; this
            # rejects the exponential shapes outright.
            return ("error: pattern rejected — nested quantifiers (like "
                    "(a+)+) can take exponential time; simplify the pattern.")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"error: invalid regular expression: {e}"
        path = (path or "").strip().strip("/")
        # Scope resolution: an explicit file wins; a directory means its
        # direct children; no path means the diff-touched files. Each branch
        # is bounded by GREP_MAX_FILES before any content is fetched, and
        # the scope label DISCLOSES the bound (an earlier review round: the diff-wide
        # branch truncated silently before).
        try:
            files, scope_note = self._grep_scope(path)
        except fetch.GitHubError as e:
            return f"error: GitHub API failure scoping grep to {path or 'diff files'}: {e}"
        except Exception as e:  # noqa: BLE001
            return f"error: could not scope grep: {e}"
        matches: list[str] = []   # display lines: matches AND context echoes
        real_matches = 0         # matches only — caps count these, not context
        scanned = 0
        missing = 0
        incomplete = 0
        byte_skipped = 0
        bytes_fetched = 0
        truncated = []
        scan_t0 = time.monotonic()
        for f in files:
            if scanned >= GREP_MAX_FILES:
                # Defense in depth: _grep_scope already slices at the same
                # cap, so this branch is normally unreachable — but if a
                # future scope change lets a longer list through, count
                # every unexamined kind: byte-cap-skipped and unreadable
                # files are not "scanned", so the remainder must subtract
                # them too or the note overcounts (an earlier review round).
                rem = len(files) - scanned - missing - byte_skipped
                if rem > 0:
                    truncated.append(f"{rem} file(s) not scanned")
                break
            try:
                text = self._file_text(f)
            except _CacheFull:
                truncated.append("file cache full — remaining files not scanned")
                break
            except fetch.GitHubError as e:
                # A mid-scan fetch failure must not discard what earlier
                # files already found (an earlier review round): degrade to a partial
                # answer with the failure disclosed, never a bare error
                # that buries discovered matches.
                truncated.append(f"GitHub API failure reading {f}: {e}")
                break
            except Exception as e:  # noqa: BLE001
                truncated.append(f"could not read {f}: {e}")
                break
            if text is None:
                missing += 1  # deleted by the diff, over 1MB, or a directory
                continue
            if bytes_fetched + len(text) > GREP_MAX_BYTES:
                # Fetched but not searched — skip THIS file and keep
                # scanning (an earlier review round): the old check broke the
                # whole loop on the first oversize file. Counted silently
                # and disclosed ONCE after the loop (r3: a note per
                # skipped file is cap-amplified noise). The head says
                # "searched", so a file fetched but never matched against
                # the pattern is not silently in that count (r5/r6).
                # bytes_fetched counts only SEARCHED bytes by design
                # (an earlier review round): it bounds grep's scan work, not
                # fetching — fetched content is bounded separately by
                # CACHEABLE_FILE_BYTES (uncached oversize files) and the
                # file-cache cap.
                byte_skipped += 1
                continue
            remaining = GREP_SCAN_TIME_S - (time.monotonic() - scan_t0)
            if remaining <= 0:
                truncated.append("scan-time cap reached — remaining files not scanned")
                break
            # A file's worker gets at most PER_FILE_SCAN_S of the remaining
            # budget (an earlier review round): one slow file — legit-big or
            # pathological — must not starve every file after it.
            # Search the FULL line, truncate only what is SHOWN (the previous reviewer
            # an earlier round): a match past column 400 is a real finding, and
            # silently dropping it made grep lie by omission. The
            # worker-process budget bounds the search itself (an earlier review round):
            # the line-length cap bounds only the DISPLAY, not the fuel a
            # pathological pattern gets.
            file_matches, file_match_count, lines_searched, search_timed_out, capped = _timed_scan(
                rx, text, f, real_matches, min(remaining, PER_FILE_SCAN_S),
                context_lines)
            if lines_searched:
                # Counted only once a line was actually searched (Wrasse
                # Review an earlier round r6): a file whose scan timed out before any
                # line finished was fetched but never matched — "searched"
                # must not include it.
                scanned += 1
                bytes_fetched += len(text)
            matches.extend(file_matches)
            real_matches += file_match_count
            if capped:
                truncated.append(f"match cap reached at {GREP_MAX_MATCHES}")
            if search_timed_out:
                # Per-file ceiling, not a verdict on the whole grep (the previous reviewer
                # an earlier round r4): the file's own slice ran out while the total
                # budget may still have plenty — keep the streamed matches,
                # disclose the incomplete file, and CONTINUE. Killed
                # workers per call are bounded by the budgets:
                # ceil(GREP_SCAN_TIME_S / PER_FILE_SCAN_S).
                truncated.append(f"{f} scan incomplete — per-file time cap")
                incomplete += 1
            # Flags, not a string sniff of truncated[-1] (an earlier review round
            # r4): an equality-of-intent check must not depend on note
            # wording.
            if capped or real_matches >= GREP_MAX_MATCHES:
                break
        if byte_skipped:
            truncated.insert(0, f"{byte_skipped} file(s) fetched but skipped — byte cap")
        head = f"searched {scanned} file(s) in {scope_note}"
        if missing:
            head += f"; {missing} unreadable (deleted or oversize)"
        if incomplete and scanned:
            # A matched-but-killed file counts as searched (its match proves
            # a line completed) while its scan is incomplete — the head
            # must not overstate coverage, so the count of partial scans
            # rides along (an earlier review round).
            head += f"; {incomplete} scan(s) incomplete"
        if not matches:
            notes = f"\n[{'; '.join(truncated)}]" if truncated else ""
            if scanned == 0 and truncated:
                # Nothing was searched and something went wrong: "no
                # matches" would read as pattern-absent when the scan
                # never happened (an earlier review round).
                self._last_partial = bool(truncated)
                return f"{head}\nno files were searched{notes}"
            self._last_partial = bool(truncated)
            return f"{head}\nno matches{notes}"
        body = "\n".join(matches)
        notes = f"\n[{'; '.join(truncated)}]" if truncated else ""
        self._last_partial = bool(truncated)
        return f"{head}\n{body}{notes}"

    def _repo_tree(self) -> tuple[list[str], bool]:
        """All blob paths at the head sha, cached after ONE trees-API call.
        Raises like any fetch — grep() answers the turn with the error
        string and the loop continues."""
        if self._tree is None:
            self._tree, self._tree_truncated = fetch.fetch_tree_paths(
                self.repo, self.head_sha, self.token)
        return self._tree, self._tree_truncated

    def repo_map(self, max_chars: int = 3000) -> str:
        """A bounded two-level layout of the repo (an earlier round): the tree is
        already fetched for recursive grep, so the model's FIRST turn can
        carry orientation for free — no discovery list_dir turns, and
        greps get well-scoped paths from the start. Cold-start on an
        unfamiliar codebase is a product requirement (one yml + key).
        Empty string when the tree is unavailable: the map is an
        optimization, never a dependency."""
        try:
            tree, truncated = self._repo_tree()
        except Exception:  # noqa: BLE001 — the map degrades to nothing
            return ""
        top: dict[str, set[str]] = {}
        for p in tree:
            parts = p.split("/")
            if len(parts) == 1:
                top.setdefault("", set()).add(parts[0])
            else:
                top.setdefault(parts[0], set()).add(parts[1])
        lines = []
        # Per-LINE bound check (an earlier review round): a group-level check lets one
        # huge directory blow far past the limit and cut off everything
        # alphabetically after it — a big enough "zz/" would make "web/"
        # vanish. Stop mid-directory and say so.
        for name in sorted(top):
            children = sorted(top[name])
            entries = ([f"{c}" for c in children] if name == ""
                       else [f"{name}/"] + [f"  {c}" for c in children])
            for entry in entries:
                if sum(len(ln) + 1 for ln in lines) + len(entry) + 1 > max_chars:
                    lines.append("…(layout truncated)")
                    out = "\n".join(lines)
                    if truncated:
                        out += "\n(GitHub truncated the tree listing; some paths missing)"
                    return out
                lines.append(entry)
        out = "\n".join(lines)
        if truncated:
            out += "\n(GitHub truncated the tree listing; some paths missing)"
        return out

    def _grep_scope(self, path: str) -> tuple[list[str], str]:
        """(files to scan, human label). Bounded to GREP_MAX_FILES here so
        the caps in grep() see a finite list — and the label says so when
        the bound bit (an earlier review round).

        Explicit paths resolve against the cached recursive tree (an earlier round): a directory scopes to its whole subtree in ONE grep. The
        non-recursive grep made "find usages of X" a search-by-enumeration
        walk — measured on the an earlier round head, 15+ calls probing web/src/pages,
        web/src/components, … one directory at a time, several of them
        verbatim repeats."""
        if not path:
            if len(self.diff_paths) > GREP_MAX_FILES:
                return (self.diff_paths[:GREP_MAX_FILES],
                        f"the diff's changed files (truncated at {GREP_MAX_FILES} files)")
            return self.diff_paths, "the diff's changed files"
        try:
            tree, tree_truncated = self._repo_tree()
        except fetch.GitHubError:
            # Tree fetch failed: fall back to the directory listing
            # (an earlier review round) — an explicit FILE path must stay searchable
            # when the trees API is down; the listing branch is the
            # pre-recursive behavior, direct children only.
            listing = fetch.fetch_dir_listing(self.repo, self.head_sha, path, self.token)
            if listing is not None:
                files = [f"{path}/{e['name']}" for e in listing if e["type"] == "file"]
                label = f"directory {path} (direct files — recursive tree unavailable)"
                if len(listing) >= LISTING_CAP:
                    # an earlier review round: the contents API caps a listing at 1000
                    # entries with NO truncated flag — a bigger directory
                    # silently under-scopes. 1000 is "at or past the cap",
                    # so the label says "may": an exactly-1000-entry
                    # directory reports this too.
                    label += (f" — listing may be truncated at GitHub's "
                              f"{LISTING_CAP}-entry cap, coverage incomplete")
            else:
                files, label = [path], f"file {path}"
            if len(files) > GREP_MAX_FILES:
                return files[:GREP_MAX_FILES], f"{label} (truncated at {GREP_MAX_FILES} files)"
            return files, label
        prefix = path + "/"
        subtree = [p for p in tree if p.startswith(prefix)]
        if subtree:
            files, label = subtree, f"directory {path} (recursive)"
        else:
            # Not a directory with files under it: treat as a single file —
            # a blob path (grep that one file) or a deleted-by-diff path
            # (the fetch 404s and grep reports it as unreadable).
            files, label = [path], f"file {path}"
        if tree_truncated:
            label += (" — GitHub truncated the tree listing, paths beyond "
                      "the cap were not searchable")
        if len(files) > GREP_MAX_FILES:
            return files[:GREP_MAX_FILES], f"{label} (truncated at {GREP_MAX_FILES} files)"
        return files, label

    def dispatch(self, name, args_json) -> str:
        """Execute one function_call item. NEVER raises: unknown tool, bad
        JSON, and API failures all come back as error strings so the loop
        continues and the model can adjust.

        Exact-duplicate guard (an earlier round): a verbatim repeat of a call
        already answered this review is answered with a pointer to the
        earlier result instead of re-executing — measured 5/39 calls on
        one head were verbatim repeats, pure budget waste (content at the
        pinned sha never changes, so a repeat can never add information).
        Only SUCCESSFUL calls are remembered: a failed call stays retryable
        (a transient API failure must not poison the tool for the rest of
        the review)."""
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            return f"error: arguments are not valid JSON: {str(args_json)[:200]!r}"
        if not isinstance(args, dict):
            return "error: arguments must be a JSON object."
        key = f"{name} {json.dumps(args, sort_keys=True, separators=(',', ':'))}"
        if key in self._dispatched:
            return ("error: this exact call was already answered earlier in "
                    "this review — its result is in the conversation above; "
                    "vary the pattern or path if you need different "
                    "information.")
        # Per-call partial flag (an earlier review round): a grep that hit a time/byte
        # cap or a mid-scan fetch failure came back INCOMPLETE — caching it
        # would turn the model's retry into a pointer to the partial
        # result, stranding the unsearched files. Only complete results
        # are remembered.
        self._last_partial = False
        try:
            if name == "read_file":
                result = self.read_file(args.get("path"), args.get("offset"))
            elif name == "file_history":
                result = self.file_history(args.get("path"))
            elif name == "list_dir":
                result = self.list_dir(args.get("path"))
            elif name == "grep":
                result = self.grep(args.get("pattern"), args.get("path"),
                                   args.get("context_lines"))
            else:
                return (f"error: unknown tool {name!r} — use read_file, "
                        "file_history, list_dir, or grep.")
        except Exception as e:  # noqa: BLE001 — last-resort guard, same rule
            return f"error: tool execution failed: {e}"
        if not result.startswith("error:") and not self._last_partial:
            self._dispatched.add(key)
        return result


def run_agent_review(
    *,
    api_key: str,
    model: str,
    base_url: str,
    session_id: str,
    repo: str,
    head_sha: str,
    token: str | None,
    diff: str,
    context: str,
    already_raised: list[str] | None,
    already_resolved: list[str] | None = None,
    pr_intent: dict | None = None,
    max_tool_turns: int = 10,
    wall_timeout_s: float = 600.0,
    per_turn_timeout_s: float = 300.0,
    transport=call_responses,
    tools_exec=None,
) -> AgentReply:
    """The v2 review: a bounded agentic loop, findings JSON out.

    At most ``max_tool_turns`` turns may produce tool calls; every
    tool-output batch carries BUDGET_NOTE with the turns remaining, so the
    model plans its own exit instead of being cut off mid-thought. If the
    budget (turns or wall clock) runs out first, ONE forced final turn
    without tools ends the review (``budget_exhausted=True``) — and it
    ANSWERS any pending tool calls first, because the gateway's chat
    translation errors on a ``function_call`` with no output (an earlier review round).
    A turn that yields neither tool calls nor text is a harness error —
    same rule as v1's unparseable output: inconclusive, never "no bugs".
    """
    if max_tool_turns < 1:
        raise ValueError(
            "max_tool_turns must be >= 1 — a review that never shows the "
            "model the diff is not a review (an earlier review round)")
    if wall_timeout_s <= 0:
        raise ValueError("wall_timeout_s must be > 0 (an earlier review round)")
    toolbox = tools_exec or ToolBox(repo, head_sha, token, diff)
    instructions = findings_mod.SYSTEM_PROMPT + "\n\n" + AGENT_ADDENDUM
    # Orientation for free (an earlier round): the tree is fetched for recursive
    # grep anyway — a bounded layout in the first turn removes discovery
    # turns and scopes greps from the start. Degrades to nothing when the
    # tree is unavailable or the tools are disabled (an earlier review round: an
    # offline/local run must not pay a doomed API attempt before turn 1);
    # never a dependency.
    repo_map = (toolbox.repo_map() if isinstance(toolbox, ToolBox)
                and toolbox.enabled else "")
    # First-turn context only (an earlier round pre-A/B): open threads as
    # do-not-re-report, and the RESOLVED history as do-not-re-derive —
    # without the latter the loop re-derives the same shapes every round
    # (an earlier round: two rounds, 17 findings, zero new). No transport change.
    next_input = findings_mod.build_user_prompt(
        diff, context, already_raised, already_resolved, pr_intent,
        repo_map=repo_map or None)
    prev_id: str | None = None

    t0 = time.monotonic()
    turns = 0
    tool_calls = 0
    tool_counts: dict = {}  # per-name counts (an earlier round benchmark rows)
    prompt_tokens = 0
    completion_tokens = 0
    model_s = 0.0
    budget_exhausted = False

    def call(inp, tools):
        nonlocal turns, prompt_tokens, completion_tokens, model_s, prev_id
        if tools is None:
            # The forced final runs after the deadline by design — it takes
            # a fixed grace window.
            timeout = min(per_turn_timeout_s, FORCED_FINAL_GRACE_S)
            deadline = t0 + wall_timeout_s + FORCED_FINAL_GRACE_S
        elif turns == 0:
            # First turn: the loop admits it regardless of the wall clock
            # (the model must see the diff), so it must also get a REAL
            # allowance — clamping it to the leftover milliseconds would
            # kill the request the loop just allowed (an earlier review round).
            timeout = per_turn_timeout_s
            deadline = t0 + wall_timeout_s
        else:
            # A later in-budget turn gets at most the wall time it has
            # LEFT, so the deadline bounds the review, not just the next
            # turn (an earlier review round): a turn that cannot fit in what remains
            # fails visibly (GatewayError → error status) instead of
            # silently overshooting.
            elapsed = time.monotonic() - t0
            timeout = min(per_turn_timeout_s, max(0.0, wall_timeout_s - elapsed))
            # The wall deadline clamps the transport's ONE transient-failure
            # retry (network or 502/503; an earlier review round): a turn that already
            # spent its remaining wall clock on a fast connection failure
            # must not get the same allowance twice on the retry.
            deadline = t0 + wall_timeout_s
        turn = transport(
            api_key=api_key,
            model=model,
            input=inp,
            base_url=base_url,
            session_id=session_id,
            instructions=instructions,
            tools=tools,
            previous_response_id=prev_id,
            timeout_s=timeout,
            deadline_s=deadline,
        )
        turns += 1
        model_s += turn.elapsed_s
        prompt_tokens += turn.input_tokens
        completion_tokens += turn.output_tokens
        prev_id = turn.id or prev_id
        return turn

    while True:
        out_of_turns = turns >= max_tool_turns
        # The wall clock is only checked once the model has SEEN the diff
        # (an earlier review round): a zero-adjacent budget must degrade to a one-turn
        # review, never to a blind verdict.
        out_of_time = turns >= 1 and time.monotonic() - t0 >= wall_timeout_s
        if out_of_turns or out_of_time:
            # Forced final: tools omitted so the model must answer. The
            # budget flag rides on the reply so the status/summary can say
            # the review was cut short — a truncated review must not read
            # as a clean one.
            budget_exhausted = True
            if isinstance(next_input, list):
                # A tool turn just ran: its outputs are still undelivered.
                # Answer the calls and append the report-now message in
                # ONE input — an orphaned function_call can make the next
                # request fail outright (an earlier review round).
                inp = next_input + [{"role": "user", "content": FORCED_FINAL_INPUT}]
            else:
                inp = FORCED_FINAL_INPUT
            turn = call(inp, tools=None)
            text = extract_text(turn.output)
            if text is None:
                raise RuntimeError(
                    "no final answer under budget (model kept calling tools "
                    "or returned no text)"
                )
            return AgentReply(text, model_s, prompt_tokens, completion_tokens,
                              turns, tool_calls, budget_exhausted,
                              final_truncated=turn.status == "incomplete",
                              tool_counts=tool_counts)

        turn = call(next_input, tools=TOOLS)
        calls = extract_function_calls(turn.output)
        if not calls:
            text = extract_text(turn.output)
            if text is None:
                raise RuntimeError(
                    "model turn produced neither tool calls nor text "
                    f"(output items: {[it.get('type') for it in turn.output if isinstance(it, dict)]})"
                )
            # An answer the gateway marked incomplete was cut by the
            # output-token limit (an earlier review round): the text may still parse, so
            # the review posts — but the flag rides out and the summary and
            # status say the answer was truncated, never a silent
            # under-report.
            return AgentReply(text, model_s, prompt_tokens, completion_tokens,
                              turns, tool_calls, budget_exhausted,
                              final_truncated=turn.status == "incomplete",
                              tool_counts=tool_counts)

        tool_calls += len(calls)
        outputs = []
        for fc in calls:
            tool_counts[fc.get("name")] = tool_counts.get(fc.get("name"), 0) + 1
            output = toolbox.dispatch(fc.get("name"), fc.get("arguments"))
            outputs.append({
                "type": "function_call_output",
                # call_id pairs the output with its call; fall back to the
                # item id if the model/gateway omitted it.
                "call_id": fc.get("call_id") or fc.get("id") or "",
                "output": output,
            })
        # Budget disclosure rides with every tool-output batch (BUDGET_NOTE,
        # above): the model plans its exit instead of being cut off.
        outputs.append({"role": "user", "content": BUDGET_NOTE.format(
            remaining=max_tool_turns - turns, max_tool_turns=max_tool_turns)})
        next_input = outputs
