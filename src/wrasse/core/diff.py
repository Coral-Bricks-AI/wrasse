"""Unified-diff hunk parsing: which new-file line numbers a review may comment on.

GitHub rejects inline review comments whose ``line`` falls outside the diff
hunks for that file (HTTP 422), so every finding has to be checked against the
RIGHT side of the diff before posting. ``parse_hunks`` returns per-path hunks
with inclusive new-side ranges; ``line_in_hunks`` is the membership check.
Findings outside every hunk are still reported, but in the review summary body
instead of as inline threads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    """Inclusive new-side line range [new_start, new_start + new_count - 1]."""

    new_start: int
    new_count: int


def parse_hunks(diff: str) -> dict[str, list[Hunk]]:
    """Parse a unified diff into per-file new-side hunk ranges.

    Deleted files (``+++ /dev/null``) and binary files produce no entry.
    """
    hunks: dict[str, list[Hunk]] = {}
    path: str | None = None
    for line in diff.splitlines():
        m = _HUNK_RE.match(line)
        if m is not None:
            if path is None:
                continue  # hunk before any +++ header: malformed, ignore
            start = int(m.group(1))
            count = int(m.group(2) or 1)
            if count > 0:  # count 0 = file emptied out: nothing commentable
                hunks.setdefault(path, []).append(Hunk(start, count))
            continue
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+++ "):
            path = None  # "/dev/null": deleted file, nothing commentable
        elif line.startswith("diff --git "):
            path = None  # new file section: wait for its +++ b/ header
    return hunks


def line_in_hunks(hunks: dict[str, list[Hunk]], path: str, line: int) -> bool:
    """True if ``line`` of ``path`` is inside a hunk the diff touches."""
    for hunk in hunks.get(path, ()):
        if hunk.new_start <= line <= hunk.new_start + hunk.new_count - 1:
            return True
    return False


def clamp_end_to_hunk(
    hunks: dict[str, list[Hunk]], path: str, line: int, end_line: int | None
) -> int | None:
    """Clamp a finding's end_line to the hunk containing ``line``.

    GitHub rejects a multi-line review comment whose range crosses a hunk
    boundary (422), so a span end past the hunk's last new-side line folds
    back to it. ``line`` itself must be in a hunk (the caller routes
    out-of-hunk findings to the summary). No end_line or an end inside the
    hunk passes through unchanged.
    """
    if end_line is None:
        return None
    for hunk in hunks.get(path, ()):
        last = hunk.new_start + hunk.new_count - 1
        if hunk.new_start <= line <= last:
            return min(end_line, last)
    return None  # line not in any hunk: caller routes this to the summary


def resolve_diff_path(path: str, hunks: dict[str, list[Hunk]]) -> tuple[str, bool]:
    """Map a model-emitted path onto the diff's actual file paths.

    The prompt says "path as it appears in the diff", so the model may emit
    the header form `b/<path>` (or `a/<path>`) instead of the bare path.
    But a repo can also contain a real directory named `a/` or `b/` —
    stripping unconditionally mangled a real file `a/x.py` into `x.py`
    (an earlier review round). Strip the prefix only when the stripped form matches an
    actual file in the diff.

    Returns (resolved_path, ambiguous). ambiguous=True means both the
    literal and the stripped form are real diff files (the repo really
    contains `x.py` AND `b/x.py`): the model's intent is unknowable, so
    the caller must demote the finding to the summary instead of guessing
    (an earlier review round). An unknown path returns (path, False) — it will fail
    hunk lookup and land in the summary, visible rather than wrong.
    """
    if path in hunks:
        stripped = path[2:] if path.startswith(("a/", "b/")) else None
        if stripped is not None and stripped in hunks:
            return path, True  # collision: x.py AND b/x.py both changed
        return path, False
    if path.startswith(("a/", "b/")) and path[2:] in hunks:
        return path[2:], False
    return path, False
