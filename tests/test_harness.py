"""Unit tests for the v1 review harness (an earlier round): parser, diff mapping, filters.

Canned model output only — the parser is what these tests pin down, not the
model. End-to-end runs happen via --dry-run against the real gateway and
planted-bug draft PRs (instant ground truth), not here.
"""

from __future__ import annotations

import json

import sys
from pathlib import Path

from wrasse.core import diff as difflines
from wrasse.core import contract as F
from wrasse.platforms import github as fetch
from wrasse.platforms.publish import comment_body  # noqa: E402

# ---------------------------------------------------------------- parse


def test_parse_fenced_json():
    text = (
        "Here is my review:\n```json\n"
        '[{"path": "a.py", "line": 3, "severity": "high",'
        ' "title": "Off by one", "detail": "range excludes last element"}]\n'
        "```\nDone."
    )
    out = F.parse_findings(text)
    assert len(out) == 1
    f = out[0]
    assert (f.path, f.line, f.severity, f.title) == ("a.py", 3, "high", "Off by one")
    assert f.detail == "range excludes last element"
    assert len(f.fid) == 10


def test_parse_last_fenced_block_wins():
    text = (
        "```json\n[]\n```\n"
        "On reflection: ```json\n"
        '[{"path": "b.py", "line": 1, "severity": "low", "title": "T", "detail": ""}]\n'
        "```"
    )
    out = F.parse_findings(text)
    assert len(out) == 1 and out[0].path == "b.py"


def test_parse_bare_array_fallback():
    text = 'no fences at all: [{"path": "c.py", "line": 2, "severity": "medium", "title": "X"}]'
    out = F.parse_findings(text)
    assert len(out) == 1 and out[0].path == "c.py" and out[0].severity == "medium"


def test_parse_bare_array_ignores_prose_brackets():
    # A citation "[1]" and a markdown "[link]" must not shadow the real
    # array, and must not make the whole candidate undecodable (the platform
    # an earlier round: first-'['-to-last-']' slicing failed this shape).
    text = (
        "See [1] and also [the docs]. My findings: "
        '[{"path": "c.py", "line": 2, "severity": "medium", "title": "X"}] '
        "as noted in [2]."
    )
    out = F.parse_findings(text)
    assert len(out) == 1 and out[0].path == "c.py"


def test_parse_bare_citation_only_is_inconclusive():
    # A reply whose only bracket list is a citation never answered the
    # contract: that is a parse failure (None), NOT a clean "no bugs".
    assert F.parse_findings("Looks good overall. See [1] for details.") is None


def test_coerce_strips_ab_prefix():
    # "path as it appears in the diff" includes the a//b/ header prefix;
    # resolve_diff_path maps it onto real diff files — see the dedicated
    # tests there. Here: the parser itself leaves the path untouched.
    text = '```json\n[{"path": "b/ml/x.py", "line": 3, "severity": "low", "title": "T"}]\n```'
    out = F.parse_findings(text)
    assert out[0].path == "b/ml/x.py"


def test_bare_empty_array_answer_is_valid():
    # An ENTIRE bare [] answer (fence dropped) is a clean "no bugs" reply —
    # it must not regress into an inconclusive failure status (an earlier review round).
    assert F.parse_findings("[]") == []
    assert F.parse_findings("\n  []  \n") == []
    # ...but [] embedded in other prose stays inconclusive.
    assert F.parse_findings("I looked, and [] was my conclusion, see [1].") is None


def test_select_findings_cap_drops_lowest_severity():
    items = [F.Finding(path="a.py", line=i, severity="low", title=f"t{i}")
             for i in range(30)]
    items.append(F.Finding(path="a.py", line=99, severity="high", title="the important one"))
    selected, over_cap = F.select_findings(items, max_findings=30)
    assert over_cap == 1
    assert selected[0].severity == "high" and selected[0].title == "the important one"
    assert all(f.severity == "low" for f in selected[1:])
    # Stable within a severity: model order preserved.
    assert [f.title for f in selected[1:3]] == ["t0", "t1"]


def test_parse_hunks_section_heading():
    # Hunk headers may carry a trailing section heading; re.match anchors at
    # the start, so the heading must not break recognition (an earlier review round).
    d = ("diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
         "@@ -1,2 +3,2 @@ def foo():\n ctx\n+new\n")
    assert [(h.new_start, h.new_count) for h in difflines.parse_hunks(d)["f.py"]] == [(3, 2)]


def test_resolve_diff_path():
    # Prefix stripped only when the stripped form is a real diff file;
    # a repo file that literally lives at a/x.py survives (an earlier review round).
    hunks = difflines.parse_hunks(
        "diff --git a/a/x.py b/a/x.py\n--- a/a/x.py\n+++ b/a/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    assert difflines.resolve_diff_path("b/src/m.py", hunks) == ("src/m.py", False)  # header form
    assert difflines.resolve_diff_path("a/x.py", hunks) == ("a/x.py", False)        # real path, kept
    assert difflines.resolve_diff_path("b/a/x.py", hunks) == ("a/x.py", False)      # real path via header
    assert difflines.resolve_diff_path("src/m.py", hunks) == ("src/m.py", False)     # bare, kept
    assert difflines.resolve_diff_path("nope.py", hunks) == ("nope.py", False)       # unknown, untouched


def test_resolve_diff_path_collision_is_ambiguous():
    # The repo really contains BOTH x.py and b/x.py: a model emitting
    # "b/x.py" is unknowable — header form for x.py, or the literal file.
    # Must be flagged ambiguous, never guessed (an earlier review round).
    hunks = difflines.parse_hunks(
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/b/x.py b/b/x.py\n--- a/b/x.py\n+++ b/b/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    assert difflines.resolve_diff_path("b/x.py", hunks) == ("b/x.py", True)
    assert difflines.resolve_diff_path("x.py", hunks) == ("x.py", False)  # bare is unambiguous


def test_rekey_makes_normalized_duplicates_share_fid():
    # x.py and header-form b/x.py for one bug: distinct fids at parse time,
    # same fid after normalization + rekey — one thread, not two (an earlier review round).
    a = F.Finding(path="x.py", line=3, severity="low", title="Off by one")
    b = F.Finding(path="b/x.py", line=3, severity="low", title="off by one")
    assert a.fid != b.fid
    b.path = "x.py"
    b.rekey()
    assert a.fid == b.fid


def test_dedupe_prefers_higher_severity():
    # Same bug emitted twice with different severities after normalization:
    # the HIGH must survive, not whichever came first (an earlier review round).
    low = F.Finding(path="x.py", line=3, severity="low", title="Off by one")
    high = F.Finding(path="x.py", line=3, severity="high", title="off by one")
    out = F.dedupe_findings([low, high])
    assert len(out) == 1 and out[0].severity == "high"
    # Order-independent: high first, low second — still one high.
    out = F.dedupe_findings([high, low])
    assert len(out) == 1 and out[0].severity == "high"


def test_dedupe_stable_among_equal_severities():
    a = F.Finding(path="x.py", line=3, severity="low", title="A")
    b = F.Finding(path="x.py", line=3, severity="low", title="a")
    out = F.dedupe_findings([a, b])
    assert len(out) == 1 and out[0].title == "A"


def test_dedupe_keeps_first_mention_position_with_upgraded_severity():
    # Cap ties must follow the bug's FIRST mention, not the winning
    # duplicate's position: A emitted low first, upgraded to high by the
    # later duplicate — A stays ahead of B (an earlier review round).
    a_low = F.Finding(path="x.py", line=1, severity="low", title="A")
    b_high = F.Finding(path="y.py", line=2, severity="high", title="B")
    a_high = F.Finding(path="x.py", line=1, severity="high", title="a")
    out = F.dedupe_findings([a_low, b_high, a_high])
    assert len(out) == 2
    assert [f.title for f in out] == ["A", "B"]      # first-mention order
    assert [f.severity for f in out] == ["high", "high"]  # best severity kept


def test_parse_empty_array_is_valid_no_bugs():
    assert F.parse_findings("```json\n[]\n```") == []


def test_parse_garbage_is_parse_failure_not_empty():
    # None = inconclusive (caller must NOT post "no bugs found"); [] is
    # reserved for a valid empty reply (an earlier review round: truncation ≠ clean).
    assert F.parse_findings("the model rambled about the weather") is None
    assert F.parse_findings("```json\n{not json}\n```") is None
    assert F.parse_findings("") is None
    assert F.parse_findings('```json\n[{"path":') is None  # truncated reply


def test_parse_drops_invalid_entries_keeps_valid():
    entries = [
        {"path": "", "line": 1, "severity": "low", "title": "T"},       # empty path
        {"path": "a.py", "line": 0, "severity": "low", "title": "T"},  # line < 1
        {"path": "a.py", "line": "x", "severity": "low", "title": "T"},  # non-numeric
        {"path": "a.py", "line": 2, "severity": "low"},                # no title
        "not a dict",
        {"path": "a.py", "line": 4, "severity": "low", "title": "ok"},
    ]
    out = F.parse_findings("```json\n" + json.dumps(entries) + "\n```")
    assert [f.line for f in out] == [4]


def test_parse_severity_aliases_and_unknown():
    text = json.dumps([
        {"path": "a.py", "line": 1, "severity": "critical", "title": "t1"},
        {"path": "a.py", "line": 2, "severity": "weird", "title": "t2"},
    ])
    out = F.parse_findings("```json\n" + text + "\n```")
    assert [f.severity for f in out] == ["high", "medium"]


def test_parse_dedups_same_path_line_title():
    text = json.dumps([
        {"path": "a.py", "line": 5, "severity": "low", "title": "Same Title"},
        {"path": "a.py", "line": 5, "severity": "low", "title": "same title"},
        {"path": "a.py", "line": 6, "severity": "low", "title": "same title"},
    ])
    out = F.parse_findings("```json\n" + text + "\n```")
    assert len(out) == 2


def test_fid_stable_for_same_finding():
    a = F.Finding(path="a.py", line=3, severity="high", title="Off by one")
    b = F.Finding(path="a.py", line=3, severity="medium", title="off by one")  # severity drifts
    assert a.fid == b.fid


# ---------------------------------------------------------------- path filter


def test_keep_path_filters():
    for bad in ("package-lock.json", "yarn.lock", "x/y.min.js", "sourcemap.map",
                "datasets/foo.csv", "graphify-out/graphs/x.json", ".venv/lib/x.py"):
        assert not F.keep_path(bad), bad
    for good in ("src/wrasse/core/review.py", "backend/src/index.ts", "app.css"):
        assert F.keep_path(good), good


# ---------------------------------------------------------------- diff hunks

DIFF = """\
diff --git a/ml/x.py b/ml/x.py
index 111..222 100644
--- a/ml/x.py
+++ b/ml/x.py
@@ -1,4 +1,6 @@
 ctx
-old
+new
+added
 ctx
 ctx
@@ -20,3 +22,4 @@
 ctx
+another
 ctx
 ctx
diff --git a/gone.txt b/gone.txt
index 333..000 100644
--- a/gone.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-bye
-bye
diff --git a/img.bin b/img.bin
Binary files differ
"""


def test_parse_hunks_new_side_ranges():
    hunks = difflines.parse_hunks(DIFF)
    assert set(hunks) == {"ml/x.py"}
    ranges = [(h.new_start, h.new_count) for h in hunks["ml/x.py"]]
    assert ranges == [(1, 6), (22, 4)]


def test_line_in_hunks():
    hunks = difflines.parse_hunks(DIFF)
    assert difflines.line_in_hunks(hunks, "ml/x.py", 1)
    assert difflines.line_in_hunks(hunks, "ml/x.py", 6)
    assert difflines.line_in_hunks(hunks, "ml/x.py", 25)
    assert not difflines.line_in_hunks(hunks, "ml/x.py", 7)      # gap between hunks
    assert not difflines.line_in_hunks(hunks, "ml/x.py", 100)
    assert not difflines.line_in_hunks(hunks, "other.py", 1)     # untouched file


def test_clamp_end_to_hunk():
    hunks = difflines.parse_hunks(DIFF)
    # DIFF's first hunk is lines 1-6, the second starts at 22.
    assert difflines.clamp_end_to_hunk(hunks, "ml/x.py", 2, 99) == 6
    assert difflines.clamp_end_to_hunk(hunks, "ml/x.py", 2, 4) == 4
    assert difflines.clamp_end_to_hunk(hunks, "ml/x.py", 2, None) is None
    assert difflines.clamp_end_to_hunk(hunks, "ml/x.py", 7, 9) is None  # gap: no hunk
    assert difflines.clamp_end_to_hunk(hunks, "other.py", 1, 3) is None


def test_parse_hunks_explicit_zero_vs_omitted_count():
    # Pinned after a the platform false positive (an earlier round): an explicit ",0"
    # count (pure deletion) must drop the hunk, an omitted count means 1
    # per the unified-diff spec — group(2) is the *string* "0", which is
    # truthy, so `int(m.group(2) or 1)` gives 0, not 1.
    zero = ("diff --git a/x.txt b/x.txt\n--- a/x.txt\n+++ b/x.txt\n"
            "@@ -1,3 +2,0 @@\n-a\n-b\n-c\n")
    assert difflines.parse_hunks(zero) == {}
    one = ("diff --git a/x.txt b/x.txt\n--- a/x.txt\n+++ b/x.txt\n"
           "@@ -1 +2 @@\n-a\n+b\n")
    assert [(h.new_start, h.new_count) for h in difflines.parse_hunks(one)["x.txt"]] == [(2, 1)]


# ---------------------------------------------------------------- comment body


def test_comment_body_carries_marker():
    f = F.Finding(path="a.py", line=9, severity="high", title="T", detail="D")
    body = comment_body(f, "0123456789abcdef")
    assert body.startswith("**[high] T**")
    assert "<!-- wrasse id=" in body
    assert f"severity=high kind=bug sha=0123456789 -->" in body


def test_comment_body_marker_carries_end_line():
    f = F.Finding(path="a.py", line=9, severity="high", title="T",
                  end_line=14)
    body = comment_body(f, "0123456789abcdef")
    assert "end_line=14 -->" in body
    single = F.Finding(path="a.py", line=9, severity="high", title="T")
    assert "end_line=" not in comment_body(single, "0123456789abcdef")


def test_finding_end_line_normalization():
    # absent -> single line; inverted range collapses to single line
    f = F.Finding(path="a.py", line=10, severity="low", title="t")
    assert f.end_line is None and f.span() == (10, 10)
    g = F.Finding(path="a.py", line=10, severity="low", title="t", end_line=8)
    assert g.end_line is None and g.span() == (10, 10)
    h = F.Finding(path="a.py", line=10, severity="low", title="t", end_line=12)
    assert h.span() == (10, 12)


def test_parse_findings_reads_end_line():
    text = ('```json\n[{"path": "a.py", "line": 5, "end_line": 9, '
            '"severity": "high", "title": "T", "detail": "D"}]\n```')
    out = F.parse_findings(text)
    assert out and out[0].span() == (5, 9)
    flat = ('```json\n[{"path": "a.py", "line": 5, "end_line": 3, '
            '"severity": "high", "title": "T", "detail": "D"}]\n```')
    assert F.parse_findings(flat)[0].end_line is None
    # Non-finite (an earlier review round: 1e999 parsed as float inf would raise
    # OverflowError in int() and take the whole review down).
    inf = ('```json\n[{"path": "a.py", "line": 5, "end_line": 1e999, '
           '"severity": "high", "title": "T", "detail": "D"}]\n```')
    assert F.parse_findings(inf)[0].end_line is None


def test_thread_spans_raise_time_and_current_anchors():
    # Marker span is the raise-time range; current line is its own anchor.
    t = {"line": 80, "original_line": 10, "marker_start": 9, "marker_end": 14}
    assert F.thread_spans(t) == [(9, 14), (80, 80)]
    # Pre-span marker: the raised position (original when moved) + current.
    old = {"line": 80, "original_line": 10, "marker_start": None, "marker_end": None}
    assert F.thread_spans(old) == [(10, 10), (80, 80)]
    # Unmoved single-line thread: one span, not two.
    same = {"line": 10, "original_line": 10, "marker_start": None, "marker_end": None}
    assert F.thread_spans(same) == [(10, 10)]
    assert F.thread_spans({"path": "a.py"}) == []
    assert F._spans_overlap((1, 5), (5, 9)) and not F._spans_overlap((1, 4), (5, 9))


def test_system_prompt_pins_output_contract():
    assert "```json" in F.SYSTEM_PROMPT
    assert "[]\n" in F.SYSTEM_PROMPT or "[]" in F.SYSTEM_PROMPT
    assert "bugs only" in F.SYSTEM_PROMPT.lower()
    # v2.7 (the previous reviewer-transferables, an earlier round): the prompt pins the review
    # discipline mined from the the previous reviewer automation's own prompt — value
    # tracing, verified dismissal, documented-workflow findings, and the
    # hunk-by-hunk completeness pass. A prompt regression here is a
    # version step, so it is pinned like the contract.
    assert "Trace every changed value" in F.SYSTEM_PROMPT
    assert "verify each by reading the code" in F.SYSTEM_PROMPT
    assert "documented workflow" in F.SYSTEM_PROMPT
    assert "re-walk the diff hunk by hunk" in F.SYSTEM_PROMPT
    assert "concrete fix" in F.SYSTEM_PROMPT


# ---------------------------------------------------------------- summary


class _Reply:
    elapsed_s = 1.2
    prompt_tokens = 100
    completion_tokens = 5


def test_build_summary_counts_each_finding_once():
    from wrasse.core import review as R

    inline = [F.Finding(path="a.py", line=1, severity="low", title="t1")]
    overflow = [F.Finding(path="a.py", line=99, severity="low", title="t2",
                         detail="outside hunks")]
    s = R.build_summary("m", _Reply(), inline, overflow, dropped_paths=1, dropped_cap=2)
    assert "Found **2** bug(s): 1 inline below, 1 outside the diff hunks" in s
    assert "outside hunks" in s              # overflow keeps its detail
    assert "ignored paths" in s             # dropped_paths surfaced
    assert "--max-findings cap" in s        # dropped_cap surfaced


def test_build_summary_no_bugs():
    from wrasse.core import review as R

    s = R.build_summary("m", _Reply(), [], [], dropped_paths=0, dropped_cap=0)
    assert "No bugs found in this diff." in s


# ------------------------------------------------- thread dedupe + resolve


def _thread(tid, path, line, outdated=False, created="2026-10-15T10:00:00Z",
            marker_start=None, marker_end=None):
    return {
        "thread_id": tid,
        "comment_id": 100,
        "path": path,
        "line": line,
        "original_line": line,
        "marker_start": marker_start,
        "marker_end": marker_end,
        "outdated": outdated,
        "created_at": created,
        "title": "already raised",
    }


def test_split_already_raised_skips_open_positions():
    threads = [
        _thread("t1", "a.py", 3),
        _thread("t2", "a.py", 10, outdated=True),  # outdated: doesn't block
        _thread("t3", "b.py", 7),
    ]
    fresh = [
        F.Finding(path="a.py", line=3, severity="high", title="same position"),
        F.Finding(path="a.py", line=10, severity="high", title="re-flagged"),
        F.Finding(path="c.py", line=1, severity="low", title="new"),
    ]
    out, withheld = F.split_already_raised(fresh, threads)
    assert [(f.path, f.line) for f in out] == [("a.py", 10), ("c.py", 1)]
    assert [f.line for f in withheld] == [3]


def test_split_already_raised_no_threads_keeps_all():
    fresh = [F.Finding(path="a.py", line=3, severity="high", title="t")]
    out, withheld = F.split_already_raised(fresh, [])
    assert out == fresh and withheld == []


def test_split_already_raised_span_overlap_withholds():
    # an earlier round adjacent-line survivor: an open thread raised on the block
    # 1850-1860 must swallow a re-flag at 1852 — and still leave a
    # genuinely disjoint finding alone.
    threads = [_thread("t1", "registry.py", 1860,
                       marker_start=1850, marker_end=1860)]
    fresh = [
        F.Finding(path="registry.py", line=1852, severity="high",
                  title="same_box dedup re-flag"),
        F.Finding(path="registry.py", line=1840, severity="low", title="disjoint"),
        F.Finding(path="registry.py", line=1860, severity="low", title="edge touch"),
    ]
    out, withheld = F.split_already_raised(fresh, threads)
    assert [f.line for f in withheld] == [1852, 1860]
    assert [f.line for f in out] == [1840]


def test_split_already_raised_span_repost_at_start_withholds():
    # an earlier review round: GitHub records a range comment's `line` as the
    # range END — the START lives only in the marker. A re-derivation at
    # the original start line must dedupe.
    threads = [_thread("t1", "a.py", 14, marker_start=9, marker_end=14)]
    fresh = [F.Finding(path="a.py", line=9, severity="high", title="re-flag at start")]
    out, withheld = F.split_already_raised(fresh, threads)
    assert withheld == fresh and out == []


def test_split_already_raised_moved_thread_does_not_fabricate_a_span():
    # an earlier review round: a thread that moved 10 -> 80 covers exactly its
    # raised position and its current line — NOT every line between.
    threads = [_thread("t1", "a.py", 80)]
    threads[0]["original_line"] = 10
    fresh = [
        F.Finding(path="a.py", line=50, severity="high", title="unrelated middle"),
        F.Finding(path="a.py", line=80, severity="low", title="at current position"),
    ]
    out, withheld = F.split_already_raised(fresh, threads)
    assert [f.line for f in out] == [50]
    assert [f.line for f in withheld] == [80]


def test_split_already_raised_adjacent_single_lines_still_post():
    # Two single-line threads on neighboring lines do NOT overlap —
    # withholding there would guess (a distinct bug may live next door).
    threads = [_thread("t1", "a.py", 1851)]
    fresh = [F.Finding(path="a.py", line=1852, severity="high", title="adjacent")]
    out, withheld = F.split_already_raised(fresh, threads)
    assert out == fresh and withheld == []


def test_plan_thread_resolution_marks_outdated_and_duplicates():
    threads = [
        _thread("old", "a.py", 3, created="2026-10-15T09:00:00Z"),
        _thread("new", "a.py", 3, created="2026-10-15T10:00:00Z"),
        _thread("stale", "b.py", 7, outdated=True),
        _thread("keep", "c.py", 9),
    ]
    plan = F.plan_thread_resolution(threads)
    ids = {t["thread_id"] for t, _ in plan}
    assert ids == {"new", "stale"}  # oldest dup kept, distinct line untouched
    notes = {t["thread_id"]: n for t, n in plan}
    assert "changed in a newer head" in notes["stale"]
    assert "Duplicate" in notes["new"]


def test_plan_thread_resolution_empty_when_clean():
    threads = [_thread("t1", "a.py", 3)]
    assert F.plan_thread_resolution(threads) == []


def test_build_user_prompt_lists_already_raised():
    prompt = F.build_user_prompt(
        "diff --git a/a b/a\n", "", ["- `a.py`:3 — already raised"])
    assert "do NOT re-report" in prompt
    assert "`a.py`:3" in prompt
    # without the section, the prompt must be unchanged in shape
    assert "already raised" not in F.build_user_prompt("diff --git a/a b/a\n", "")


def test_build_user_prompt_carries_pr_intent():
    """The author's stated purpose is the cheapest false-positive guard
    (an earlier round): a diff reviewed blind manufactures findings on intended
    behavior. Present when given, absent when not."""
    prompt = F.build_user_prompt(
        "diff --git a/a.py b/a.py\n", "",
        pr_intent={"title": "Fix trial amounts", "body": "Orders by id asc is intended."})
    assert "The author's stated intent" in prompt
    assert "**Title:** Fix trial amounts" in prompt
    assert "Orders by id asc is intended." in prompt
    assert "stated intent" not in F.build_user_prompt("diff --git a/a.py b/a.py\n", "")
    # Empty title+body omit the section too (no PR, or no description)
    assert "stated intent" not in F.build_user_prompt(
        "diff --git a/a.py b/a.py\n", "", pr_intent={"title": "", "body": ""})


def test_build_user_prompt_truncates_pr_intent_body():
    long_body = "x" * (F.PR_INTENT_MAX_CHARS + 500)
    prompt = F.build_user_prompt(
        "diff --git a/a.py b/a.py\n", "", pr_intent={"title": "t", "body": long_body})
    assert "(description truncated)" in prompt
    assert len(prompt) < len(long_body) + 2000  # the body did not ride in whole


def test_build_user_prompt_carries_repo_map():
    """First-turn orientation (an earlier round): the layout section teaches real
    paths so greps scope correctly from turn 1 — omitted when empty."""
    prompt = F.build_user_prompt(
        "diff --git a/a.py b/a.py\n", "", repo_map="ml/\n  platform\nweb/")
    assert "Repository layout (top two levels)" in prompt
    assert "ml/" in prompt
    assert "layout" not in F.build_user_prompt("diff --git a/a.py b/a.py\n", "")


def test_split_resolved_rederivations_matches_raised_and_current_lines():
    """The hard filter behind an earlier round data point (an earlier round): all
    three round-4 findings were re-derivations of resolved threads at the
    SAME positions with near-identical claims. Tier 2 withholds a
    coordinate match whose title is a rephrase of the resolved claim."""
    def finding(path, line, title):
        return F.Finding(path=path, line=line, severity="medium", title=title)

    history = [
        # raised at 30, code has since moved to 64
        {"path": "domain/billing/trial-amount.ts", "line": 64, "original_line": 30,
         "fid": "", "title": "grantedTrialCents orders by id asc, not the actual grant time"},
        # unchanged position, exact same claim
        {"path": "shared/email/welcome-email.ts", "line": 60, "original_line": 60,
         "fid": "", "title": "granted state with trialCents === 0 falls into the disabled branch"},
    ]
    fresh, withheld = F.split_resolved_rederivations(
        [finding("domain/billing/trial-amount.ts", 64,
                 "grantedTrialCents orders by id asc, not the actual grant times"),  # rephrase
         finding("domain/billing/trial-amount.ts", 30,
                 "grantedTrialCents orders by id asc, not the actual grant time"),   # moved + same
         finding("shared/email/welcome-email.ts", 60,
                 "granted state with trialCents === 0 falls into the disabled branch"),  # verbatim
         finding("shared/email/welcome-email.ts", 61, "something else entirely"),   # adjacent — passes
         finding("other/file.ts", 5, "anything")],
        history)
    assert [f.line for f in withheld] == [64, 30, 60]
    assert [f.line for f in fresh] == [61, 5]


def test_split_resolved_rederivations_fid_is_tier_one():
    """Tier 1: the resolved thread's marker fid equals the finding's fid
    — the same claim verbatim, no coordinate or title heuristics needed."""
    f = F.Finding(path="a.py", line=3, severity="low", title="off by one")
    history = [{"path": "a.py", "line": 3, "original_line": 3, "fid": f.fid,
                "title": "totally different stored title"}]
    fresh, withheld = F.split_resolved_rederivations([f], history)
    assert fresh == [] and len(withheld) == 1


def test_split_resolved_rederivations_posts_a_different_claim_at_a_reused_line():
    """an earlier review round: six distinct claims lived at one line on an earlier round — a
    reused coordinate must not swallow an UNRELATED bug. A coordinate match
    with a dissimilar title passes and becomes a real thread."""
    def finding(title):
        return F.Finding(path="welcome-email.ts", line=60, severity="medium", title=title)
    history = [{"path": "welcome-email.ts", "line": 60, "original_line": 60,
                "fid": "", "title": "granted state with trialCents === 0 gets the wrong CTA"}]
    fresh, withheld = F.split_resolved_rederivations(
        [finding("SQL injection in the verify-note builder")], history)
    assert len(fresh) == 1 and withheld == []


def test_split_resolved_rederivations_empty_history_is_noop():
    def finding(path, line):
        return F.Finding(path=path, line=line, severity="low", title="t")
    fresh, withheld = F.split_resolved_rederivations(
        [finding("a.py", 1), finding("b.py", 2)], None)
    assert len(fresh) == 2 and withheld == []


def test_split_resolved_rederivations_span_overlap_withholds():
    # A resolved thread raised on the span 1850-1860 (marker span)
    # withholds a rephrased re-derivation at 1852 — the tier-2 coordinate
    # test is span overlap now, not exact-line membership.
    def finding(line, title):
        return F.Finding(path="registry.py", line=line, severity="medium",
                         title=title)
    history = [{"path": "registry.py", "line": 1860, "original_line": 1860,
                "marker_start": 1850, "marker_end": 1860, "fid": "",
                "title": "same_box lookup skips the newest registry row"}]
    fresh, withheld = F.split_resolved_rederivations(
        [finding(1852, "same_box lookup misses the newest registry entry")],
        history)
    assert fresh == [] and len(withheld) == 1
    # same span, unrelated claim: still posted
    fresh, withheld = F.split_resolved_rederivations(
        [finding(1852, "SQL injection in the lookup builder")], history)
    assert len(fresh) == 1 and withheld == []
    # disjoint span, same claim: still posted (the bug moved)
    fresh, withheld = F.split_resolved_rederivations(
        [finding(1900, "same_box lookup misses the newest registry entry")],
        history)
    assert len(fresh) == 1 and withheld == []


def test_split_resolved_rederivations_skips_entries_without_a_line():
    def finding(path, line):
        return F.Finding(path=path, line=line, severity="low", title="t")
    history = [{"path": "a.py", "line": None, "original_line": None, "fid": ""}]
    fresh, withheld = F.split_resolved_rederivations([finding("a.py", 1)], history)
    assert len(fresh) == 1 and withheld == []


def test_build_summary_reports_dropped_dupes():
    from wrasse.core import review as R

    dupes = [F.Finding(path="a.py", line=3, severity="medium", title="dupe")]
    s = R.build_summary("m", _Reply(), [], [], dropped_paths=0, dropped_cap=0,
                        dropped_dupes=dupes)
    assert "No NEW bugs" in s
    assert "`a.py:3`" in s         # withheld finding stays visible
    assert "dupe" in s


def test_build_summary_new_findings_plus_dupes():
    from wrasse.core import review as R

    inline = [F.Finding(path="b.py", line=2, severity="low", title="t1")]
    dupes = [F.Finding(path="a.py", line=3, severity="medium", title="dupe")]
    s = R.build_summary("m", _Reply(), inline, [], dropped_paths=0, dropped_cap=0,
                        dropped_dupes=dupes)
    assert "Found **1** bug(s)" in s
    assert "withheld" in s and "`a.py:3`" in s


# ---------------------------------------------------------------- 422 fallback


def test_comment_position_single_vs_span():
    from wrasse.platforms import publish as post_mod

    single = F.Finding(path="a.py", line=7, severity="high", title="T")
    assert post_mod._comment_position(single) == {
        "path": "a.py", "line": 7, "side": "RIGHT"}
    span = F.Finding(path="a.py", line=7, severity="high", title="T", end_line=12)
    assert post_mod._comment_position(span) == {
        "path": "a.py", "line": 12, "side": "RIGHT",
        "start_line": 7, "start_side": "RIGHT"}


# ------------------------------------------------- summary minimization (B3)

def _summary(node_id, sha, minimized=False):
    return {"node_id": node_id, "sha": sha, "minimized": minimized}


def test_summaries_to_minimize_all_resolved_round_goes():
    summaries = [_summary("s1", "aaa"), _summary("s2", "bbb")]
    open_threads, resolved_threads = [], [{"sha": "aaa"}, {"sha": "aaa"}]
    out = fetch.summaries_to_minimize(summaries, open_threads, resolved_threads)
    assert [s["node_id"] for s in out] == ["s1"]


def test_summaries_to_minimize_open_round_stays():
    summaries = [_summary("s1", "aaa")]
    out = fetch.summaries_to_minimize(
        summaries, [{"sha": "aaa"}], [{"sha": "aaa"}])
    assert out == []


def test_summaries_to_minimize_clean_review_stays():
    # A round that raised no threads is the head's reviewed-record marker:
    # it is not noise, it is the cheapest idempotence signal.
    assert fetch.summaries_to_minimize([_summary("s1", "aaa")], [], []) == []


def test_summaries_to_minimize_skips_minimized_and_markerless():
    summaries = [_summary("s1", "aaa", minimized=True), _summary("s2", "")]
    out = fetch.summaries_to_minimize(
        summaries, [], [{"sha": "aaa"}, {"sha": ""}])
    assert out == []


def test_own_thread_record_parses_marker_sha(monkeypatch):
    from wrasse.platforms import github as fetch_mod
    node = {
        "id": "T_", "isOutdated": False, "isResolved": False,
        "comments": {"nodes": [{
            "databaseId": 1, "author": {"login": "github-actions"},
            "path": "a.py", "line": 3, "originalLine": 3,
            "createdAt": "2026-10-16T00:00:00Z",
            "body": "**[high] T**\n\n<!-- wrasse id=ab12 severity=high "
                    "kind=bug sha=deadbeef00 start_line=3 end_line=9 -->",
        }]},
    }
    rec = fetch_mod._own_thread_record(node, resolved=False)
    assert rec["sha"] == "deadbeef00"
    assert rec["marker_start"] == 3 and rec["marker_end"] == 9
    assert F.thread_spans(rec) == [(3, 9)]


def test_head_already_reviewed_commit_id_fallback(monkeypatch):
    # Minimization replaces a review's REST body with GitHub's placeholder,
    # which drops the head marker — commit_id matching survives it.
    from wrasse.platforms import github as fetch_mod

    def fake_gh_get(path, token, accept="application/vnd.github+json"):
        return json.dumps([
            {"user": {"login": "github-actions[bot]"},
             "commit_id": "0123456789abcdef",
             "body": "> [!NOTE]\n**Newer findings are available below.**"},
        ])

    monkeypatch.setattr(fetch_mod, "gh_get", fake_gh_get)
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "0123456789abcdef") is True
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "ffffffffffffffff") is False


def test_post_review_422_fallback_preserves_findings(monkeypatch):
    from wrasse.platforms import github as fetch_mod
    from wrasse.platforms import publish as post_mod

    calls = []

    def fake_gh_post(path, token, payload):
        calls.append(payload)
        if payload["comments"]:
            raise fetch_mod.GitHubError(422, path, "position out of range")
        return {"id": 1}

    monkeypatch.setattr(post_mod, "gh_post", fake_gh_post)
    findings = [F.Finding(path="a.py", line=1, severity="high", title="T", detail="D")]
    post_mod.post_review("tok", "r", 1, "0123456789abcdef", "summary", findings)

    assert len(calls) == 2
    assert calls[0]["comments"]  # first attempt carried the inline comment
    assert calls[1]["comments"] == []  # fallback posts summary-only
    assert "T — D" in calls[1]["body"]  # but the finding text survives
    assert "head=0123456789 v=" in calls[1]["body"]  # head marker + version present


# ---------------------------------------------------------------- CLI guard


def test_diff_file_without_dry_run_is_rejected(tmp_path):
    # A bare --diff-file run would otherwise reach post_review with
    # pr_number=None and fail on a /pulls/None 404 (an earlier review round).
    import pytest

    from wrasse.core import review as R

    d = tmp_path / "x.diff"
    d.write_text("diff --git a/a b/a\n")
    with pytest.raises(SystemExit):
        R.main(["--diff-file", str(d)])


# ------------------------------------------------- self-retraction guard
# an earlier round pre-A/B: entries whose own text concludes "no bug" posted
# anyway on an earlier round (15 of one round's 25) and an earlier round (10+). The guard drops
# them at parse time; these pin the verdict-position rule against canned
# outputs, with the false-drop cases first-class citizens.


def _parse_one(detail: str, title: str = "T"):
    reply = (
        "```json\n"
        f'[{{"path": "a.py", "line": 3, "severity": "high",'
        f' "title": "{title}", "detail": "{detail}"}}]\n'
        "```"
    )
    return F.parse_findings(reply)


def test_retraction_drops_no_bug_verdict():
    # an earlier round shape: the body's own conclusion retracts the "finding".
    kept, dropped = F.drop_self_retracted(_parse_one("Checked the caller. No bug."))
    assert kept == [] and len(dropped) == 1


def test_retraction_drops_parenthesized_not_reporting():
    # an earlier round shape: "(Not reporting.)" as the final verdict.
    kept, dropped = F.drop_self_retracted(
        _parse_one("Cosmetic inconsistency at most. (Not reporting.)"))
    assert kept == [] and len(dropped) == 1


def test_retraction_drops_short_filler_after_verdict():
    # "no bug here." — a short residue is still a verdict.
    kept, dropped = F.drop_self_retracted(
        _parse_one("Re-examining: no bug here."))
    assert kept == [] and len(dropped) == 1


def test_retraction_drops_tests_pass_conclusion():
    # an earlier round shape: the detail's last word is "tests pass".
    kept, dropped = F.drop_self_retracted(
        _parse_one("Wrote a repro and checked the code, but tests pass"))
    assert kept == [] and len(dropped) == 1


def test_retraction_drops_retraction_in_title():
    # an earlier round shape: the title itself ends "— no bug".
    kept, dropped = F.drop_self_retracted(
        _parse_one("Body reports a real mechanism.", title="X divides correctly — no bug"))
    assert kept == [] and len(dropped) == 1


def test_retraction_keeps_tests_pass_with_a_real_point_after():
    # The false-drop guard: "tests pass" followed by the actual finding is
    # reasoning, not a verdict — the clause after the phrase is the point.
    kept, dropped = F.drop_self_retracted(
        _parse_one("The assertion is inverted; tests pass even when the handler is broken."))
    assert len(kept) == 1 and dropped == []


def test_retraction_keeps_no_bug_there_before_a_real_defect():
    # an earlier round, verbatim shape: a REAL finding whose body contains "No bug
    # there." mid-text, then "The actual defect: …". A whole-text grep
    # would drop this; the verdict-position rule must not.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The marker check is a substring match, so it still matches. "
        "No bug there. The actual defect: if the first post raises a 422 "
        "but the review was created, the fallback posts a second review "
        "with the same head marker."))
    assert len(kept) == 1 and dropped == []


def test_retraction_keeps_scoped_not_an_issue_then_a_real_bug():
    # A phrase opening the first segment, then the bug: kept.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Not an issue for the happy path, but the retry path corrupts state."))
    assert len(kept) == 1 and dropped == []


def test_retraction_all_entries_retracted_is_valid_no_bugs():
    reply = (
        "```json\n"
        '[{"path": "a.py", "line": 3, "severity": "high",'
        ' "title": "X", "detail": "Actually correct. No bug."},\n'
        '{"path": "b.py", "line": 4, "severity": "low",'
        ' "title": "Y", "detail": "Guarded upstream. Not reporting."}]\n'
        "```"
    )
    kept, dropped = F.drop_self_retracted(F.parse_findings(reply))
    assert kept == [] and len(dropped) == 2


def test_retraction_keeps_real_findings_untouched():
    reply = (
        "```json\n"
        '[{"path": "a.py", "line": 3, "severity": "high",'
        ' "title": "Off by one", "detail": "range excludes last element"},\n'
        '{"path": "b.py", "line": 9, "severity": "low",'
        ' "title": "Leak", "detail": "handle never closed on the error path"}]\n'
        "```"
    )
    kept, dropped = F.drop_self_retracted(F.parse_findings(reply))
    assert [f.path for f in kept] == ["a.py", "b.py"] and dropped == []


def test_retraction_keeps_a_bug_claim_despite_trailing_context():
    # "not a bug" mid-sentence with a long residue is reasoning: kept.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The wraparound is not a bug in isolation, but here the index "
        "feeds the length check, so the last element is dropped."))
    assert len(kept) == 1 and dropped == []


def test_retraction_drops_intended_behavior_conclusion():
    kept, dropped = F.drop_self_retracted(
        _parse_one("Checked the truncation cap; the behavior is intended."))
    assert kept == [] and len(dropped) == 1


# --------------------------------------------- resolved-thread history
# an earlier round pre-A/B: the first turn must know what was already raised AND
# resolved, or the reviewer re-derives the same shapes every round.


def _gh_thread(node_id, resolved, outdated, comments):
    return {
        "id": node_id,
        "isResolved": resolved,
        "isOutdated": outdated,
        # pageInfo as the live query fetches it (completeness signal).
        "comments": {"nodes": comments, "pageInfo": {"hasNextPage": False}},
    }


def _gh_comment(body, path="a.py", line=3, original_line=3, cid=100):
    return {
        "databaseId": cid,
        "author": {"login": "github-actions"},
        "body": body,
        "path": path,
        "line": line,
        "originalLine": original_line,
        "createdAt": "2026-10-15T10:00:00Z",
    }


def _patch_graphql(monkeypatch, nodes):
    from wrasse.platforms import github as fetch_mod

    def fake_gh_graphql(query, token, variables=None):
        return {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": nodes,
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }}}}}

    monkeypatch.setattr(fetch_mod, "gh_graphql", fake_gh_graphql)


def test_fetch_resolved_own_threads_carries_resolution(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc severity=high kind=bug sha=dead -->"
    _patch_graphql(monkeypatch, [
        _gh_thread("T1", resolved=True, outdated=False, comments=[
            _gh_comment(marker + "\n**[high] Leaks the handle**\nbody", cid=1),
            {"databaseId": 2, "author": {"login": "vmic2002"},
             "body": "Fixed in the retry path.", "path": "a.py", "line": 3,
             "originalLine": 3, "createdAt": "2026-10-15T11:00:00Z"},
        ]),
        _gh_thread("T2", resolved=False, outdated=False, comments=[
            _gh_comment(marker + "\n**[medium] Still open**\nbody", cid=3),
        ]),
    ])
    out = fetch_mod.fetch_resolved_own_threads("tok", "o/r", 1)
    assert len(out) == 1
    assert out[0]["thread_id"] == "T1"
    assert out[0]["title"].startswith("[high] Leaks the handle")
    assert out[0]["resolution"] == "Fixed in the retry path."


def test_fetch_resolved_own_threads_falls_back_to_original_line(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc -->"
    _patch_graphql(monkeypatch, [
        _gh_thread("T1", resolved=True, outdated=True, comments=[
            _gh_comment(marker + "\n**[low] Old position**", line=None,
                        original_line=42, cid=1),
        ]),
    ])
    out = fetch_mod.fetch_resolved_own_threads("tok", "o/r", 1)
    assert out[0]["line"] == 42  # names the position it was raised at


def test_fetch_open_own_threads_unchanged_line_semantics(monkeypatch):
    # Open threads keep the exact old record shape: line stays None for
    # outdated threads (no originalLine fallback) — the dedupe and
    # resolution consumers gate on that.
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc -->"
    _patch_graphql(monkeypatch, [
        _gh_thread("T1", resolved=True, outdated=False, comments=[
            _gh_comment(marker + "\n**[high] Resolved one**", cid=1),
        ]),
        _gh_thread("T2", resolved=False, outdated=True, comments=[
            _gh_comment(marker + "\n**[medium] Open outdated**",
                        line=None, original_line=42, cid=2),
        ]),
    ])
    out = fetch_mod.fetch_open_own_threads("tok", "o/r", 1)
    assert len(out) == 1 and out[0]["thread_id"] == "T2"
    assert out[0]["line"] is None and out[0]["original_line"] == 42
    assert "resolution" not in out[0]


def test_build_user_prompt_lists_resolved_history():
    prompt = F.build_user_prompt(
        "diff --git a/a.py b/a.py\n", "",
        already_raised=["- `a.py:3` — open one"],
        already_resolved=[
            "- `a.py:9` — pct() rank convention (resolved: by design, kept)",
            "- `b.py:12` — URL-construction misread (resolved: false positive)",
        ],
    )
    assert "## Findings previously raised on this PR and already resolved" in prompt
    assert "pct() rank convention" in prompt
    assert "(resolved: by design, kept)" in prompt
    assert "## Findings already raised on this PR" in prompt
    # Do-not-re-derive guidance rides along.
    assert "Do not re-derive or re-report" in prompt


def test_build_user_prompt_omits_resolved_block_when_empty():
    prompt = F.build_user_prompt("diff --git a/a.py b/a.py\n", "",
                                 already_raised=None, already_resolved=[])
    assert "already resolved" not in prompt


def test_split_already_raised_still_ignores_resolved_history():
    # The hard dedupe sees OPEN threads only — resolved history is prompt
    # context, never suppression, or a re-introduced bug at a resolved
    # line would become invisible.
    f = F.Finding(path="a.py", line=9, severity="high", title="Re-introduced")
    fresh, withheld = F.split_already_raised([f], open_threads=[
        {"path": "a.py", "line": 9, "outdated": False},
    ])
    assert withheld == [f] and fresh == []
    # With no OPEN thread at the position (only a resolved one, which is
    # never passed here), the finding posts.
    fresh, withheld = F.split_already_raised([f], open_threads=[])
    assert fresh == [f] and withheld == []


# ------------------------------------------------ gateway retry (an earlier round pre-A/B)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_CHAT_OK = b'{"choices": [{"message": {"content": "[]"}}], "usage": {}}'


def test_call_model_retries_once_on_network_failure(monkeypatch):
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            # "Remote end closed connection without response" (an earlier round) —
            # urllib wraps it as URLError(reason=RemoteDisconnected).
            raise urllib.error.URLError("Remote end closed connection without response")
        return _FakeResponse(_CHAT_OK)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    reply = GW.call_model(api_key="k", model="m", system="s", user="u")
    assert reply.content == "[]"
    assert len(calls) == 2  # one retry, not a loop
    # A fresh Request object each attempt — a killed connection can leave
    # a Request urllib will not accept twice.
    assert calls[0] is not calls[1]


def test_call_model_gives_up_after_second_network_failure(monkeypatch):
    import urllib.error

    from wrasse.providers import base as GW

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_model(api_key="k", model="m", system="s", user="u")
    assert ei.value.code == 0


def test_call_model_never_retries_non_retryable_http_errors(monkeypatch):
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b"boom"

    def fake_urlopen(req, timeout):
        calls.append(req)
        # 500: the gateway answered with a real verdict on the request —
        # re-sending is not ours to decide. 502/503 are the exception
        # (see _HTTP_RETRY_STATUSES); 429 has its own retry-after cadence.
        raise _HTTPError("url", 500, "Server Error", None, None)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_model(api_key="k", model="m", system="s", user="u")
    assert ei.value.code == 500
    assert len(calls) == 1  # the gateway answered; no retry


def test_call_responses_retries_once_on_gateway_502(monkeypatch):
    # The 2026-10-17 an earlier round shape: the gateway's owned-502 ("nowhere to
    # hop") on a mid-loop turn. A re-send is safe — the turn never ran —
    # and the same second of traffic served 200s, so the retry survives
    # the blip instead of red-statusing the head.
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error": {"code": "internal_error"}}'

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise _HTTPError("url", 502, "Bad Gateway", None, None)
        return _FakeResponse(b'{"id": "r", "output": [], "usage": {}, "status": "completed"}')

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    turn = GW.call_responses(api_key="k", model="m", input="hi")
    assert turn.id == "r" and turn.status == "completed"
    assert len(calls) == 2  # one retry, not a loop


def test_call_model_retries_once_on_gateway_503(monkeypatch):
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error": {"code": "internal_error"}}'

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise _HTTPError("url", 503, "Service Unavailable", None, None)
        return _FakeResponse(_CHAT_OK)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    reply = GW.call_model(api_key="k", model="m", system="s", user="u")
    assert reply.content == "[]"
    assert len(calls) == 2


def test_gateway_502_gives_up_after_second_502(monkeypatch):
    # ONE retry: a gateway that is down for real must not have its wall
    # clock doubled on every review.
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error": {"code": "internal_error"}}'

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise _HTTPError("url", 502, "Bad Gateway", None, None)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(api_key="k", model="m", input="hi")
    assert ei.value.code == 502
    assert len(calls) == 2


def test_gateway_504_is_never_retried(monkeypatch):
    # 504 is the timeout shape: the turn may have RUN upstream, so a
    # re-send could double-charge and duplicate output.
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error": {"code": "timeout"}}'

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise _HTTPError("url", 504, "Gateway Timeout", None, None)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(api_key="k", model="m", input="hi")
    assert ei.value.code == 504
    assert len(calls) == 1  # never retried


def test_gateway_upstream_stream_incomplete_502_is_never_retried(monkeypatch):
    # an earlier review round: provider.py raises this 502 AFTER the upstream stream
    # ran and ended without a terminal event — the generation already
    # happened, so a re-send repeats completed work and can double-charge.
    # The error code reaches the client un-scrubbed, so the harness can
    # and must exclude it from the 502/503 retry.
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return (b'{"error": {"message": "upstream stream ended without a '
                    b'terminal Responses event", "type": "api_error", '
                    b'"code": "upstream_stream_incomplete"}}')

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise _HTTPError("url", 502, "Bad Gateway", None, None)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(api_key="k", model="m", input="hi")
    assert ei.value.code == 502
    assert "upstream_stream_incomplete" in ei.value.body
    assert len(calls) == 1  # post-execution failure: never re-sent


def test_gateway_502_with_malformed_error_code_envelope_is_safe(monkeypatch):
    # an earlier review round: an error.code that is an array would make `in
    # frozenset` raise TypeError (unhashable) — review.py's handler
    # catches GatewayError, not TypeError, so the run would die without
    # posting its error status. Unknown envelope shapes read as
    # retryable transport junk: retried once, then a clean GatewayError.
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error": {"code": ["internal_error"]}}'

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise _HTTPError("url", 502, "Bad Gateway", None, None)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(api_key="k", model="m", input="hi")
    assert ei.value.code == 502  # GatewayError, not TypeError
    assert len(calls) == 2


def test_gateway_502_retry_skipped_when_no_deadline_budget(monkeypatch):
    # Near the wall deadline, failing now beats granting the remaining
    # allowance twice — the 502 retry clamps exactly like the network one.
    import time as time_mod

    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    class _HTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error": {"code": "internal_error"}}'

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise _HTTPError("url", 502, "Bad Gateway", None, None)

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(
            api_key="k", model="m", input="hi",
            deadline_s=time_mod.monotonic() + 3.0)  # < backoff + 1s floor
    assert "no retry budget" in str(ei.value)
    assert ei.value.code == 502
    assert len(calls) == 1


def test_call_responses_retries_once_on_network_failure(monkeypatch):
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.URLError("Remote end closed connection without response")
        return _FakeResponse(b'{"id": "resp_1", "output": [], "usage": {}, "status": "completed"}')

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    turn = GW.call_responses(api_key="k", model="m", input="hi")
    assert turn.id == "resp_1" and turn.status == "completed"
    assert len(calls) == 2


def test_build_summary_counts_dropped_retractions():
    # The summary takes a COUNT (an earlier review round caught the call site
    # passing the list itself, printing a raw Finding repr into the body).
    from wrasse.core import review as R

    class _Reply:
        elapsed_s = 1.0
        prompt_tokens = 1
        completion_tokens = 1

    body = R.build_summary("m", _Reply(), [], [], 0, 0, dropped_retracted=2)
    assert "2 finding(s) dropped" in body
    assert "Finding(" not in body


def test_retraction_keeps_scoped_no_bug_in_tests():
    # an earlier review round: "No bug in tests." scopes the claim to tests while
    # the finding reports a real production defect — the residue SAYS
    # something, so it is not verdict filler. Kept.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The handler crashes on empty input. No bug in tests."))
    assert len(kept) == 1 and dropped == []


def test_retraction_keeps_short_real_consequence_clauses():
    # an earlier review round: a retraction phrase followed by a short real
    # consequence is reasoning, not a verdict.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The flag is set; as intended, it crashes."))
    assert len(kept) == 1 and dropped == []
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The mock hides it; tests pass, prod breaks."))
    assert len(kept) == 1 and dropped == []


def test_retraction_drops_as_a_qualifier():
    # an earlier round shape, verbatim: "not reporting as a definite bug." — the
    # "as a <noun>" tail names what is not being reported: a verdict.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Cannot confirm without seeing fetch.py; not reporting as a definite bug."))
    assert kept == [] and len(dropped) == 1


def test_retraction_filler_whitelist_still_drops_here():
    kept, dropped = F.drop_self_retracted(_parse_one("Re-examining: no bug here."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one("Checked all paths; no bug there."))
    assert kept == [] and len(dropped) == 1


def test_prompt_allows_same_position_regression():
    # an earlier review round: a resolved bug reintroduced at its ORIGINAL line must
    # stay reportable — the prompt may not tell the model "new position
    # only". Since the hard filter (an earlier round, an earlier round), a same-position
    # report is listed in the summary rather than posted as a thread — the
    # prompt must say exactly that, so the contract the model sees matches
    # the enforcement.
    prompt = F.build_user_prompt("d", "", already_resolved=[
        "- `a.py:9` — old finding (resolved: fixed)"])
    assert "reintroduces a resolved bug, report it" in prompt
    assert "listed in the review summary" in prompt


def test_fetch_resolution_uses_true_last_comment(monkeypatch):
    # an earlier review round: a >10-comment thread's resolving reply lives outside
    # comments(first: 10); the aliased last: 1 connection must supply it.
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc -->"
    first = _gh_comment(marker + "\n**[high] Long thread**", cid=1)
    resolving = {
        "databaseId": 99, "author": {"login": "vmic2002"},
        "body": "Fixed in a later commit; comment 11 is the real end.",
        "path": "a.py", "line": 3, "originalLine": 3,
        "createdAt": "2026-10-15T12:00:00Z",
    }
    nodes = [{
        "id": "T1", "isResolved": True, "isOutdated": False,
        # the first-10 connection truncates: its last node is stale
        "comments": {"nodes": [first]},
        "lastComment": {"nodes": [resolving]},
    }]

    def fake_gh_graphql(query, token, variables=None):
        return {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": nodes,
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }}}}}

    monkeypatch.setattr(fetch_mod, "gh_graphql", fake_gh_graphql)
    out = fetch_mod.fetch_resolved_own_threads("tok", "o/r", 1)
    assert out[0]["resolution"].startswith("Fixed in a later commit")


def test_fetch_resolution_empty_when_thread_has_no_reply(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc -->"
    first = _gh_comment(marker + "\n**[high] Silent resolve**", cid=1)
    nodes = [{
        "id": "T1", "isResolved": True, "isOutdated": False,
        "comments": {"nodes": [first]},
        "lastComment": {"nodes": [first]},
    }]

    def fake_gh_graphql(query, token, variables=None):
        return {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": nodes,
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }}}}}

    monkeypatch.setattr(fetch_mod, "gh_graphql", fake_gh_graphql)
    out = fetch_mod.fetch_resolved_own_threads("tok", "o/r", 1)
    # Resolved with no reply: nothing to quote — echoing the finding's own
    # body as a "resolution" would be a fabrication.
    assert out[0]["resolution"] == ""


def test_latest_status_state_takes_newest_matching_context(monkeypatch):
    # an earlier review round: an OLDER success must not mask a NEWER error on the same
    # head — the repair paths repair on the latest status, not on whether
    # any historical success exists.
    from wrasse.platforms import github as fetch_mod

    statuses = [
        {"context": "Other Context", "state": "failure",
         "updated_at": "2026-10-17T23:30:00Z"},
        {"context": "Wrasse Review", "state": "success",
         "updated_at": "2026-10-17T20:00:00Z"},
        {"context": "Wrasse Review", "state": "error",
         "updated_at": "2026-10-17T22:00:00Z"},
    ]

    def fake_gh_get(path, token):
        assert "statuses?per_page=100" in path
        return json.dumps(statuses)

    monkeypatch.setattr(fetch_mod, "gh_get", fake_gh_get)
    assert fetch_mod.latest_status_state(
        "t", "o/r", "sha", "Wrasse Review") == "error"  # newest wins


def test_latest_status_state_none_when_context_absent(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    monkeypatch.setattr(
        fetch_mod, "gh_get",
        lambda path, token: json.dumps(
            [{"context": "Other", "state": "success",
              "updated_at": "2026-10-17T20:00:00Z"}]))
    assert fetch_mod.latest_status_state("t", "o/r", "sha", "Wrasse Review") is None


def test_gateway_timeout_is_never_retried(monkeypatch):
    # an earlier review round: socket.timeout had its full allowance; a retry
    # would double the wall clock of a hung gateway.
    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_model(api_key="k", model="m", system="s", user="u")
    assert ei.value.code == 0
    assert len(calls) == 1  # no retry on timeout


def test_gateway_retry_skipped_when_no_deadline_budget(monkeypatch):
    # an earlier review round: near the wall deadline, failing now beats granting
    # the remaining allowance twice.
    import time as time_mod

    import urllib.error

    from wrasse.providers import base as GW

    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise urllib.error.URLError("Remote end closed connection without response")

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(
            api_key="k", model="m", input="hi",
            deadline_s=time_mod.monotonic() + 3.0)  # < backoff + 1s floor
    assert "no retry budget" in str(ei.value)
    assert len(calls) == 1


def test_gateway_retry_clamps_to_deadline(monkeypatch):
    import time as time_mod

    import urllib.error

    from wrasse.providers import base as GW

    seen_timeouts = []

    def fake_urlopen(req, timeout):
        seen_timeouts.append(timeout)
        if len(seen_timeouts) == 1:
            raise urllib.error.URLError("Remote end closed connection without response")
        return _FakeResponse(b'{"id": "r", "output": [], "usage": {}, "status": "completed"}')

    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(GW.time, "sleep", lambda s: None)
    GW.call_responses(
        api_key="k", model="m", input="hi",
        timeout_s=300.0, deadline_s=time_mod.monotonic() + 30.0)
    assert len(seen_timeouts) == 2
    assert seen_timeouts[0] == 300.0
    assert seen_timeouts[1] < 30.0  # clamped to the remaining wall clock


def test_retraction_keeps_as_a_consequence_clauses():
    # an earlier review round: "as a <anything>" is too broad — these are real
    # findings whose "as a …" tail is the consequence, not a verdict.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The mock client stubs the handler, so tests pass as a side effect while prod crashes."))
    assert len(kept) == 1 and dropped == []
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Tests pass as a result of the mocked client."))
    assert len(kept) == 1 and dropped == []
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The retry is by design as a fallback but it double-charges the user."))
    assert len(kept) == 1 and dropped == []


def test_retraction_still_drops_explicit_as_a_bug_noun():
    # The an earlier round verbatim shape must survive the narrowing.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Cannot confirm without seeing fetch.py; not reporting as a definite bug."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Speculative only; not reporting as a real issue."))
    assert kept == [] and len(dropped) == 1


def test_fetch_resolution_falls_back_only_on_untruncated_comments(monkeypatch):
    # an earlier review round: with lastComment absent, the first-10 connection
    # may be a truncated prefix — quoting its [-1] fabricates a resolution.
    # Fall back only when the connection is provably complete (< 10 nodes).
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc -->"
    replies = [
        {"databaseId": 10 + i, "author": {"login": "vmic2002"},
         "body": f"reply {i}", "path": "a.py", "line": 3,
         "originalLine": 3, "createdAt": "2026-10-15T10:0%d:00Z" % i}
        for i in range(1, 4)
    ]
    # The connection's pageInfo, not the node count, decides completeness:
    # an exactly-10-node page may be the whole thread (an earlier review round).
    def _conn(nodes, has_next):
        return {"nodes": nodes, "pageInfo": {"hasNextPage": has_next}}

    nodes_complete = [{
        "id": "T1", "isResolved": True, "isOutdated": False,
        # lastComment missing from the response
        "comments": _conn([_gh_comment(marker + "\n**[high] F**", cid=1)] + replies, False),
    }]
    nodes_exactly_ten = [{
        "id": "T1", "isResolved": True, "isOutdated": False,
        "comments": _conn([_gh_comment(marker + "\n**[high] F**", cid=1)]
                          + replies + [dict(replies[-1], databaseId=99)] * 6, False),
    }]
    nodes_truncated = [{
        "id": "T1", "isResolved": True, "isOutdated": False,
        "comments": _conn([_gh_comment(marker + "\n**[high] F**", cid=1)]
                          + replies + [dict(replies[-1], databaseId=99)] * 6, True),
    }]

    def _fetch(nodes):
        def fake(query, token, variables=None):
            return {"data": {"repository": {"pullRequest": {"reviewThreads": {
                "nodes": nodes,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }}}}}
        monkeypatch.setattr(fetch_mod, "gh_graphql", fake)
        return fetch_mod.fetch_resolved_own_threads("tok", "o/r", 1)

    assert _fetch(nodes_complete)[0]["resolution"] == "reply 3"
    # 10 nodes + hasNextPage false: complete — the true last is quoted
    # (the node-count heuristic would have dropped this resolution).
    assert _fetch(nodes_exactly_ten)[0]["resolution"] == "reply 3"
    # 10 nodes + hasNextPage true: a truncated prefix — no quote, no fabrication.
    assert _fetch(nodes_truncated)[0]["resolution"] == ""


def test_gateway_retry_reclamps_to_post_sleep_clock(monkeypatch):
    # The retry's allowance is whatever is ACTUALLY left after the backoff
    # sleep — which can overshoot. Pre-subtraction alone bounded the
    # common case; this pins the airtight one.
    import urllib.error

    from wrasse.providers import base as GW

    class _Clock:
        def __init__(self):
            self.now = 1000.0

        def monotonic(self):
            return self.now

        def sleep(self, s):
            self.now += s + 12.0  # oversleeps well past the deadline

    clock = _Clock()
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(timeout)
        raise urllib.error.URLError("Remote end closed connection without response")

    monkeypatch.setattr(GW, "time", clock)
    monkeypatch.setattr(GW.urllib.request, "urlopen", fake_urlopen)
    import pytest
    with pytest.raises(GW.GatewayError) as ei:
        GW.call_responses(api_key="k", model="m", input="hi",
                          timeout_s=300.0, deadline_s=clock.now + 10.0)
    # pre-check passes (10 - 3 = 7 > 1), the oversleeping backoff eats the
    # rest, and no second request is made.
    assert len(calls) == 1
    assert "budget" in str(ei.value)


def test_retraction_drops_potential_and_possible_qualifiers():
    # an earlier review round (round on 832229089): "not reporting as a potential bug"
    # is an explicit withdrawal the adjective list must cover.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Cannot confirm without a repro; not reporting as a potential bug."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Speculative shape only; not reporting as a possible issue."))
    assert kept == [] and len(dropped) == 1


def test_retraction_keeps_weak_phrases_as_real_conclusions():
    # an earlier review round (round on 538c2d774): weak phrases end real findings too —
    # mid-sentence "is intended" and consequence "so tests pass" are the
    # finding's point, not a verdict.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "handle() returns a str where an int is intended."))
    assert len(kept) == 1 and dropped == []
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The early return skips the finally block, so the second retry never runs as intended."))
    assert len(kept) == 1 and dropped == []
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The mock swallows the exception, so tests pass."))
    assert len(kept) == 1 and dropped == []


def test_retraction_drops_weak_phrases_only_in_a_verdict_frame():
    # The same weak phrases ARE verdicts when the final segment opens with
    # them or they follow a contrast "but".
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Checked the truncation cap; the behavior is intended."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Minor coercion only; this is by design."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Re-examined the loop bounds; it works as intended."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Wrote a repro and checked the callers, but tests pass."))
    assert kept == [] and len(dropped) == 1


def test_fetch_pageinfo_absent_means_unknown_not_complete(monkeypatch):
    # an earlier review round (round on 24349e391): a comments connection with
    # no pageInfo at all must not read as complete.
    from wrasse.platforms import github as fetch_mod

    marker = "<!-- wrasse id=abc -->"
    replies = [
        {"databaseId": 10 + i, "author": {"login": "vmic2002"},
         "body": f"reply {i}", "path": "a.py", "line": 3,
         "originalLine": 3, "createdAt": "2026-10-15T10:00:00Z"}
        for i in range(1, 4)
    ]
    nodes = [{
        "id": "T1", "isResolved": True, "isOutdated": False,
        # no lastComment AND no pageInfo: completeness unknown
        "comments": {"nodes": [_gh_comment(marker + "\n**[high] F**", cid=1)] + replies},
    }]

    def fake(query, token, variables=None):
        return {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": nodes,
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }}}}}

    monkeypatch.setattr(fetch_mod, "gh_graphql", fake)
    out = fetch_mod.fetch_resolved_own_threads("tok", "o/r", 1)
    assert out[0]["resolution"] == ""  # unknown completeness: no quote


def test_retraction_drops_copula_working_as_intended_verdicts():
    # an earlier review round (round on ed6fef23b): "this is working as intended" and
    # "this is as intended" are verdicts the frame must catch.
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Checked the retry path; this is working as intended."))
    assert kept == [] and len(dropped) == 1
    kept, dropped = F.drop_self_retracted(_parse_one(
        "Checked the callers; this is as intended."))
    assert kept == [] and len(dropped) == 1
    # The prior kept-cases stay kept (mid-sentence / consequence forms).
    kept, dropped = F.drop_self_retracted(_parse_one(
        "The early return skips the finally block, so the second retry never runs as intended."))
    assert len(kept) == 1 and dropped == []


def test_format_thread_position_lists_both_anchors():
    # an earlier review round: the dedupe withholds on the raise-time span
    # AND the current line, so the prompt must show both — otherwise the
    # model is instructed to report at the moved-to line and the finding
    # is silently withheld.
    moved = {"path": "a.py", "line": 80, "original_line": 14,
             "marker_start": 9, "marker_end": 14}
    assert F.format_thread_position(moved) == "a.py:9-14 (now at a.py:80)"
    unmoved = {"path": "a.py", "line": 10, "original_line": 10,
               "marker_start": None, "marker_end": None}
    assert F.format_thread_position(unmoved) == "a.py:10"
    span = {"path": "a.py", "line": 9, "original_line": 9,
            "marker_start": 9, "marker_end": 14}
    assert F.format_thread_position(span) == "a.py:9-14"
    legacy = {"path": "a.py", "line": 3, "original_line": 3}
    assert F.format_thread_position(legacy) == "a.py:3"


def test_thread_spans_current_range_covers_moved_interior():
    # an earlier review round BUG-1: after a range comment moves, GitHub's live
    # fields carry the whole CURRENT range (startLine..line) — anchoring
    # only the shifted end left the moved interior uncovered, so an
    # interior re-flag posted a duplicate thread.
    t = {"line": 84, "start_line": 80, "original_line": 14,
         "original_start_line": 9, "marker_start": 9, "marker_end": 14}
    assert F.thread_spans(t) == [(9, 14), (80, 84)]
    # single-line thread: current anchor stays a point
    assert F.thread_spans({"line": 7}) == [(7, 7)]


def test_parse_findings_huge_int_end_line_does_not_crash():
    # an earlier review round BUG-2: math.isfinite converts its argument to float,
    # so a huge INTEGER end_line (arbitrary precision in Python) raised
    # OverflowError inside the guard itself. Ints are always finite —
    # the isfinite check applies to floats only.
    huge = ('```json\n[{"path": "a.py", "line": 5, "end_line": '
            '100000000000000000000000000000000000000000000, '
            '"severity": "high", "title": "T", "detail": "D"}]\n```')
    out = F.parse_findings(huge)
    assert out and out[0].end_line is not None  # huge but accepted
    # and the non-finite FLOAT still collapses to single-line
    inf = ('```json\n[{"path": "a.py", "line": 5, "end_line": 1e999, '
           '"severity": "high", "title": "T", "detail": "D"}]\n```')
    assert F.parse_findings(inf)[0].end_line is None


def test_fetch_own_summaries_paginates_and_flags_incomplete(monkeypatch):
    # an earlier review round: the reviews connection was a single first:100 page,
    # so later summaries on long-lived PRs were never minimized. Now the
    # walk follows the cursor and reports completeness at the hard stop.
    from wrasse.platforms import github as fetch_mod

    pages = [
        {"data": {"repository": {"pullRequest": {"reviews": {
            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
            "nodes": [{"id": "s1", "author": {"login": "github-actions"},
                       "isMinimized": False,
                       "body": "<!-- wrasse head=aaa v=2.6.1 -->"}]}}}}},
        {"data": {"repository": {"pullRequest": {"reviews": {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [{"id": "s2", "author": {"login": "github-actions"},
                       "isMinimized": True, "body": "placeholder"}]}}}}},
    ]
    calls = []

    def fake_gql(q, token, variables=None):
        calls.append(variables.get("cursor"))
        return pages[len(calls) - 1]

    monkeypatch.setattr(fetch_mod, "gh_graphql", fake_gql)
    recs, complete = fetch_mod.fetch_own_summaries("tok", "r", 1)
    assert [r["node_id"] for r in recs] == ["s1", "s2"]
    assert recs[0]["sha"] == "aaa" and recs[1]["sha"] == ""
    assert complete is True
    assert calls == [None, "c1"]


def test_fetch_own_summaries_hard_stop_is_incomplete(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    page = {"data": {"repository": {"pullRequest": {"reviews": {
        "pageInfo": {"hasNextPage": True, "endCursor": "c"},
        "nodes": []}}}}}

    monkeypatch.setattr(fetch_mod, "gh_graphql", lambda q, t, v=None: page)
    _, complete = fetch_mod.fetch_own_summaries("tok", "r", 1)
    assert complete is False


def test_thread_fetch_reports_incomplete_at_cap(monkeypatch):
    # an earlier review round: >200 threads previously read as "all resolved"
    # to the minimize pass; now the walk reports incompleteness.
    from wrasse.platforms import github as fetch_mod

    page = {"data": {"repository": {"pullRequest": {"reviewThreads": {
        "pageInfo": {"hasNextPage": True, "endCursor": "c"},
        "nodes": []}}}}}

    monkeypatch.setattr(fetch_mod, "gh_graphql", lambda q, t, v=None: page)
    _, complete = fetch_mod._fetch_own_threads("tok", "r", 1, resolved=False)
    assert complete is False


def test_minimize_pass_refuses_partial_state(monkeypatch, capsys):
    # The caller-level guard: an incomplete fetch must skip the pass
    # entirely — one redundant retry beats a minimized live round.
    from wrasse.core import review as review_mod

    monkeypatch.setattr(review_mod.fetch, "fetch_own_summaries",
                        lambda *a, **k: ([{"node_id": "s", "sha": "aaa",
                                           "minimized": False}], False))
    out = review_mod.minimize_resolved_rounds("tok", "r", 1)
    assert out == 0
    assert "pagination cap" in capsys.readouterr().err
def test_head_already_reviewed_agent_mode_ignores_v1(monkeypatch):
    # an earlier review round: after the v2 flip, a head only v1-reviewed must get its
    # v2 review — the guard matches the summary's MODE word in agent mode,
    # never the version digits.
    from wrasse.platforms import github as fetch_mod

    def fake_gh_get(path, token, accept="application/vnd.github+json"):
        return json.dumps([
            {"user": {"login": "github-actions[bot]"}, "commit_id": "0123456789",
             "body": "**Wrasse Review** — one-turn review (v2.6.1) via `deepseek-v4.1-flash-fast`."
                     "\n\n<!-- wrasse head=0123456789 -->"},
            {"user": {"login": "github-actions[bot]"}, "commit_id": "0123456789",
             "body": "**Wrasse Review** — agentic review (v2.6.1) via `glm-5.3`."
                     "\n\n<!-- wrasse head=0123456789 v=2.6.1 -->"},
        ])

    monkeypatch.setattr(fetch_mod, "gh_get", fake_gh_get)
    # v1 mode: the first review already counts (legacy marker shape).
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "0123456789abcdef") is True
    # agent mode: only an agentic review on the head counts.
    def v1_only(path, token, accept="application/vnd.github+json"):
        return json.dumps([
            {"user": {"login": "github-actions[bot]"}, "commit_id": "0123456789",
             "body": "**Wrasse Review** — one-turn review (v2.6.1) via `deepseek-v4.1-flash-fast`."
                     "\n\n<!-- wrasse head=0123456789 v=2.6.1 -->"},
        ])
    monkeypatch.setattr(fetch_mod, "gh_get", v1_only)
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "0123456789abcdef", require_agent=True) is False
    monkeypatch.setattr(fetch_mod, "gh_get", fake_gh_get)
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "0123456789abcdef", require_agent=True) is True

    # an earlier review round: a one-turn review whose FINDINGS prose mentions
    # "agentic review" must not count as an agentic review of the head.
    def prose_trap(path, token, accept="application/vnd.github+json"):
        return json.dumps([
            {"user": {"login": "github-actions[bot]"}, "commit_id": "0123456789",
             "body": "**Wrasse Review** — one-turn review (v2.6.1) via `deepseek-v4.1-flash-fast`."
                     "\n\n- `src/wrasse/core/agent.py:12` — the string 'agentic review (v2)'"
                     " appears in the agent prompt\n\n<!-- wrasse head=0123456789 v=2.6.1 -->"},
        ])
    monkeypatch.setattr(fetch_mod, "gh_get", prose_trap)
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "0123456789abcdef", require_agent=True) is False
    assert fetch_mod.head_already_reviewed(
        "tok", "r", 1, "0123456789abcdef") is True


def test_review_version_advertised():
    # an earlier round shadow-week attribution: every summary names the harness
    # version, and the head marker carries it.
    from wrasse.platforms import publish as post_mod
    from wrasse.core import review as review_mod

    class _R:
        elapsed_s = 1.0
        prompt_tokens = 10
        completion_tokens = 5

    agent_head = review_mod.build_summary(
        "glm-5.3", _R(), [], [], 0, 0, agent=True).splitlines()[0]
    assert f"agentic review (v{post_mod.REVIEW_VERSION})" in agent_head
    oneturn_head = review_mod.build_summary(
        "glm-5.3", _R(), [], [], 0, 0, agent=False).splitlines()[0]
    assert f"one-turn review (v{post_mod.REVIEW_VERSION})" in oneturn_head


def test_summaries_to_minimize_never_minimizes_agent_summaries():
    # First live v2 review (an earlier round): an agentic summary is the head's
    # agent-mode guard record — minimizing it kills the mode word and
    # re-arms an unbounded re-review loop under require_agent.
    agentic = {"node_id": "s1", "sha": "aaa", "minimized": False,
               "agent": True}
    oneturn = {"node_id": "s2", "sha": "aaa", "minimized": False,
               "agent": False}
    out = fetch.summaries_to_minimize(
        [agentic, oneturn], [], [{"sha": "aaa"}])
    assert [s["node_id"] for s in out] == ["s2"]
    # legacy records without the flag behave as one-turn (minimize)
    legacy = {"node_id": "s3", "sha": "aaa", "minimized": False}
    out = fetch.summaries_to_minimize([legacy], [], [{"sha": "aaa"}])
    assert [s["node_id"] for s in out] == ["s3"]
