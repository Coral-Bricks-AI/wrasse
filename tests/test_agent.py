"""Unit tests for the v2 agentic loop (an earlier round): turn sequencing, budget
caps, tool dispatch, and the grep/read bounds.

Canned /v1/responses payloads only — the loop is what these tests pin
down, not the model (same principle as the v1 parser tests). End-to-end
runs happen via --dry-run against the real gateway.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from wrasse.core import agent as A
from wrasse.platforms import github as fetch_mod
from wrasse.providers import base as gateway_mod

FINDINGS_JSON = (
    "```json\n"
    '[{"path": "a.py", "line": 3, "severity": "high", "title": "T", "detail": "D"}]\n'
    "```\n"
)


def _fc(call_id, name, args):
    return {"type": "function_call", "id": f"item_{call_id}", "call_id": call_id,
            "name": name, "arguments": args}


def _msg(text):
    return {"type": "message",
            "content": [{"type": "output_text", "text": text}]}


class _Transport:
    """Replays scripted output lists in order; records every call. A script
    entry is either an output list (status "completed") or an
    (output list, status) tuple — the latter for incomplete-turn cases."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def __call__(self, *, api_key, model, input, base_url, session_id,
                 instructions, tools, previous_response_id, timeout_s,
                 deadline_s=None):
        self.calls.append({
            "input": input,
            "instructions": instructions,
            "tools": tools,
            "previous_response_id": previous_response_id,
            "timeout_s": timeout_s,
            "deadline_s": deadline_s,
        })
        entry = self.script[min(len(self.calls), len(self.script)) - 1]
        out, status = entry if isinstance(entry, tuple) else (entry, "completed")
        return gateway_mod.ResponsesTurn(
            id=f"resp_{len(self.calls)}", output=out, elapsed_s=0.01,
            input_tokens=10, output_tokens=5, status=status)


class _Tools:
    """Records dispatches; returns a canned string per call."""

    def __init__(self, answer="file contents here"):
        self.answer = answer
        self.dispatched = []

    def dispatch(self, name, args_json):
        self.dispatched.append((name, args_json))
        return self.answer


def _run(transport, tools=None, **kw):
    return A.run_agent_review(
        api_key="k", model="m", base_url="http://gw", session_id="s",
        repo="o/r", head_sha="deadbeef", token="tok",
        diff="diff --git a/a.py b/a.py\n", context="", already_raised=[],
        transport=transport, tools_exec=tools or _Tools(), **kw)


# ---------------------------------------------------------------- loop


def test_tool_turn_then_final():
    t = _Transport([[_fc("c1", "read_file", '{"path": "a.py"}')], [_msg(FINDINGS_JSON)]])
    tools = _Tools()
    reply = _run(t, tools=tools)
    assert reply.content == FINDINGS_JSON.strip("\n") or "T" in reply.content
    assert reply.turns == 2 and reply.tool_calls == 1
    assert not reply.budget_exhausted
    # Turn 1: full prompt as a string, tools present, no previous id.
    assert isinstance(t.calls[0]["input"], str)
    assert "diff" in t.calls[0]["input"]
    assert t.calls[0]["tools"] == A.TOOLS
    assert t.calls[0]["previous_response_id"] is None
    assert "json" in t.calls[0]["instructions"]  # findings contract attached
    # Turn 2: chained, output answers the call_id, tools still offered.
    assert t.calls[1]["previous_response_id"] == "resp_1"
    assert isinstance(t.calls[1]["input"], list)
    out = t.calls[1]["input"][0]
    assert out["type"] == "function_call_output" and out["call_id"] == "c1"
    assert out["output"] == "file contents here"
    assert tools.dispatched == [("read_file", '{"path": "a.py"}')]


def test_instructions_sent_every_turn():
    # Not inherited across a previous_response_id chain (gateway contract) —
    # every turn must carry the findings contract or the chain drops it.
    t = _Transport([[_fc("c1", "grep", '{"pattern": "x"}')], [_msg(FINDINGS_JSON)]])
    _run(t)
    assert len(t.calls) == 2
    assert all(c["instructions"] for c in t.calls)
    assert A.AGENT_ADDENDUM in t.calls[0]["instructions"]


def test_pr_intent_reaches_the_first_turn():
    """The PR's stated intent rides in turn 1's input (an earlier round): the
    agent must see the same purpose-blind diff guard the one-turn call does."""
    t = _Transport([[_msg(FINDINGS_JSON)]])
    _run(t, pr_intent={"title": "Fix trial amounts", "body": "id-asc order intended"})
    assert "The author's stated intent" in t.calls[0]["input"]
    assert "id-asc order intended" in t.calls[0]["input"]


def test_tool_counts_ride_on_the_reply():
    """Per-tool counts on the reply (an earlier round benchmark rows): the row
    must show WHICH tools a review used — "is it exploring smartly" is the
    first question every subset analysis asks, and aggregate counts force
    manual gateway-log inference."""
    t = _Transport([
        [_fc("c1", "read_file", '{"path": "a.py"}'),
         _fc("c2", "grep", '{"pattern": "x"}')],
        [_fc("c3", "grep", '{"pattern": "y"}')],
        [_msg(FINDINGS_JSON)],
    ])
    reply = _run(t)
    assert reply.tool_counts == {"read_file": 1, "grep": 2}


def test_tool_counts_survive_a_forced_final():
    """Counts accumulated before exhaustion ride out on the cut-short reply
    too — an exhausted review's tool profile is exactly the profile the
    benchmark wants to inspect."""
    t = _Transport([[_fc("c1", "file_history", '{"path": "a.py"}')],
                    [_msg(FINDINGS_JSON)]])
    reply = _run(t, max_tool_turns=1)
    assert reply.budget_exhausted
    assert reply.tool_counts == {"file_history": 1}


def test_first_turn_can_answer_directly():
    t = _Transport([[_msg(FINDINGS_JSON)]])
    reply = _run(t)
    assert reply.turns == 1 and reply.tool_calls == 0


def test_parallel_calls_all_answered_in_one_turn():
    t = _Transport([
        [_fc("c1", "read_file", '{"path": "a.py"}'),
         _fc("c2", "list_dir", '{"path": ""}')],
        [_msg(FINDINGS_JSON)],
    ])
    tools = _Tools()
    _run(t, tools=tools)
    assert len(tools.dispatched) == 2
    outs = [o for o in t.calls[1]["input"] if o.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outs] == ["c1", "c2"]


def test_budget_note_rides_with_every_tool_output_batch():
    """The model must see its remaining turns on every tool turn (BUDGET_NOTE,
    an earlier round): without it, flash-tier explores until the cap and the forced
    final loses findings — measured 3/3 heads self-terminated with the note
    where without it 100% exhausted the cap."""
    t = _Transport([
        [_fc("c1", "grep", '{"pattern": "x"}')],
        [_fc("c2", "grep", '{"pattern": "y"}')],
        [_msg(FINDINGS_JSON)],
    ])
    _run(t, max_tool_turns=10)
    second = t.calls[1]["input"][-1]
    third = t.calls[2]["input"][-1]
    assert second == {"role": "user", "content": A.BUDGET_NOTE.format(
        remaining=9, max_tool_turns=10)}
    assert third == {"role": "user", "content": A.BUDGET_NOTE.format(
        remaining=8, max_tool_turns=10)}


def test_budget_note_says_zero_remaining_on_the_cap_turn():
    """max_tool_turns=1: the single tool turn's note reads 0 of 1 — the count
    is turns taken minus cap, so the forced final that follows never lies to
    the model about turns it does not have."""
    t = _Transport([[_fc("c1", "grep", '{"pattern": "x"}')], [_msg(FINDINGS_JSON)]])
    _run(t, max_tool_turns=1)
    note = [i for i in t.calls[1]["input"] if i.get("role") == "user"][0]
    assert "0 of 1 tool turn(s) remaining" in note["content"]


def test_turn_cap_forces_final_without_tools():
    # Script always calls tools; with max_tool_turns=1 the second call is
    # the forced final: no tools offered, budget flag set — and the pending
    # tool-call outputs are DELIVERED (an orphaned function_call can make
    # the chained request fail outright; an earlier review round).
    t = _Transport([[_fc("c1", "grep", '{"pattern": "x"}')], [_msg(FINDINGS_JSON)]])
    reply = _run(t, max_tool_turns=1)
    assert reply.budget_exhausted and reply.turns == 2
    assert t.calls[1]["tools"] is None
    assert t.calls[1]["previous_response_id"] == "resp_1"
    inp = t.calls[1]["input"]
    assert isinstance(inp, list) and inp[0]["type"] == "function_call_output"
    assert inp[-1] == {"role": "user", "content": A.FORCED_FINAL_INPUT}


def test_wall_clock_never_skips_the_diff():
    # A near-zero wall budget must degrade to a one-turn review — the model
    # still SEES the diff once; a blind forced-final verdict is worse than
    # a slow first turn (an earlier review round).
    t = _Transport([[_fc("c1", "grep", '{"pattern": "x"}')], [_msg(FINDINGS_JSON)]])
    reply = _run(t, wall_timeout_s=1e-9)
    assert isinstance(t.calls[0]["input"], str) and "diff" in t.calls[0]["input"]
    assert t.calls[0]["tools"] == A.TOOLS
    assert reply.budget_exhausted and reply.turns == 2


def test_zero_turns_budget_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        _run(_Transport([[]]), max_tool_turns=0)
    with pytest.raises(ValueError):
        _run(_Transport([[]]), wall_timeout_s=0)


def test_model_turn_timeout_clamped_to_deadline():
    # First turn: the loop admits it regardless of the wall clock (the
    # model must see the diff), so it gets the FULL per-turn allowance —
    # clamping it to leftover milliseconds would kill the request the
    # loop just allowed (an earlier review round). The forced final takes the fixed
    # grace window.
    t = _Transport([[_fc("c1", "grep", '{"pattern": "x"}')], [_msg(FINDINGS_JSON)]])
    _run(t, wall_timeout_s=1e-9, per_turn_timeout_s=300.0)
    assert t.calls[0]["timeout_s"] == 300.0
    assert t.calls[1]["timeout_s"] == A.FORCED_FINAL_GRACE_S


def test_later_turns_timeout_clamped_to_remaining_wall(monkeypatch):
    # From turn 2 on, an in-budget turn gets at most the wall time it has
    # LEFT — the deadline bounds the review, and a turn that cannot fit
    # fails visibly instead of silently overshooting (an earlier review round).
    class _Clock:
        t = 1000.0

        @staticmethod
        def monotonic():
            _Clock.t += 100.0
            return _Clock.t

    monkeypatch.setattr(A, "time", _Clock)
    t = _Transport([
        [_fc("c1", "grep", '{"pattern": "x"}')],
        [_fc("c2", "grep", '{"pattern": "y"}')],
        [_msg(FINDINGS_JSON)],
    ])
    reply = _run(t, max_tool_turns=10, wall_timeout_s=250.0, per_turn_timeout_s=300.0)
    assert t.calls[0]["timeout_s"] == 300.0   # first turn: full allowance
    assert t.calls[1]["timeout_s"] == 50.0    # 200s spent, 50 of the 250 left
    assert t.calls[2]["timeout_s"] == A.FORCED_FINAL_GRACE_S  # past deadline
    assert reply.budget_exhausted and reply.turns == 3


def test_forced_final_with_no_text_is_inconclusive():
    # A model that answers the forced final with another tool call (or
    # nothing) must be a harness error, never "no bugs".
    t = _Transport([[_fc("c1", "grep", '{"pattern": "x"}')], []])
    try:
        _run(t, max_tool_turns=1)
    except RuntimeError as e:
        assert "budget" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_turn_with_neither_calls_nor_text_raises():
    t = _Transport([[{"type": "reasoning", "summary": []}]])
    try:
        _run(t)
    except RuntimeError as e:
        assert "neither tool calls nor text" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_unknown_tool_gets_error_string_loop_continues(monkeypatch):
    # The REAL ToolBox answers an unknown tool with an error string, and
    # the loop carries on: the model sees the error and can adjust.
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: None)
    monkeypatch.setattr(fetch_mod, "fetch_dir_listing",
                        lambda repo, ref, path, token: None)
    t = _Transport([[_fc("c1", "shell", '{"cmd": "rm -rf"}')], [_msg(FINDINGS_JSON)]])
    reply = _run(t, tools=box)
    assert "unknown tool" in t.calls[1]["input"][0]["output"]
    assert reply.turns == 2


def test_real_dispatch_unknown_tool():
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a b/a\n")
    assert "unknown tool" in box.dispatch("shell", "{}")


def test_real_dispatch_bad_json_arguments():
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a b/a\n")
    assert "not valid JSON" in box.dispatch("read_file", "{path: a.py}")


def test_local_mode_tools_unavailable():
    # --diff-file without --head-sha: tools answer per-call, review lives on.
    box = A.ToolBox("o/r", "local", "tok", "diff --git a/a b/a\n")
    for name, args in (("read_file", '{"path": "a.py"}'),
                       ("list_dir", '{"path": ""}'),
                       ("grep", '{"pattern": "x"}')):
        assert "unavailable" in box.dispatch(name, args)


# ---------------------------------------------------------------- ToolBox vs GitHub


def _box(monkeypatch, files: dict):
    box = A.ToolBox(
        "o/r", "deadbeef", "tok",
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n",
    )
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: files.get(path))
    monkeypatch.setattr(fetch_mod, "fetch_dir_listing",
                        lambda repo, ref, path, token: None)
    return box


def test_read_file_truncates_and_names_the_next_offset(monkeypatch):
    box = _box(monkeypatch, {"a.py": "l\n" * (A.READ_MAX_CHARS // 2 + 500)})
    out = box.read_file("a.py")
    assert len(out) < A.READ_MAX_CHARS + 2000
    assert "[truncated" in out
    # The truncation note is the recovery instruction (an earlier round): a big
    # file is no longer a dead end — the note names the next line.
    assert "offset=" in out and "read_file again with offset" in out
    import re as _re
    m = _re.search(r"offset=(\d+)", out)
    assert m, out[-120:]
    # Following the note reads the continuation.
    rest = box.read_file("a.py", offset=int(m[1]))
    assert rest.startswith("l\n") or rest.startswith("l")
    assert "[truncated" not in rest  # the tail fits in the window


def test_read_file_offset_slicing_and_bounds(monkeypatch):
    box = _box(monkeypatch, {"a.py": "\n".join(f"line{i}" for i in range(1, 12))})
    assert box.read_file("a.py", offset=8) == "line8\nline9\nline10\nline11"
    assert "past the end" in box.read_file("a.py", offset=99)
    assert "positive 1-based" in box.read_file("a.py", offset=0)
    assert "positive 1-based" in box.read_file("a.py", offset="8")


def test_file_history_formats_commits_newest_first(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    def fake_history(repo, path, token, at_ref, per_page=10):
        assert per_page == A.FILE_HISTORY_MAX
        return [
            {"sha": "bbbbbbbbbbbbbbbbbbbbbbbb", "date": "2026-10-15T10:00:00Z",
             "message": "second commit\n\nbody"},
            {"sha": "aaaaaaaaaaaaaaaaaaaaaaaa", "date": "2026-10-01T10:00:00Z",
             "message": "first commit"},
        ]

    box = _box(monkeypatch, {})
    monkeypatch.setattr(fetch_mod, "fetch_path_history", fake_history)
    out = box.file_history("src/x.py")
    assert out == ("bbbbbbbbbb 2026-10-15 second commit\n"
                   "aaaaaaaaaa 2026-10-01 first commit")


def test_file_history_empty_and_errors_answer_the_turn(monkeypatch):
    from wrasse.platforms import github as fetch_mod
    box = _box(monkeypatch, {})
    monkeypatch.setattr(fetch_mod, "fetch_path_history",
                        lambda repo, path, token, at_ref, per_page=10: [])
    assert "no commits found" in box.file_history("src/x.py")

    def boom(repo, path, token, at_ref, per_page=10):
        raise fetch_mod.GitHubError(403, "/commits", "rate limited")

    monkeypatch.setattr(fetch_mod, "fetch_path_history", boom)
    assert box.file_history("src/x.py").startswith("error: GitHub API failure")
    assert "needs a non-empty string path" in box.file_history("")


def test_offset_reads_are_distinct_calls_for_the_duplicate_guard(monkeypatch):
    """read_file(path) and read_file(path, offset=N) are different keys —
    the continuation read must execute, not get pointer-answered. A
    many-line file: truncation lands mid-file with a next line to name."""
    box = _box(monkeypatch, {"a.py": "l\n" * (A.READ_MAX_CHARS // 2 + 100)})
    first = box.dispatch("read_file", '{"path": "a.py"}')
    assert "offset=" in first
    import re as _re
    nxt = int(_re.search(r"offset=(\d+)", first)[1])
    second = box.dispatch("read_file", f'{{"path": "a.py", "offset": {nxt}}}')
    assert "already answered" not in second


def test_truncation_landing_on_a_newline_skips_no_line(monkeypatch):
    """an earlier review round: a cut landing exactly ON a newline used to count the
    just-terminated line as consumed and advertise an offset one line
    past the next unread line — following the note silently dropped a
    line of source."""
    monkeypatch.setattr(A, "READ_MAX_CHARS", 10)
    box = _box(monkeypatch, {"a.py": "aaaa\nbbbb\ncccc\ndddd\n"})
    out = box.read_file("a.py")
    assert out.startswith("aaaa\nbbbb\n")   # cut exactly after line 2's \n
    import re as _re
    nxt = int(_re.search(r"offset=(\d+)", out)[1])
    assert nxt == 3                        # line 3 is next, not line 4
    assert box.read_file("a.py", offset=nxt).startswith("cccc")


def test_truncation_inside_a_final_long_line_admits_the_dead_end(monkeypatch):
    """an earlier review round / Wrasse an earlier round: a single line longer than the window has no
    next line to resume at — the note must not advertise an offset that
    errors; it says the remainder is unreachable."""
    monkeypatch.setattr(A, "READ_MAX_CHARS", 10)
    box = _box(monkeypatch, {"a.py": "xxxxxxxxxxxxxxxxxxx"})
    out = box.read_file("a.py")
    assert "cannot be read with offsets" in out
    assert "offset=" not in out


def test_truncation_mid_line_discloses_the_skipped_tail(monkeypatch):
    monkeypatch.setattr(A, "READ_MAX_CHARS", 12)
    box = _box(monkeypatch, {"a.py": "aaaa\nbbbb\ncccc\ndddd\n"})
    out = box.read_file("a.py")            # cut mid-line inside line 3
    assert "inside line 3" in out and "is skipped" in out
    assert box.read_file("a.py", offset=4).startswith("dddd")


def test_file_history_pins_the_reviewed_sha_and_quotes_the_path(monkeypatch):
    """an earlier review round: unpinned history walks the default branch — for a
    replayed head that is a different revision than the one under review.
    Wrasse an earlier round: repo paths contain spaces ('ml/SWE-Bench Verified/…'),
    so the path must be urlencoded, not interpolated."""
    import urllib.parse
    from wrasse.platforms import github as fetch_mod
    seen = []

    def fake_gh_get(path, token, accept=None):
        seen.append(path)
        return json.dumps([
            {"sha": "a" * 40,
             "commit": {"author": {"date": "2026-10-15T10:00:00Z"},
                        "message": "touched it"}},
        ])

    monkeypatch.setattr(fetch_mod, "gh_get", fake_gh_get)
    box = _box(monkeypatch, {})
    out = box.file_history("ml/SWE-Bench Verified/predictions.jsonl")
    assert "aaaaaaaaaa 2026-10-15 touched it" in out
    query = seen[0].split("?", 1)[1]
    params = dict(urllib.parse.parse_qsl(query))
    assert params["sha"] == "deadbeef"
    assert params["path"] == "ml/SWE-Bench Verified/predictions.jsonl"
    assert " " not in query  # the space was encoded, not interpolated raw


def test_read_file_missing(monkeypatch):
    box = _box(monkeypatch, {})
    assert "not found" in box.read_file("nope.py")


def test_grep_default_scope_is_diff_files(monkeypatch):
    box = _box(monkeypatch, {"a.py": "one\ntwo bug here\nthree\n"})
    out = box.grep("bug")
    assert "a.py:2: two bug here" in out


def test_grep_no_matches(monkeypatch):
    box = _box(monkeypatch, {"a.py": "one\ntwo\n"})
    assert "no matches" in box.grep("zzz")


def test_grep_invalid_regex(monkeypatch):
    box = _box(monkeypatch, {"a.py": "x"})
    assert "invalid regular expression" in box.grep("*broken(")


def _diff_for(*paths):
    return "".join(
        f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-old\n+new\n"
        for p in paths
    )


def _box_paths(monkeypatch, files, paths):
    box = A.ToolBox("o/r", "deadbeef", "tok", _diff_for(*paths))
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: files.get(path))
    monkeypatch.setattr(fetch_mod, "fetch_dir_listing",
                        lambda repo, ref, path, token: None)
    return box


def test_grep_skips_oversize_files_and_continues(monkeypatch):
    # One file over the byte cap must be SKIPPED, not abort the whole scan
    # (an earlier review round) — and disclosed once as an aggregate note, not one
    # note per file, with the "scanned" count still exact (r3).
    box = _box_paths(monkeypatch, {
        "huge.py": "x" * (A.GREP_MAX_BYTES + 10),
        "small.py": "a bug here\n",
    }, ["huge.py", "small.py"])
    out = box.grep("bug")
    assert "small.py:1: a bug here" in out
    assert "1 file(s) fetched but skipped — byte cap" in out
    assert "searched 1 file(s)" in out


def test_risky_pattern_end_of_pattern_group_is_safe():
    # `nxt in "*+?{"` is True for the empty string, so a pattern merely
    # ENDING in a quantified group was falsely rejected — (a+) and
    # (a+)(b+) are not nested-quantifier shapes (caught while verifying
    # an earlier round r4; missed by every review round).
    assert not A._risky_pattern("(a+)")
    assert not A._risky_pattern("(a+)(b+)")
    assert not A._risky_pattern("(a+)$")
    assert A._risky_pattern("(a+)+")  # still caught, of course


def test_grep_match_cap_break_does_not_depend_on_note_wording(monkeypatch):
    # The outer break uses an explicit cut_short flag, not a string sniff
    # of the last note (an earlier review round) — a file that hits the match
    # cap and a file that hits the time cap both stop the scan cleanly.
    box = _box_paths(monkeypatch, {"a.py": "\n".join(f"m{i}" for i in range(500)),
                                    "b.py": "m\n"}, ["a.py", "b.py"])
    out = box.grep("m")
    assert f"match cap reached at {A.GREP_MAX_MATCHES}" in out
    assert "searched 1 file(s)" in out  # capped file WAS searched (an earlier review round)
    assert "b.py" not in out.replace("scanned", "")  # b.py never scanned


def test_grep_byte_cap_one_aggregate_note(monkeypatch):
    # Once the cumulative byte cap is near, several files skip — ONE note,
    # not one per file (an earlier review round: the per-file notes were the
    # cap-amplified noise), and the "scanned" count stays exact.
    files = {"a.py": "bug\n", "big.py": "x" * (A.GREP_MAX_BYTES + 10),
             "b.py": "bug\n", "big2.py": "y" * (A.GREP_MAX_BYTES + 10)}
    box = _box_paths(monkeypatch, files, ["big.py", "a.py", "big2.py", "b.py"])
    out = box.grep("bug")
    assert out.count("skipped — byte cap") == 1
    assert "2 file(s) fetched but skipped — byte cap" in out
    assert "a.py:1: bug" in out and "b.py:1: bug" in out
    assert "searched 2 file(s)" in out


def test_grep_cache_full_note(monkeypatch):
    # A full per-review cache must say so, not report files as missing
    # (an earlier review round: a false "not found" is a false negative the model
    # acts on). a.py fills the cache, b.py trips it.
    box = _box_paths(monkeypatch, {"a.py": "bug\n", "b.py": "bug\n"},
                     ["a.py", "b.py"])
    monkeypatch.setattr(A, "FILE_CACHE_MAX_BYTES", 1)
    out = box.grep("bug")
    assert "file cache full" in out
    assert "a.py:1: bug" in out  # the file fetched before the cap still reports


def test_read_file_cache_full_error(monkeypatch):
    box = _box_paths(monkeypatch, {"a.py": "bug\n"}, ["a.py"])
    monkeypatch.setattr(A, "FILE_CACHE_MAX_BYTES", 0)
    out = box.read_file("a.py")
    assert "cache is full" in out


def test_grep_rejects_nested_quantifiers(monkeypatch):
    # Best-effort ReDoS guard (an earlier review round): the classic exponential shapes
    # are rejected outright; safe patterns and escaped literals pass.
    box = _box(monkeypatch, {"a.py": "x"})
    assert "pattern rejected" in box.grep("(a+)+$")
    assert "pattern rejected" in box.grep("(a*)*b")
    assert "pattern rejected" not in box.grep(r"\* pointer")   # escaped literal
    assert "pattern rejected" not in box.grep("colou?r")       # simple quantifier
    assert "no matches" in box.grep("colou?r")                 # and it ran


def test_risky_pattern_noncapturing_groups_pass():
    # Group prefixes are not quantifiers: (?:foo|bar)+ is a safe pattern
    # and must RUN, not error (an earlier review round). Real nesting inside a
    # noncapturing group is still rejected.
    assert not A._risky_pattern("(?:foo|bar)+")
    assert not A._risky_pattern(r"(?<=ab)c+")
    assert not A._risky_pattern(r"(?P<name>a|b)+")
    assert not A._risky_pattern(r"(?i)CASE")
    assert A._risky_pattern("(?:a+)+")          # nested — genuinely risky
    assert A._risky_pattern("(?P<x>a*)*b")     # nested — genuinely risky


def test_grep_noncapturing_search_runs(monkeypatch):
    box = _box(monkeypatch, {"a.py": "foobar\nbaz\n"})
    out = box.grep("(?:foo|bar)+")
    assert "foobar" in out and "pattern rejected" not in out


def test_risky_pattern_inline_flag_scoping():
    # an earlier review round claimed (?i:a+)+ slips past the guard — traced
    # and verified FALSE: the single group frame collects the inner a+, so
    # the close-with-quantifier check flags it. Pinned here so a future
    # _risky_pattern refactor cannot regress it.
    assert A._risky_pattern("(?i:a+)+")         # nested — genuinely risky
    assert not A._risky_pattern("(?i:a|b)+")    # single quantifier — safe
    assert A._risky_pattern("(?i)(a+)+")        # flag group then nesting
    assert not A._risky_pattern("(?i)CASE")     # bare flags, no group


def test_grep_discloses_diff_scope_truncation(monkeypatch):
    # >200 changed files: the scope label must say the bound bit, like the
    # directory branch already did (an earlier review round).
    diff = "".join(
        f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n"
        f"@@ -1 +1 @@\n-old\n+new\n"
        for i in range(A.GREP_MAX_FILES + 5)
    )
    box = A.ToolBox("o/r", "deadbeef", "tok", diff)
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n")
    out = box.grep("bug")
    assert f"truncated at {A.GREP_MAX_FILES} files" in out


def test_grep_reports_unreadable_files(monkeypatch):
    # A file in the diff hunks but not readable at the sha (deleted, or
    # over 1MB) is counted and disclosed, not silently dropped from the
    # "scanned" number.
    box = _box_paths(monkeypatch, {"a.py": "bug\n"}, ["a.py", "b.py"])
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n" if path == "a.py" else None)
    out = box.grep("bug")
    assert "1 unreadable" in out


def test_grep_directory_scope_is_recursive(monkeypatch):
    """One call at a directory covers its whole subtree (an earlier round): the
    non-recursive grep made find-usages a directory-by-directory walk —
    15+ calls on the an earlier round head probing web/src/pages, web/src/components, …
    one at a time. The tree is fetched ONCE per review."""
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    files = {"src/x.py": "alpha\nbeta bug\n", "src/y.py": "gamma\n",
             "src/sub/deep/z.py": "delta bug\n"}
    tree_calls = []

    def fake_tree(repo, ref, token):
        tree_calls.append(ref)
        return sorted(files), False

    monkeypatch.setattr(fetch_mod, "fetch_tree_paths", fake_tree)
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: files.get(path))
    out = box.grep("bug", "src")
    assert tree_calls == ["deadbeef"]  # one trees-API call, then cached
    assert "src/x.py:2: beta bug" in out
    assert "src/sub/deep/z.py:1: delta bug" in out  # the subtree, not just direct children
    assert "directory src (recursive)" in out


def test_grep_file_scope_still_single_file(monkeypatch):
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr(fetch_mod, "fetch_tree_paths",
                        lambda repo, ref, token: (["src/x.py", "src/y.py"], False))
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n" if path == "src/x.py" else None)
    out = box.grep("bug", "src/x.py")
    assert "file src/x.py" in out and "recursive" not in out
    assert "src/x.py:1: bug" in out


def test_grep_discloses_a_tree_listing_github_truncated(monkeypatch):
    """GitHub truncates recursive tree listings past ~100k entries; the
    scope label must say some paths were unsearchable, never silently
    narrow the scan."""
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr(fetch_mod, "fetch_tree_paths",
                        lambda repo, ref, token: (["src/x.py"], True))
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n")
    out = box.grep("bug", "src")
    assert "GitHub truncated the tree listing" in out


def test_grep_subtree_cap_truncates_at_max_files(monkeypatch):
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    tree = [f"src/f{i}.py" for i in range(A.GREP_MAX_FILES + 5)]
    monkeypatch.setattr(fetch_mod, "fetch_tree_paths",
                        lambda repo, ref, token: (tree, False))
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n")
    monkeypatch.setattr(A, "GREP_MAX_FILES", 3)
    out = box.grep("bug", "src")
    assert "truncated at 3 files" in out


def test_grep_tree_fetch_failure_answers_the_turn(monkeypatch):
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")

    def boom_tree(repo, ref, token):
        raise fetch_mod.GitHubError(0, "/git/trees", "conn reset")

    def boom_listing(repo, ref, path, token):
        raise fetch_mod.GitHubError(0, "/contents", "conn reset")

    monkeypatch.setattr(fetch_mod, "fetch_tree_paths", boom_tree)
    monkeypatch.setattr(fetch_mod, "fetch_dir_listing", boom_listing)
    out = box.grep("bug", "src")
    assert out.startswith("error: GitHub API failure scoping grep")


def test_grep_tree_failure_falls_back_to_the_listing(monkeypatch):
    """an earlier review round: an explicit FILE path must stay searchable when the
    trees API is down — the directory-listing branch is the fallback, and
    a file path never needed a listing at all."""
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")

    def boom(repo, ref, token):
        raise fetch_mod.GitHubError(0, "/git/trees", "conn reset")

    monkeypatch.setattr(fetch_mod, "fetch_tree_paths", boom)
    monkeypatch.setattr(fetch_mod, "fetch_dir_listing",
                        lambda repo, ref, path, token: None)
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n")
    out = box.grep("bug", "src/x.py")
    assert "file src/x.py" in out and "src/x.py:1: bug" in out

    calls = []

    def fake_listing(repo, ref, path, token):
        calls.append(path)
        return [{"name": "x.py", "type": "file"}]

    monkeypatch.setattr(fetch_mod, "fetch_dir_listing", fake_listing)
    out = box.grep("bug", "src")
    assert calls == ["src"]
    assert "direct files — recursive tree unavailable" in out


def test_fallback_listing_at_githubs_cap_discloses_incomplete_coverage(monkeypatch):
    """an earlier review round: the contents API caps a listing at 1000 entries with NO
    truncated flag — a bigger directory silently under-scopes in the
    tree-failure fallback. At the cap the label must say coverage may be
    incomplete, never claim a complete scan silently."""
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")

    def boom(repo, ref, token):
        raise fetch_mod.GitHubError(0, "/git/trees", "conn reset")

    monkeypatch.setattr(fetch_mod, "fetch_tree_paths", boom)
    big = [{"name": f"f{i}.py", "type": "file"} for i in range(A.LISTING_CAP)]
    monkeypatch.setattr(fetch_mod, "fetch_dir_listing",
                        lambda repo, ref, path, token: big)
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "bug\n")
    out = box.grep("bug", "src")
    assert "may be truncated at GitHub's 1000-entry cap" in out


def test_partial_grep_is_not_cached_so_a_retry_reruns_it(monkeypatch):
    """an earlier review round: a grep that hit a scan cap came back INCOMPLETE — the
    duplicate guard must not remember it, or the model's retry becomes a
    pointer to the partial result and the unsearched files stay
    unsearched."""
    box = _box(monkeypatch, {"a.py": "bug\nbug\n"})
    scans = []

    def slow_scan(rx, text, path, start_count, budget_s, context_lines=0):
        scans.append(path)
        return [], 0, 0, True, False  # timed out before any line finished

    monkeypatch.setattr(A, "_timed_scan", slow_scan)
    args = '{"pattern": "bug"}'
    first = box.dispatch("grep", args)
    assert "no files were searched" in first and "scan incomplete" in first
    second = box.dispatch("grep", args)
    assert len(scans) == 2  # re-executed, not pointer-answered
    assert "already answered" not in second


def test_grep_context_lines_echo_surroundings_indented(monkeypatch):
    """context_lines (an earlier round): the neighborhood around a match rides
    with the grep, indented — the follow-up read_file turn is the thing
    this saves (9 read_file calls on the an earlier round head were partly that)."""
    text = "\n".join(f"line{i} bug" if i == 4 else f"line{i}"
                     for i in range(1, 9))
    box = _box(monkeypatch, {"a.py": text})
    out = box.grep("bug", "a.py", context_lines=2)
    lines = out.splitlines()
    assert "a.py:4: line4 bug" in lines
    assert "  a.py:2: line2" in lines and "  a.py:3: line3" in lines
    assert "  a.py:5: line5" in lines and "  a.py:6: line6" in lines
    assert "  a.py:1:" not in out and "  a.py:7:" not in out
    # default stays the old shape: no context without asking
    plain = box.grep("bug", "a.py")
    assert "  a.py:3:" not in plain


def test_grep_context_does_not_trip_the_match_cap(monkeypatch):
    """Context echoes are display-only — the cap counts real matches, or a
    chatty context setting would silently shrink the search."""
    monkeypatch.setattr(A, "GREP_MAX_MATCHES", 3)
    text = "\n".join(
        f"line{i} bug" if i in (2, 8, 15) else f"line{i}"
        for i in range(1, 20))
    box = _box(monkeypatch, {"a.py": text})
    out = box.grep("bug", "a.py", context_lines=3)
    assert f"match cap reached at 3" in out
    assert out.count(": line") - out.count("  a.py") == 3  # 3 real matches
    assert out.count("  a.py") >= 4  # context still echoed around them


def test_grep_cap_still_echoes_the_last_matchs_trailing_context(monkeypatch):
    """an earlier review round: hitting the match cap used to exit before the last
    match's trailing window — the caller asked for that context and a cut
    mid-window reads as a broken scan."""
    monkeypatch.setattr(A, "GREP_MAX_MATCHES", 3)
    text = "\n".join(
        f"line{i} bug" if i in (2, 8, 15) else f"line{i}"
        for i in range(1, 25))
    box = _box(monkeypatch, {"a.py": text})
    out = box.grep("bug", "a.py", context_lines=3)
    assert "  a.py:16: line16" in out
    assert "  a.py:17: line17" in out
    assert "  a.py:18: line18" in out
    assert "match cap reached at 3" in out


def test_grep_cap_window_freezes_in_a_dense_match_file(monkeypatch):
    """an earlier review round: post-cap matching lines must not re-enter the
    match branch — in a file where everything matches, bumping trail_to on
    each new match chained the scan past the cap entirely."""
    monkeypatch.setattr(A, "GREP_MAX_MATCHES", 3)
    text = "\n".join(f"line{i} bug" for i in range(1, 500))
    box = _box(monkeypatch, {"a.py": text})
    out = box.grep("bug", "a.py", context_lines=3)
    # exactly the cap's matches, plus its one frozen trailing window
    assert out.count(": line") - out.count("  a.py") == 3
    assert "match cap reached at 3" in out
    assert "line200" not in out  # the scan stopped at the cap, not the EOF


def test_grep_cap_at_eof_still_reports_capped_and_stays_uncached(monkeypatch):
    """an earlier review round: a cap reached within context_lines of EOF never hits
    the frozen-window exit — the EOF done record used to say capped=False,
    so the cap note vanished, later files silently went unscanned, and the
    duplicate guard cached the incomplete grep as complete."""
    monkeypatch.setattr(A, "GREP_MAX_MATCHES", 3)
    diff = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n"
            "-old\n+new\n"
            "diff --git a/b.py b/b.py\n--- b/b.py\n+++ b/b.py\n@@ -1 +1 @@\n"
            "-old\n+new\n")
    box = A.ToolBox("o/r", "deadbeef", "tok", diff)
    files = {"a.py": "\n".join(f"line{i} bug" if i >= 8 else f"line{i}"
                              for i in range(1, 11)) + "\n",
            "b.py": "bug\n"}
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: files.get(path))
    out = box.grep("bug", context_lines=3)
    assert "match cap reached at 3" in out      # the cap is disclosed again
    # and the incomplete result is not cached as complete
    again = box.dispatch("grep", '{"pattern": "bug", "context_lines": 3}')
    assert "already answered" not in again


def test_grep_context_lines_validation(monkeypatch):
    box = _box(monkeypatch, {"a.py": "bug\n"})
    assert "between 0 and 3" in box.grep("bug", context_lines=7)
    assert "between 0 and 3" in box.grep("bug", context_lines=-1)
    assert "between 0 and 3" in box.grep("bug", context_lines="2")


def test_repo_map_is_a_bounded_two_level_layout(monkeypatch):
    """First-turn orientation for free (an earlier round): the tree is fetched
    for recursive grep anyway — a bounded layout removes discovery
    list_dir turns and scopes greps from the start."""
    from wrasse.platforms import github as fetch_mod
    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    tree = ["README.md", "ml/x.py", "ml/platform/y.py", "ml/platform/infra/z.py",
            "web/src/a.tsx", "docs/g.md"]
    monkeypatch.setattr(fetch_mod, "fetch_tree_paths",
                        lambda repo, ref, token: (tree, False))
    m = box.repo_map()
    assert "ml/" in m and "  platform" in m and "README.md" in m
    assert "web/" in m and "  src" in m
    # bounded and honest (fresh ToolBox: the tree is cached per review)
    big = [f"d{i}/f{j}.py" for i in range(400) for j in range(3)]
    box2 = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr(fetch_mod, "fetch_tree_paths",
                        lambda repo, ref, token: (big, True))
    m = box2.repo_map(max_chars=500)
    assert "layout truncated" in m and "some paths missing" in m
    assert len(m) < 2000


def test_repo_map_degrades_to_empty_when_the_tree_is_unavailable(monkeypatch):
    from wrasse.platforms import github as fetch_mod

    def boom(repo, ref, token):
        raise fetch_mod.GitHubError(0, "/git/trees", "down")

    box = A.ToolBox("o/r", "deadbeef", "tok", "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr(fetch_mod, "fetch_tree_paths", boom)
    assert box.repo_map() == ""


def test_addendum_teaches_truncation_recovery():
    """The recovery path for a truncated recursive search is narrowing the
    path — the guidance must not forbid it (an earlier review round analysis)."""
    assert "narrow the path" in A.AGENT_ADDENDUM


def test_duplicate_call_answers_with_a_pointer_not_a_reexecution(monkeypatch):
    """Verbatim repeats are budget waste (an earlier round: 5/39 calls on one
    head) — content at the pinned sha never changes, so a repeat can never
    add information. The second identical call points at the first result."""
    box = _box(monkeypatch, {"a.py": "the content\n"})
    fetches = []

    real_file = fetch_mod.fetch_file

    def counting_file(repo, ref, path, token):
        fetches.append(path)
        return "the content\n"

    monkeypatch.setattr(fetch_mod, "fetch_file", counting_file)
    first = box.dispatch("read_file", '{"path": "a.py"}')
    second = box.dispatch("read_file", '{"path": "a.py"}')
    assert first == "the content\n"
    assert "already answered earlier in this review" in second
    assert fetches == ["a.py"]  # executed once, not twice
    # Different arguments are a different call and still execute.
    monkeypatch.setattr(fetch_mod, "fetch_file",
                        lambda repo, ref, path, token: "other\n")
    assert box.dispatch("read_file", '{"path": "b.py"}') == "other\n"


def test_duplicate_guard_keeps_failed_calls_retryable(monkeypatch):
    """Only SUCCESSFUL calls are remembered: a transient GitHub failure must
    not poison the tool for the rest of the review."""
    box = _box(monkeypatch, {"a.py": "fixed\n"})
    state = {"fail": True}

    def flaky_file(repo, ref, path, token):
        if state["fail"]:
            raise fetch_mod.GitHubError(502, "/contents", "boom")
        return "fixed\n"

    monkeypatch.setattr(fetch_mod, "fetch_file", flaky_file)
    first = box.dispatch("read_file", '{"path": "a.py"}')
    assert first.startswith("error: GitHub API failure")  # failed → not cached
    state["fail"] = False
    assert box.dispatch("read_file", '{"path": "a.py"}') == "fixed\n"  # retry works


def test_grep_match_cap(monkeypatch):
    box = _box(monkeypatch, {"a.py": "\n".join(f"match{i}" for i in range(500))})
    out = box.grep("match")
    assert f"match cap reached at {A.GREP_MAX_MATCHES}" in out
    assert out.count(": match") <= A.GREP_MAX_MATCHES


def test_grep_skips_oversize_lines(monkeypatch):
    box = _box(monkeypatch, {"a.py": "x" * 10_000 + "\nshort bug line\n"})
    out = box.grep("bug")
    assert "short bug line" in out


# ---------------------------------------------------------------- post-an earlier round follow-ups


def test_final_truncated_propagates():
    # A turn the gateway marked "incomplete" (output-token limit) must not
    # masquerade as a finished answer: the flag rides out to the status and
    # summary (an earlier review round) — on the normal exit and the forced-final exit.
    t = _Transport([([_msg(FINDINGS_JSON)], "incomplete")])
    reply = _run(t)
    assert reply.final_truncated

    t = _Transport([
        ([_fc("c1", "grep", '{"pattern": "x"}')], "completed"),
        ([_msg(FINDINGS_JSON)], "incomplete"),
    ])
    reply = _run(t, max_tool_turns=1)
    assert reply.budget_exhausted and reply.final_truncated

    t = _Transport([[_msg(FINDINGS_JSON)]])
    assert not _run(t).final_truncated


def test_unavailable_message_names_the_actual_gap(monkeypatch):
    # an earlier review round: a replay run with a sha but no token used to claim
    # the SHA was missing. The message must name what is actually absent.
    box = A.ToolBox("o/r", "deadbeef", "", "diff --git a/a b/a\n")
    assert "no GitHub token" in box.dispatch("read_file", '{"path": "a.py"}')
    local = A.ToolBox("o/r", "local", "tok", "diff --git a/a b/a\n")
    assert "--head-sha" in local.dispatch("read_file", '{"path": "a.py"}')
    assert "no GitHub token" not in local.dispatch("read_file", '{"path": "a.py"}')


def test_grep_searches_full_line_shows_truncated(monkeypatch):
    # an earlier review round: a match past column 400 is a real finding — grep searches
    # the FULL line and truncates only the display, with a visible marker.
    long_line = "x" * 500 + " needle"
    box = _box(monkeypatch, {"a.py": long_line + "\n"})
    out = box.grep("needle")
    # Pre-fix, the search saw only the first 400 chars: no a.py:1 line at
    # all. Now the full line is searched — the match IS reported…
    assert "a.py:1:" in out
    # …with the display clipped at 400 chars and a visible marker.
    assert "line truncated for display" in out
    assert "needle" not in out  # beyond the display window, by design


def test_oversize_reads_have_their_own_budget(monkeypatch):
    # an earlier review round: with oversize files uncached, the cache-full guard
    # would be silently unexhaustible — a review could refetch big blobs
    # forever. Uncached reads accrue against OVERSIZE_FETCH_MAX_BYTES and
    # trip the same _CacheFull signal.
    box = _box_paths(monkeypatch, {"a.py": "z" * 60}, ["a.py"])
    monkeypatch.setattr(A, "CACHEABLE_FILE_BYTES", 10)
    monkeypatch.setattr(A, "OVERSIZE_FETCH_MAX_BYTES", 100)
    out1 = box.read_file("a.py")
    assert out1.startswith("z")           # first read fine (and uncached)
    out2 = box.read_file("a.py")
    assert "cache is full" in out2         # 60 + 60 > 100 → budget tripped


def test_grep_timeout_keeps_matches_already_found(monkeypatch):
    # an earlier review round: a scan killed on a runaway line must not lose the
    # matches it already found — the worker streams each match as it is
    # found, so a kill mid-file keeps them. r3: the match event carries
    # the searched-line count, so the file also COUNTS as searched.
    text = "the needle line\n" + "a" * 40 + "b" + "\n"
    box = _box(monkeypatch, {"a.py": text})
    monkeypatch.setattr(A, "GREP_SCAN_TIME_S", 0.5)
    out = box.grep("needle|(a|a)+$")  # matches line 1 fast, then stalls
    assert "a.py:1: the needle line" in out
    assert "a.py scan incomplete — per-file time cap" in out
    assert "searched 1 file(s)" in out  # its match proves it was searched
    assert "1 scan(s) incomplete" in out  # and the head says it was partial


def test_worker_death_without_done_is_inconclusive(monkeypatch):
    # an earlier review round: a worker that dies before writing its done record
    # (crash, not kill) must not read as a clean scan — EOF without done
    # is the same verdict as a timeout.
    monkeypatch.setattr(
        A, "_SCAN_WORKER_SRC",
        'import json, sys\n'
        'sys.stdout.write(json.dumps({"t": "m", "v": "a.py:1: x", "l": 1}) + "\\n")\n'
        'sys.stdout.flush()\n')
    import re as re_mod
    matches, match_count, lines, timed_out, capped = A._timed_scan(
        re_mod.compile("x"), "x\n", "a.py", 0, 5.0)
    assert timed_out and not capped
    assert matches == ["a.py:1: x"] and match_count == 1 and lines == 1


def test_grep_runaway_search_is_abandoned_not_fatal(monkeypatch):
    # an earlier review round security: (a|a)+ — ambiguous alternation, invisible to
    # _risky_pattern — gets real fuel from full-line search. The
    # worker-process budget must kill it: the grep RETURNS (within its
    # scan budget), discloses the incomplete file, and the review lives on.
    assert not A._risky_pattern("(a|a)+$")  # the hole this test guards
    bomb = "a" * 40 + "b"          # exponential backtracking, never matches
    box = _box(monkeypatch, {"a.py": bomb})
    monkeypatch.setattr(A, "GREP_SCAN_TIME_S", 0.2)
    out = box.grep("(a|a)+$")
    assert "a.py scan incomplete — per-file time cap" in out
    assert "searched 0 file(s)" in out  # nothing counted as searched


def test_grep_per_file_timeout_continues_to_next_file(monkeypatch):
    # an earlier review round: a per-file ceiling hit is NOT a verdict on the whole
    # grep — the slow file is disclosed and skipped, later files still get
    # scanned while the total budget lasts.
    files = {"a.py": "a" * 40 + "b", "b.py": "the needle\n"}
    box = _box_paths(monkeypatch, files, ["a.py", "b.py"])
    monkeypatch.setattr(A, "PER_FILE_SCAN_S", 0.2)
    out = box.grep("needle|(a|a)+$")
    assert "a.py scan incomplete — per-file time cap" in out
    assert "b.py:1: the needle" in out  # the file AFTER the stall still ran
    assert "searched 1 file(s)" in out


def test_grep_first_file_failure_is_not_no_matches(monkeypatch):
    # an earlier review round: when the FIRST file's fetch fails, "no matches"
    # would read as pattern-absent — but nothing was searched. The output
    # must say so.
    box = _box_paths(monkeypatch, {}, ["a.py", "b.py"])

    def failing(repo, ref, path, token):
        raise fetch_mod.GitHubError(500, path, "boom")

    monkeypatch.setattr(fetch_mod, "fetch_file", failing)
    out = box.grep("bug")
    assert "no files were searched" in out
    assert "no matches" not in out
    assert "GitHub API failure reading a.py" in out


def test_grep_keeps_matches_when_a_later_fetch_fails(monkeypatch):
    # an earlier review round: a fetch failure on a LATER file must not discard the
    # matches earlier files already found — partial answer + disclosed
    # failure, never a bare error that buries discovered findings.
    box = _box_paths(monkeypatch, {"a.py": "bug\n"}, ["a.py", "b.py"])

    def flaky(repo, ref, path, token):
        if path == "b.py":
            raise fetch_mod.GitHubError(500, path, "boom")
        return "bug\n"

    monkeypatch.setattr(fetch_mod, "fetch_file", flaky)
    out = box.grep("bug")
    assert "a.py:1: bug" in out
    assert "GitHub API failure reading b.py" in out


def test_grep_scan_time_cap_skips_remaining_files(monkeypatch):
    # The per-FILE gate: with the scan budget already spent, no worker is
    # spawned at all — remaining files are not fetched, and the output
    # says so. (A pre-exhausted budget, not a fake clock: the worker
    # deadline math makes several time.monotonic() calls, so stepped
    # clocks no longer model it deterministically.)
    box = _box_paths(monkeypatch, {"a.py": "bug\n", "b.py": "bug\n"},
                     ["a.py", "b.py"])
    monkeypatch.setattr(A, "GREP_SCAN_TIME_S", -1.0)
    out = box.grep("bug")
    assert "scan-time cap reached — remaining files not scanned" in out
    assert "searched 0 file(s)" in out
    assert "a.py:1" not in out and "b.py:1" not in out


def test_oversize_files_are_not_cached(monkeypatch):
    # an earlier review round: one grep's byte-cap skips must not pin multi-MB blobs in
    # the shared per-review cache and starve later targeted reads. Files
    # over CACHEABLE_FILE_BYTES are fetched per-use and never cached.
    fetches = []

    def fake_fetch(repo, ref, path, token):
        fetches.append(path)
        return "z" * (A.CACHEABLE_FILE_BYTES + 100)

    box = A.ToolBox("o/r", "deadbeef", "tok",
                    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n")
    monkeypatch.setattr(fetch_mod, "fetch_file", fake_fetch)
    out1 = box.read_file("a.py")
    out2 = box.read_file("a.py")
    assert len(out1) == len(out2)
    assert fetches == ["a.py", "a.py"]     # refetched: not cached


def test_summary_agent_header_not_one_turn():
    # an earlier review round: an agentic v2 review must not call itself a one-turn v1
    # test in its own summary header.
    from wrasse.core import review as R

    class _Reply:
        elapsed_s = 1.0
        prompt_tokens = 1
        completion_tokens = 1

    s = R.build_summary("m", _Reply(), [], [], dropped_paths=0, dropped_cap=0,
                        agent=True)
    # Version-advertised header (REVIEW_VERSION, an earlier round): match the mode
    # word + version parenthetical, not a pinned digit string.
    import re as _re
    assert _re.search(r"agentic review \(v[\d.]+\)", s) and "one-turn" not in s
    s1 = R.build_summary("m", _Reply(), [], [], dropped_paths=0, dropped_cap=0)
    assert "one-turn" in s1 and "agentic" not in s1


# ---------------------------------------------------------------- CLI guards


def test_head_sha_requires_diff_file():
    import pytest
    from wrasse.core import review as R

    with pytest.raises(SystemExit):
        R.main(["--pr", "1", "--head-sha", "abc", "--dry-run"])


def test_resolved_history_feeds_first_turn():
    # an earlier round pre-A/B: previously-raised RESOLVED threads go into the
    # loop's first turn as do-not-re-derive context — without them the
    # reviewer re-derives the same shapes every round (an earlier round: two rounds,
    # 17 findings, zero new). First-turn context only, no transport change.
    t = _Transport([[_msg(FINDINGS_JSON)]])
    _run(t, already_resolved=[
        "- `a.py:9` — pct() rank convention (resolved: by design)",
    ])
    first_input = t.calls[0]["input"]
    assert "already resolved" in first_input
    assert "pct() rank convention" in first_input
    # Later turns carry only tool outputs — the history rides the chain.
    assert len(t.calls) == 1


def test_transport_receives_wall_deadline():
    # The transport's one network retry is clamped to the review's wall
    # deadline (an earlier review round): an in-budget turn must not spend its
    # remaining wall clock twice. The first turn and later tool turns get
    # t0 + wall_timeout_s; the forced final gets the grace window on top.
    t = _Transport([[_fc("c1", "read_file", '{"path": "a.py"}'), ], [_msg(FINDINGS_JSON)]])
    _run(t, max_tool_turns=2, wall_timeout_s=600.0)
    d0 = t.calls[0]["deadline_s"]
    d1 = t.calls[1]["deadline_s"]
    assert d0 == d1  # both inside the wall budget: same absolute deadline
    assert t.calls[0]["timeout_s"] == 300.0  # unchanged per-turn semantics


def test_forced_final_deadline_includes_grace():
    script = [[_fc("c1", "read_file", '{"path": "a.py"}')], [_msg("[]")]]
    t = _Transport(script)
    reply = _run(t, max_tool_turns=1, wall_timeout_s=600.0)
    forced = t.calls[-1]
    assert reply.budget_exhausted
    # forced final's deadline = wall + grace, its timeout ≤ grace
    assert forced["deadline_s"] > t.calls[0]["deadline_s"]
    assert forced["timeout_s"] <= A.FORCED_FINAL_GRACE_S
