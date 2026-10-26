"""Model transport: one HTTP turn at a time, and nothing else.

Two shapes, because providers ship both:

- ``call_model`` — one turn on POST /v1/chat/completions.
- ``call_responses`` — one turn on POST /v1/responses, which the agentic loop
  threads with ``previous_response_id`` so reasoning carries across turns.

Everything provider-specific (endpoint, key, paths, retry policy, model
default) is a ``Provider`` record in ``wrasse.providers``. This module stays
transport-only: build the request, time it, shape-check the reply, and retry a
bounded number of genuinely safe failures (see ``_post_json``).

The optional ``x-session-id`` header tags every turn of one review with the
same id, so per-turn latency is attributable even when a review is many
requests.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# The default provider's endpoint. Every entry point takes ``base_url``,
# so another provider (or a staging deployment of the same one) is a
# flag, not a code change. The provider table is in wrasse.providers.
DEFAULT_BASE_URL = "https://inference.coralbricks.ai"


class GatewayError(RuntimeError):
    """Non-2xx or unreachable gateway. ``body`` is a short response excerpt."""

    def __init__(self, code: int, body: str):
        super().__init__(f"gateway HTTP {code}: {body[:500]}")
        self.code = code
        self.body = body


# One bounded retry for a transient transport failure (an earlier round pre-A/B,
# extended for 5xx on 2026-10-17). Two failure classes retry, each ONCE,
# with the same fixed backoff and deadline clamping:
#
#   * Network-level failures (GatewayError code 0 — unreachable, reset,
#     closed without response): on an earlier round two consecutive "Remote end closed
#     connection without response" failures posted a red status per head
#     and needed manual /wrasse retries.
#
#   * HTTP 502/503 from the gateway: on 2026-10-17 21:20:57 UTC (PR an earlier round
#     head review, run 39070307732) the gateway's owned-502 — a queued turn
#     whose hop deadline expired with "nowhere to hop" — killed a review
#     mid-loop. It was the ONLY 5xx of that hour across all gateway
#     traffic, and glm-5.3 served 200s one second either side, i.e. a
#     transient blip that a re-send would have survived. 502/503 are the
#     gateway saying "no backend could take this turn": the turn never
#     RAN, so a re-send cannot double-charge or duplicate output — except
#     the one named post-execution shape (upstream_stream_incomplete,
#     excluded below; an earlier review round). Other statuses do NOT retry — the
#     gateway ANSWERED, and re-sending a request it rejected
#     (400/404/429…) is not ours to decide. 504 and timeouts never retry
#     either: there the work may have completed upstream and a re-send
#     would double the wall clock and the charge.
#
# After one retry the failure propagates to the existing error-status path
# — a hung gateway must not double the wall clock of every review.
_NETWORK_RETRY_BACKOFF_S = 3.0
_HTTP_RETRY_STATUSES = frozenset({502, 503})
# Post-execution 502 shapes the harness CAN see and must NOT re-send:
# `upstream_stream_incomplete` (provider.py: the upstream stream ran,
# generated tokens, and ended without a terminal event) — a re-send
# repeats completed generation work and can double-charge (an earlier review round).
# Residual, accepted: other post-execution shapes (invalid_upstream_body,
# relayed vendor 502s) arrive in the same canonical internal_error
# envelope as the safe queue-hop class and are indistinguishable from
# the client; for those the one bounded re-send is the cheaper failure
# than red-statusing every transient review.
_HTTP_RETRY_EXCLUDED_CODES = frozenset({"upstream_stream_incomplete"})


def _is_post_execution_error(body: str) -> bool:
    """True if the error body names a failure that happened AFTER the
    upstream already did the work — a re-send would repeat it."""
    try:
        code = json.loads(body)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        # Not a JSON error envelope: transport junk, not a named verdict.
        return False
    # A malformed envelope (error.code an array/object) must not escape as
    # a TypeError from unhashable set membership: the orchestrator's handler
    # catches GatewayError, not TypeError, so the run would die without
    # posting its error status (an earlier review round). Unknown shapes read as
    # retryable, like an unparseable body.
    return isinstance(code, str) and code in _HTTP_RETRY_EXCLUDED_CODES


def _post_json(build_request, timeout_s: float, deadline_s: float | None = None):
    """POST one request and parse the JSON body, retrying once on a
    network-level failure or a re-sendable HTTP 502/503 (pre-dispatch
    gateway failures only — see _HTTP_RETRY_EXCLUDED_CODES). ``build_request`` is a
    zero-arg callable returning a fresh urllib Request — a connection
    killed mid-flight can leave a Request object in a state urllib will
    not accept twice. ``deadline_s`` is an optional absolute
    time.monotonic() deadline for the RETRY only (an earlier review round): the
    agent loop hands each turn only the wall clock it has left, so a
    retried turn must not grant that allowance twice — no meaningful
    budget left means no retry. A timeout is never retried (an earlier review round): the request had its full allowance and the gateway never
    answered; re-sending would double the wall clock of a hung gateway
    (and, on 504, possibly double-charge work that completed upstream).
    Returns ``(payload, elapsed_s)``; elapsed covers every attempt
    including the backoff, so a retried call is never reported as faster
    than it truly was."""
    t0 = time.monotonic()
    attempt = 1
    while True:
        try:
            with urllib.request.urlopen(build_request(), timeout=timeout_s) as r:
                return json.loads(r.read()), time.monotonic() - t0
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            # The gateway answered. Only 502/503 ("no backend could take
            # this turn" — the turn never ran) are ours to re-send, and
            # even those only when the error body does not name a
            # post-execution failure; every other status is a real answer,
            # and a timeout-shaped 504 means the work may have happened.
            if (e.code not in _HTTP_RETRY_STATUSES
                    or attempt == 2
                    or _is_post_execution_error(body)):
                raise GatewayError(e.code, body) from None
            failure_code, failure_detail = e.code, body
        except OSError as e:
            # URLError (with .reason) and any raw socket failure; HTTPError
            # is handled above. "Remote end closed connection without
            # response" arrives here as URLError(reason=RemoteDisconnected).
            reason = getattr(e, "reason", e)
            # socket.timeout is TimeoutError's alias (3.10+): the call had
            # its allowance and the gateway never answered.
            if isinstance(reason, TimeoutError):
                raise GatewayError(0, str(reason)) from None
            if attempt == 2:
                raise GatewayError(0, str(reason)) from None
            failure_code, failure_detail = 0, str(reason)
        # The one retry, shared by both failure classes: clamped to the
        # wall clock exactly like the original network retry.
        if deadline_s is not None:
            remaining = deadline_s - time.monotonic() - _NETWORK_RETRY_BACKOFF_S
            if remaining <= 1.0:
                # No retry budget left: failing now keeps the review
                # inside its wall deadline, where the forced-final path
                # can still produce an honest, flagged answer.
                raise GatewayError(
                    failure_code, f"no retry budget left ({remaining:.1f}s): "
                    f"{failure_detail}") from None
            timeout_s = min(timeout_s, remaining)
        time.sleep(_NETWORK_RETRY_BACKOFF_S)
        attempt = 2
        if deadline_s is not None:
            # Re-clamp to the POST-sleep clock: the pre-check is a
            # hopeful projection, and time.sleep can overshoot — the
            # retry's allowance is whatever is actually left, and none at
            # all means no retry (an earlier review round: the backoff is
            # pre-subtracted above, so the common case is already inside
            # budget; this closes the scheduler-jitter residual too).
            left = deadline_s - time.monotonic()
            if left <= 0:
                raise GatewayError(
                    failure_code, f"retry backoff consumed the remaining "
                    f"budget ({left:.1f}s): {failure_detail}") from None
            timeout_s = min(timeout_s, left)


@dataclass(frozen=True)
class ModelReply:
    content: str
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int


def call_model(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    base_url: str = DEFAULT_BASE_URL,
    session_id: str | None = None,
    timeout_s: float = 180.0,
    temperature: float = 0.0,
) -> ModelReply:
    """One-turn chat-completions call. The single seam v2 replaces."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["x-session-id"] = session_id
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    def build_request():
        return urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )

    payload, elapsed = _post_json(build_request, timeout_s)

    try:
        content = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
    except (KeyError, IndexError, TypeError) as e:
        raise GatewayError(200, f"unexpected response shape: {e}") from None
    if not isinstance(content, str):
        raise GatewayError(200, f"non-string content: {type(content).__name__}")
    return ModelReply(
        content=content,
        elapsed_s=elapsed,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


@dataclass(frozen=True)
class ResponsesTurn:
    """One /v1/responses turn, shaped for the agent loop (core/agent.py)."""

    id: str  # response id — the next turn's previous_response_id
    output: list  # raw output items (message / function_call / reasoning)
    elapsed_s: float
    input_tokens: int
    output_tokens: int
    # Gateway turn status: "completed" | "incomplete" (hit the output-token
    # limit; text can still be present and parseable) | "in_progress" etc.
    # The loop must be able to tell a FINISHED answer from a truncated one
    # (an earlier review round): a cut-short review may not read as a full one.
    status: str = ""


def call_responses(
    *,
    api_key: str,
    model: str,
    input,  # str (first turn) | list of input items (chained turns)
    base_url: str = DEFAULT_BASE_URL,
    session_id: str | None = None,
    instructions: str | None = None,
    tools: list | None = None,
    previous_response_id: str | None = None,
    timeout_s: float = 300.0,
    deadline_s: float | None = None,
    temperature: float = 0.0,
) -> ResponsesTurn:
    """One turn on POST /v1/responses — the v2 transport.

    ``instructions`` is sent on EVERY call, not just the first: the gateway
    does not inherit it across a ``previous_response_id`` chain (its own
    chain-assembly contract, responses_state.py), so a turn that omits it
    would silently drop the findings contract. ``deadline_s`` clamps the
    one transient-failure retry (network or 502/503) to the review's
    remaining wall clock (see _post_json).
    """
    url = base_url.rstrip("/") + "/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["x-session-id"] = session_id
    body = {"model": model, "input": input, "temperature": temperature}
    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
        # "auto" only: the Harmony channel rejects forced/required on
        # /v1/responses (the gateway's own smoke tests, responses_tools).
        body["tool_choice"] = "auto"
    if previous_response_id:
        body["previous_response_id"] = previous_response_id

    def build_request():
        return urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )

    payload, elapsed = _post_json(build_request, timeout_s, deadline_s)

    try:
        out = payload.get("output") or []
        usage = payload.get("usage") or {}
        if not isinstance(out, list):
            raise TypeError(f"output is {type(out).__name__}, expected list")
    except (AttributeError, TypeError) as e:
        raise GatewayError(200, f"unexpected response shape: {e}") from None
    return ResponsesTurn(
        id=str(payload.get("id") or ""),
        output=out,
        elapsed_s=elapsed,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        status=str(payload.get("status") or ""),
    )


def extract_text(items: list) -> str | None:
    """Concatenated output_text of the message items, or None if the turn
    produced no message at all (a pure tool-call or reasoning turn)."""
    texts = [
        part.get("text", "")
        for it in items
        if isinstance(it, dict) and it.get("type") == "message"
        for part in (it.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    if not texts:
        return None
    return "\n".join(t for t in texts if t)


def extract_function_calls(items: list) -> list:
    """The function_call items of a turn, in order (empty = final turn)."""
    return [
        it for it in items
        if isinstance(it, dict) and it.get("type") == "function_call"
    ]
