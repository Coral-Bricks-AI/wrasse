"""One place for every string a post is recognized by.

Three of them are a contract with future runs of this bot, not cosmetics:

- ``MARKER`` — the tag inside the hidden HTML comment on every inline thread
  and summary. Re-reviews parse it to find their own threads, so changing it
  orphans every thread posted before the change.
- ``STATUS_CONTEXT`` — the commit-status name. A branch-protection rule or a
  merge gate targets this exact string.
- ``BOT_LOGINS`` — the logins this bot's comments arrive under. GitHub Actions
  reports ``github-actions`` on GraphQL and ``github-actions[bot]`` on REST,
  which is why both are listed; a GitHub App should set ``WRASSE_BOT_LOGINS``
  to its own login or the dedupe will not see its own threads.
"""

from __future__ import annotations

import os

from wrasse import __version__

STATUS_CONTEXT = "Wrasse Review"
MARKER = "wrasse"
# Every summary's first line starts with this; the duplicate-run guard and the
# summarize/minimize pass anchor on it rather than on finding prose.
SUMMARY_PREFIX = f"**{STATUS_CONTEXT}**"

# Comma-separated, override with $WRASSE_BOT_LOGINS when running as an App.
BOT_LOGINS: tuple[str, ...] = tuple(
    login.strip()
    for login in os.environ.get(
        "WRASSE_BOT_LOGINS", "github-actions,github-actions[bot]"
    ).split(",")
    if login.strip()
)

# The version every review advertises in its summary, so a result can always
# be attributed to the harness that produced it.
REVIEW_VERSION = __version__
