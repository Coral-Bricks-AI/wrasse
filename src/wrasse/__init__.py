"""wrasse — an AI code-review agent that reads the code before it comments.

The cleaner wrasse is the reef fish that swims into a bigger fish's mouth and
removes the parasites. Same job here: it reads the pull request's diff, opens
the files around it, and posts only actual bugs.

Layout:

- ``wrasse.core``      the review itself: diff hunks, findings contract,
                       the bounded tool loop, the orchestrator.
- ``wrasse.providers`` the model transport. One HTTP call per turn.
- ``wrasse.platforms`` where reviews come from and go to (GitHub today).
- ``wrasse.tools``     the tools the reviewer may call (read_file, grep, …).

Everything runs on the standard library: no dependencies, no build step, and
the whole harness is small enough to read in an afternoon.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
