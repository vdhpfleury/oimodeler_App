# services/activity_log.py
"""
Per-session activity log.

A plain, timestamped, in-memory audit trail of what a user did this
session (uploads, filter changes, model edits, fit runs, downloads...).
It is never displayed on screen and never written to disk on its own —
it only ever gets serialized into text when a results zip is built
(core/results.build_results_zip's `extra_files`), so a downloaded result
carries a trace of how it was produced.

Design constraints (see the user's request):
- Must not impact computation performance: every call here is an O(1)
  append to a small in-memory list — negligible next to any fit's own
  compute cost. No disk I/O, no network calls, ever.
- Must never appear in the UI: no st.write/st.text/st.dataframe call
  anywhere in this module.
- Session-scoped: st.session_state is already isolated per browser tab
  (see services/storage.py's V1/V3 session isolation) — this reuses that
  same isolation rather than inventing a new "per-user" identity the app
  otherwise has none of (no login).
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

# Bounds memory for a very long-running session — old entries drop first,
# the log never grows unbounded (same spirit as services/storage.py's
# per-session quotas).
_MAX_ENTRIES = 1000


def log_event(action: str, details: str = "") -> None:
    """Append one timestamped entry to this session's activity log.

    Call this only for discrete, meaningful user actions (a button
    click, an upload, a value that actually changed) — never from code
    that runs on every Streamlit rerun regardless of what changed, or
    the log fills with noise instead of a useful trail.
    """
    log = st.session_state.setdefault("activity_log", [])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"{ts}  {action}" + (f" — {details}" if details else "")
    log.append(entry)
    if len(log) > _MAX_ENTRIES:
        del log[: len(log) - _MAX_ENTRIES]


def get_log_text() -> str:
    """Renders the current session's log as a plain-text block, for
    inclusion in a downloaded zip only — never shown in the UI."""
    log = st.session_state.get("activity_log", [])
    if not log:
        return "No activity recorded this session.\n"
    return "\n".join(log) + "\n"
