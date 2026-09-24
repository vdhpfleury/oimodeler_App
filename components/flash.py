# components/flash.py
"""
One-time success/warning message(s) that survive a `st.rerun()`.

A handler that calls `st.success(...)` (or `st.warning(...)`) immediately
followed by `st.rerun()` — needed whenever the action mutates state
something else on the page reads, e.g. adding/removing a component —
never actually shows that message to the user: `st.rerun()` raises a
RerunException right away, interrupting the script before that element
is flushed to the browser (confirmed live: the message never appeared,
not even briefly, across several buttons using this exact pattern).
`queue_flash()` + `show_pending_flash()` route the message(s) through
`st.session_state` instead, so they render on the *next* run — the one
the rerun itself triggers — rather than the one that's about to be
discarded. Queues (not overwrites) so a handler that shows both a
success and one or more follow-up warnings (e.g. an import that
succeeded but flagged a few rows) keeps all of them, in order.
"""
from __future__ import annotations

import streamlit as st

_PREFIX = "_flash_"


def queue_flash(key: str, message: str, level: str = "success") -> None:
    """Call from a button handler in place of a direct `st.success(...)`/
    `st.warning(...)` right before `st.rerun()`. `level`: "success",
    "warning", "error", or "info" (matches the st.<level> to call)."""
    slot = f"{_PREFIX}{key}"
    st.session_state.setdefault(slot, [])
    st.session_state[slot].append((level, message))


def show_pending_flash(key: str) -> None:
    """Call once near the top of the render function that owns `key`'s
    button(s) — every rerun, not just the one right after the action, so
    it fires exactly once on the run queue_flash() scheduled it for."""
    slot = f"{_PREFIX}{key}"
    for level, message in st.session_state.pop(slot, []):
        getattr(st, level)(message)
