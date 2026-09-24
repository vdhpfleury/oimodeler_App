# tests/test_flash.py
"""
Unit tests for components/flash.py — the one-time success/warning message
that survives a `st.rerun()`.

Regression test for a real bug found live: a handler that called
`st.success(...)` immediately followed by `st.rerun()` never actually
showed that message to the user (confirmed across several buttons in this
app) — the RerunException interrupts the script before the element is
flushed to the browser. queue_flash()/show_pending_flash() route the
message through st.session_state so it survives to the next run instead.

Monkeypatches `streamlit.session_state` to a plain dict and
`streamlit.success`/`streamlit.warning`/`streamlit.info`/`streamlit.error`
to recording stubs — no real Streamlit runtime/session needed for this
module's pure session_state bookkeeping.
"""
from __future__ import annotations

import streamlit as st

from components.flash import queue_flash, show_pending_flash


def _patch_streamlit(monkeypatch):
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    calls: list[tuple[str, str]] = []
    for level in ("success", "warning", "info", "error"):
        def make_stub(level=level):
            def stub(message):
                calls.append((level, message))
            return stub
        monkeypatch.setattr(st, level, make_stub(level), raising=False)
    return calls


def test_queued_message_is_not_shown_before_the_next_run(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    queue_flash("k", "hello")
    assert calls == []  # nothing shown yet — that's the whole point


def test_show_pending_flash_displays_and_clears_a_queued_message(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    queue_flash("k", "hello")
    show_pending_flash("k")
    assert calls == [("success", "hello")]

    calls.clear()
    show_pending_flash("k")
    assert calls == []  # consumed — a second show is a no-op


def test_multiple_queued_messages_are_shown_in_order(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    queue_flash("k", "imported ok")
    queue_flash("k", "row 3 skipped", level="warning")
    queue_flash("k", "row 9 skipped", level="warning")
    show_pending_flash("k")
    assert calls == [
        ("success", "imported ok"),
        ("warning", "row 3 skipped"),
        ("warning", "row 9 skipped"),
    ]


def test_different_keys_do_not_interfere(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    queue_flash("a", "for a")
    queue_flash("b", "for b")
    show_pending_flash("a")
    assert calls == [("success", "for a")]

    calls.clear()
    show_pending_flash("b")
    assert calls == [("success", "for b")]
