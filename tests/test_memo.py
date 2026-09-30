# tests/test_memo.py
"""Unit tests for components/memo.py's manual session_state memoization."""
from __future__ import annotations

import streamlit as st

from components.memo import memoize


def test_same_signature_reuses_the_cached_value(monkeypatch):
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    calls = []

    def compute():
        calls.append(1)
        return object()

    v1 = memoize("k", ("a", 1), compute)
    v2 = memoize("k", ("a", 1), compute)
    assert v1 is v2
    assert len(calls) == 1


def test_different_signature_recomputes(monkeypatch):
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    calls = []

    def compute():
        calls.append(1)
        return object()

    v1 = memoize("k", ("a", 1), compute)
    v2 = memoize("k", ("a", 2), compute)
    assert v1 is not v2
    assert len(calls) == 2


def test_different_keys_do_not_interfere(monkeypatch):
    monkeypatch.setattr(st, "session_state", {}, raising=False)

    v1 = memoize("k1", "sig", lambda: "one")
    v2 = memoize("k2", "sig", lambda: "two")
    assert v1 == "one"
    assert v2 == "two"


def test_none_signature_is_not_treated_as_cache_miss_sentinel(monkeypatch):
    # A compute() that legitimately returns None must not be recomputed
    # on the next call with the same signature.
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    calls = []

    def compute():
        calls.append(1)
        return None

    v1 = memoize("k", "sig", compute)
    v2 = memoize("k", "sig", compute)
    assert v1 is None
    assert v2 is None
    assert len(calls) == 1
