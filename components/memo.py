# components/memo.py
"""
Manual session_state memoization for expensive per-page computations.

Why not st.cache_data/st.cache_resource: those assume their inputs are
hashable/copyable, which is awkward or unsafe here — the call sites this
exists for need to reuse a live oimodeler object (oimModel, oimSimulator,
an emcee-backed oimFitterEmcee) across reruns, not a plain serializable
value.

Why this is needed at all: app.py's st.tabs() runs every page's
render() on *every* Streamlit rerun, not just the one currently visible
— and pages/modelling.py's and pages/fitting.py's own internal
st.tabs() do the exact same thing one (or two) levels deeper. A result-
display block with no caching at all redoes its work on every single
click anywhere in the whole app, for as long as that result stays
populated — typically the rest of the session, once at least one fit
has completed. See pages/modelling.py's Model summary fix for the
first instance of this; this module generalizes the same pattern for
pages/fitting.py's chi2/grid/emcee result tabs.

Usage:
    value = memoize(
        "emcee_walkers_plot", (id(er['lmfit']), discard, thin, chi2limfact),
        lambda: er['lmfit'].walkersPlot(discard=discard, thin=thin, chi2limfact=chi2limfact),
    )

`signature` must be a plain, comparable value (a tuple of primitives —
ids, numbers, strings — never a live object itself) built from exactly
the inputs that actually change `compute`'s result. Leaving one out
means a real change won't invalidate the cache (stale results);
including something irrelevant (e.g. a purely-cosmetic axis-range
widget) just means recomputing more often than necessary — safe, only
wasteful.
"""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st


def memoize(key: str, signature: Any, compute: Callable[[], Any]) -> Any:
    """Returns compute()'s result, reusing the previous one if the last
    call under this `key` had the same `signature` (compared by `==`)."""
    slot = f"_memo_{key}"
    cached = st.session_state.get(slot)
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = compute()
    st.session_state[slot] = (signature, value)
    return value
