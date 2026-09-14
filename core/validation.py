# core/validation.py
"""
Server-side re-validation of Streamlit widget values.

Streamlit 1.48's widget deserializers do not enforce `min_value`/`max_value`
on `number_input`/`slider`, do not check membership for `selectbox`/
`multiselect` (an unknown value is returned as-is), and ignore `max_chars`
on `text_input` — see docs/security_audit_2026-09.md §2.1. Every value read
from a widget must be passed back through one of these helpers before it is
used for an array/image size, a loop/step count, a file path lookup, or
anything reaching `eval` — never trust the widget's own bounds.

Pure logic, no Streamlit import — testable in isolation.
"""
from __future__ import annotations

import math
import re
from typing import Sequence, TypeVar

T = TypeVar("T")


class InvalidInput(ValueError):
    """Raised when a value read from a widget fails server-side validation."""


def num(value, lo: float, hi: float, name: str, integer: bool = False):
    """Validate a numeric widget value is finite and within [lo, hi]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InvalidInput(f"{name}: not a numeric value.")
    if not math.isfinite(v):
        raise InvalidInput(f"{name}: value is not finite.")
    if not (lo <= v <= hi):
        raise InvalidInput(f"{name} must be between {lo} and {hi}.")
    return int(v) if integer else v


def choice(value: T, allowed: Sequence[T], name: str) -> T:
    """Validate a selectbox/radio value is one of the declared options.

    Indispensable: `selectbox` returns an unrecognized client value as-is
    instead of raising.
    """
    if value not in allowed:
        raise InvalidInput(f"{name}: option not allowed.")
    return value


def choices(values: Sequence[T], allowed: Sequence[T], name: str) -> list[T]:
    """Validate every element of a multiselect value is an allowed option."""
    allowed_set = set(allowed)
    bad = [v for v in values if v not in allowed_set]
    if bad:
        raise InvalidInput(f"{name}: option(s) not allowed ({bad!r}).")
    return list(values)


def text(value: str, name: str, max_len: int = 64,
         pattern: str = r"^[\w .\-]*$") -> str:
    """Validate a text_input value: enforce max length and a safe pattern.

    `max_chars` on `text_input` is a browser-side hint only.
    """
    v = str(value)[:max_len]
    if not re.match(pattern, v):
        raise InvalidInput(f"{name}: characters not allowed.")
    return v


# ─────────────────────────────────────────────────────────────────────────
# Filter-expression allowlist (V5)
# ─────────────────────────────────────────────────────────────────────────
# The expression built here ends up in oimodeler's oimUtils.oifitsFlagWithExpression,
# which does `flags = eval(expr)` against the module's own globals() — see
# docs/security_audit_2026-09.md §4.3. Not exploitable today because the app
# only ever formats floats into it, but any future free-text filter widget
# would turn this into unauthenticated RCE unless the expression is
# strictly allowlisted before it reaches that boundary. Do not widen
# `_ALLOWED_IDENTIFIERS`/`_ALLOWED_CHARS` without re-auditing this sink.

_ALLOWED_IDENTIFIERS = {"EFF_WAVE", "EFF_BAND", "LENGTH", "PA", "SPAFREQ"}
_ALLOWED_CHARS = re.compile(r"^[A-Za-z_0-9\s().<>=!&|+\-*/]*$")

# Tokenizer, not a bare identifier regex: the expressions actually built by
# pages/data.py interpolate Python floats, whose repr uses scientific
# notation for small wavelengths (e.g. "2.9e-06" for 2.9 µm in metres).
# A naive `[A-Za-z_][A-Za-z_0-9]*` scan (as in the audit's first draft)
# matches the bare "e" inside "2.9e-06" as a standalone identifier and
# rejects every legitimate expression. Matching the numeric-literal
# alternative first (including its exponent) consumes "e-06" as part of
# the number so it's never mistaken for an identifier.
_NUMBER = r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?"
_TOKEN = re.compile(rf"(?:{_NUMBER})|([A-Za-z_][A-Za-z_0-9]*)")


def filter_expression(expr: str) -> str:
    """Strict allowlist for an oimodeler filter expression before it can
    reach `oifitsFlagWithExpression`'s `eval()`."""
    if len(expr) > 300:
        raise InvalidInput("Filter expression is too long.")
    if not _ALLOWED_CHARS.match(expr):
        raise InvalidInput("Filter expression contains a disallowed character.")
    if "__" in expr:
        raise InvalidInput("Filter expression is not allowed.")
    for match in _TOKEN.finditer(expr):
        ident = match.group(1)
        if ident is not None and ident not in _ALLOWED_IDENTIFIERS:
            raise InvalidInput(f"Identifier not allowed in filter expression: {ident}")
    return expr
