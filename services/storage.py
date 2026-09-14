# services/storage.py
"""
Session-scoped upload storage.

Root cause fixed here (see docs/security_audit_2026-09.md V1/V3/V8):
- Uploads used to land in a flat, world-readable `/tmp/<client filename>`.
  Two sessions sharing a filename overwrote/leaked each other's data, and
  `f.name` was concatenated straight into a path (path traversal, V1).
- This module gives every browser tab its own random, non-guessable
  directory. The client-supplied filename is only ever used as a *label*
  after being sanitized; it never reaches the filesystem unsanitized.
- Per-file / per-session byte quotas and a file-count cap bound disk usage
  (V8); `purge_expired()` reclaims sessions that were never cleanly closed.

Usage
-----
    from services.storage import store, resolve_selected_paths

    path = store(uploaded_file)                  # -> Path, or raises ValueError
    paths = resolve_selected_paths(list_of_names)  # allowlist lookup, V2
"""
from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path

import streamlit as st

# Never /tmp: it is world-writable, often shared with other services, and
# on many distros it's a tmpfs (RAM) — see V6/V8. Overridable via env var
# so tests / containers can point it at a dedicated volume.
BASE_DIR = Path(os.environ.get("OIMODELER_UPLOAD_DIR", "/var/lib/oimodeler/uploads"))

MAX_FILE_BYTES    = 100 * 1024 * 1024   # 100 MB per file
MAX_SESSION_BYTES = 200 * 1024 * 1024   # 200 MB per session
MAX_FILES         = 10                  # per session
SESSION_TTL       = 3600                # seconds, used by purge_expired()
ALLOWED_EXT       = (".fits", ".oifits")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

# Opportunistic purge: without an external cron (see audit §5.3), a
# long-running dev/test process would otherwise never reclaim expired
# session directories. Throttled so it doesn't stat the whole tree on
# every single upload.
_PURGE_INTERVAL = 300  # seconds
_last_purge = 0.0


def _session_id() -> str:
    """Random, non-guessable identifier tied to the browser tab's session_state.

    This — not the filename — is what isolates one user's uploads from
    another's, closing V3 (the flat /tmp namespace + a cache keyed on the
    resulting path).
    """
    if "_upload_session_id" not in st.session_state:
        st.session_state["_upload_session_id"] = uuid.uuid4().hex
    return st.session_state["_upload_session_id"]


def session_dir() -> Path:
    d = BASE_DIR / _session_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    """Reduce a client-supplied filename to a safe basename.

    - `Path(name).name` strips any directory component (neutralizes `../`
      and absolute paths) — this alone closes V1.
    - Remaining characters are restricted to a strict allowlist.
    - The extension must be an OIFITS extension; Streamlit's own
      `enforce_filename_restriction` only checks the suffix too, so we
      don't rely on it as the sole guard.
    """
    name = Path(name).name
    name = _UNSAFE.sub("_", name)[:120]
    if not name or name.startswith(".") or not name.lower().endswith(ALLOWED_EXT):
        raise ValueError("File name or extension not allowed.")
    return name


def _looks_like_fits(head: bytes) -> bool:
    # A valid FITS file starts with the SIMPLE keyword card.
    return head[:6] == b"SIMPLE"


def _unique_dest(d: Path, name: str) -> Path:
    """Avoid silently overwriting a same-named file already in this session
    (see skill's "Do not" list: never overwrite a user's file without
    confirmation). Two different uploads with the same sanitized name get
    a short disambiguating suffix instead of clobbering each other.
    """
    dest = d / name
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    for _ in range(1000):
        candidate = d / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError("Could not allocate a unique file name.")


def store(uploaded) -> Path:
    """Persist an `st.file_uploader` item under the current session's
    directory. Raises ValueError with a user-safe message on any rejection.
    """
    _maybe_purge_expired()

    if uploaded.size > MAX_FILE_BYTES:
        raise ValueError(f"File too large (max {MAX_FILE_BYTES // 1024**2} MB).")

    data = uploaded.getbuffer()
    if not _looks_like_fits(bytes(data[:6])):
        raise ValueError("File content does not look like a FITS file.")

    d = session_dir()
    existing = [p for p in d.iterdir() if p.is_file()]
    if len(existing) >= MAX_FILES:
        raise ValueError(f"Too many files in this session (max {MAX_FILES}).")
    if sum(p.stat().st_size for p in existing) + len(data) > MAX_SESSION_BYTES:
        raise ValueError(
            f"Session storage quota exceeded (max {MAX_SESSION_BYTES // 1024**2} MB)."
        )

    safe_name = _safe_name(uploaded.name)
    dest = _unique_dest(d, safe_name).resolve()
    # Belt and braces: the final path must stay under the session directory.
    if not dest.is_relative_to(d.resolve()):
        raise ValueError("Invalid destination path.")

    dest.write_bytes(data)
    return dest


def resolve_selected_paths(names: list[str]) -> list[str]:
    """Resolve a list of client-supplied dataset names against the
    session's `loaded_files` allowlist (V2 fix).

    Never reconstructs a path from `names` — any name absent from
    `st.session_state.loaded_files` is silently dropped, not passed
    through to a filesystem call.
    """
    loaded = st.session_state.get("loaded_files", {})
    return [loaded[n] for n in names if n in loaded]


def purge_expired(now: float | None = None) -> None:
    """Remove session directories whose most recent write is older than
    SESSION_TTL. Intended to be run from a periodic task (cron / scheduler)
    — see docs/security_audit_2026-09.md §5.3 — but is also invoked
    opportunistically (throttled) from `store()` so uploads don't
    accumulate unbounded even without an external scheduler configured.
    """
    now = now if now is not None else time.time()
    if not BASE_DIR.exists():
        return
    for d in BASE_DIR.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > SESSION_TTL:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            # Another process may be racing us to purge/write; skip and retry next pass.
            continue


def _maybe_purge_expired() -> None:
    global _last_purge
    now = time.time()
    if now - _last_purge < _PURGE_INTERVAL:
        return
    _last_purge = now
    purge_expired(now)
