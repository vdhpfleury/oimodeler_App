# core/job_status.py
"""
File-based status channel between a background fit worker process
(core/fit_worker.py, spawned by services/jobs.py) and whatever polls it —
deliberately dumb (one small JSON file per job) rather than a
multiprocessing.Manager/Queue: the status just needs to be *readable* by
an unrelated process at an arbitrary later moment (a Streamlit session's
polling fragment isn't guaranteed to be alive, or even running in the same
thread, at the instant a worker writes an update), and a file on disk
survives that independently. It's also exactly the same on-disk-under-
session_dir() convention already used for the Emcee HDF5 sampler backend
(pages/fitting.py) — nothing new to clean up (services/storage.py's
purge_expired() already reclaims it).

Pure Python, no Streamlit import: this module is imported both by the
worker process (which has no Streamlit runtime at all) and by the
Streamlit-side poller (services/jobs.py).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def write_status(path: Path | str, **fields: Any) -> None:
    """Atomically overwrite the status file at `path` with `fields`, plus
    an `updated` timestamp (used by the heartbeat check in
    services/jobs.py to detect a worker that died without writing a final
    state). Atomic via write-then-rename so a poller never observes a
    half-written file.
    """
    path = Path(path)
    payload = {**fields, "updated": time.time()}
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_status(path: Path | str) -> dict[str, Any] | None:
    """Returns the last status written, or None if the file doesn't exist
    yet (job not started) or was caught mid-write (treated as "no update
    yet" rather than an error — the next poll picks up the completed
    write, since write_status() above never leaves a partial file at the
    final path)."""
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
