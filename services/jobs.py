# services/jobs.py
"""
Background fit execution — process-isolated worker pool + a server-wide
concurrency cap (see docs/security_audit_2026-09.md V7 / §4.4).

Why `multiprocessing.Process` and not `threading.Thread`:
    A background *thread* would NOT fix the DoS this exists to close.
    numpy/scipy/emcee's inner loops are CPU-bound and hold the GIL almost
    continuously — a "background" thread running a fit still starves every
    other Streamlit session's own script-run thread in this same process,
    which is exactly V7 (see pages/fitting.py's old synchronous
    `random_search()`/`oimFitterEmcee.run()` calls, and the docstring of
    core/fitting.py before this module existed). Only a separate OS
    process gets the fit its own interpreter/GIL and actually frees the
    main process's CPU for other sessions.

Why a raw `multiprocessing.Process` and not `ProcessPoolExecutor`:
    A hung/runaway job must be reclaimable — `Future.result(timeout=...)`
    only raises `TimeoutError` client-side, it never kills the still-running
    task, so a stuck fit would occupy a pool worker forever and the
    concurrency cap below would silently shrink by one, permanently, per
    stuck job. Managing a raw `Process` per job means `poll_job()` can call
    `.terminate()` on exactly that job's process once its timeout elapses.

Why `spawn` and not `fork`:
    Streamlit's server is heavily multi-threaded (one ScriptRunner thread
    per session, plus its own internal threads). Forking a multi-threaded
    process is a well-known deadlock hazard — a lock held by a thread that
    doesn't exist in the (single-threaded-again) child can never be
    released. `spawn` starts a clean interpreter instead, at the cost of
    re-importing oimodeler per job (a few seconds) — negligible next to a
    fit's own runtime, and worth it for not risking an intermittent,
    load-dependent deadlock in production.

Progress crosses the process boundary via a small per-job JSON status file
(core/job_status.py), not a `multiprocessing.Manager`/`Queue` — see that
module's docstring. Concurrency is capped by simply refusing a new job
once MAX_CONCURRENT_FITS are already alive: no unbounded internal queue, no
silent indefinite wait — a user who hits the cap gets an immediate,
honest "server busy" message (Busy, below).

Nothing here knows what a "fit" actually is (no oimodeler import) — see
core/fit_worker.py for the oimodeler-specific worker entry point this
module's callers (pages/fitting.py) pass in as `target`. That split is
also what makes this module's own tests fast (tests/test_jobs.py uses a
tiny fake `target`, not a real fit).
"""
from __future__ import annotations

import logging
import multiprocessing
import pickle
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config.constants import (
    FIT_JOB_HEARTBEAT_SECONDS,
    FIT_JOB_TIMEOUT_SECONDS,
    FIT_SUBMIT_COOLDOWN_SECONDS,
    MAX_CONCURRENT_FITS,
)
from core.job_status import read_status

logger = logging.getLogger(__name__)

_CTX = multiprocessing.get_context("spawn")


class JobRejected(RuntimeError):
    """Base class for a submission refused without starting a job."""


class Busy(JobRejected):
    """The server-wide concurrency cap is currently full."""


class Cooldown(JobRejected):
    """This session submitted a job too recently."""


class NotPicklable(JobRejected):
    """`target`/`args` can't be sent to a spawned worker process.

    Surfaced as a clean, catchable error instead of letting
    `process.start()` raise deep inside `multiprocessing.reduction.dump()`
    — an uncaught exception there crashes the whole Streamlit script run
    (no st.error, no progress UI, just Streamlit's own generic "this app
    has encountered an error" page, with the real message redacted on
    Streamlit Cloud). The real exception is logged server-side (this
    module has no Streamlit import, so callers must do so) so the actual
    unpicklable type can be diagnosed from the deployment's own logs.
    """


@dataclass
class _ActiveJob:
    process: "multiprocessing.process.BaseProcess"
    status_path: Path
    started: float
    timeout: float
    heartbeat_timeout: float
    reaped: bool = False


_lock = threading.Lock()
_active: dict[str, _ActiveJob] = {}


def _reap_if_stale(job_id: str, job: _ActiveJob) -> None:
    """Terminates a job's process if it's exceeded its wall-clock timeout
    or gone quiet past its heartbeat window — called opportunistically
    from `_prune_active()` (not just from `poll_job()`) so an abandoned
    job (browser tab closed mid-fit, nobody left polling it) still gets
    reclaimed the next time *anyone* submits a new job, instead of
    occupying a concurrency slot forever.
    """
    if job.reaped or not job.process.is_alive():
        return
    now = time.time()
    status = read_status(job.status_path)
    last_update = status.get("updated") if status else job.started
    stale = (now - job.started > job.timeout) or (now - (last_update or job.started) > job.heartbeat_timeout)
    if stale:
        job.process.terminate()
        job.reaped = True


def _prune_active() -> None:
    """Reaps stale jobs, then drops any process that's no longer alive.
    Must be called with `_lock` held."""
    for job_id, job in list(_active.items()):
        _reap_if_stale(job_id, job)
    for job_id in [jid for jid, j in _active.items() if not j.process.is_alive()]:
        _active.pop(job_id, None)


def submit_fit_job(target: Callable, args: tuple, job_dir: Path, job_id: str | None = None,
                   timeout: float = FIT_JOB_TIMEOUT_SECONDS,
                   heartbeat_timeout: float = FIT_JOB_HEARTBEAT_SECONDS) -> dict[str, Any]:
    """Starts `target(*args, status_path, result_path)` in its own process,
    subject to the server-wide concurrency cap. `target` must be a
    top-level, importable, picklable callable (a `spawn`-context
    requirement) — see core/fit_worker.run_job for the one this app uses.

    Returns a job handle dict; store it in `st.session_state` and pass it
    to `poll_job()` / `collect_result()` / `release_job()` below.

    Raises Busy if MAX_CONCURRENT_FITS jobs are already running
    server-wide, and NotPicklable if `target`/`args` can't actually be
    sent to a spawned process. Never blocks waiting for a free slot (see
    module docstring) — callers should catch JobRejected and show its
    message.
    """
    job_id = job_id or uuid.uuid4().hex
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"
    result_path = job_dir / "result.pkl"
    full_args = (*args, str(status_path), str(result_path))

    # Fail fast, before touching the concurrency count, with a clear
    # message and the real cause in the server log — otherwise this
    # exact failure surfaces 20+ stack frames deep inside
    # multiprocessing.reduction.dump() during process.start() below,
    # where an uncaught exception crashes the whole Streamlit script run
    # instead of being one more thing pages/fitting.py can turn into an
    # st.error(). Doesn't perfectly replicate process.start()'s own
    # pickling (e.g. it never touches the Process object's authkey), but
    # catches the actual failure mode this exists for: something inside
    # `params` (a stray live object, a closure, ...) that pickle.dumps()
    # itself already rejects.
    try:
        pickle.dumps((target, full_args))
    except Exception as exc:
        logger.exception(
            "Fit job payload is not picklable (kind/target=%r) — "
            "see traceback above for the actual offending type.",
            getattr(target, "__qualname__", target),
        )
        raise NotPicklable(
            "This fit's configuration couldn't be sent to a background "
            "worker process. This is an internal error, not something "
            "wrong with your inputs — please report it."
        ) from exc

    with _lock:
        _prune_active()
        if len(_active) >= MAX_CONCURRENT_FITS:
            raise Busy(
                f"The server is already running the maximum of "
                f"{MAX_CONCURRENT_FITS} fits at once. Please try again "
                f"in a minute."
            )
        process = _CTX.Process(target=target, args=full_args, daemon=True)
        started = time.time()
        try:
            process.start()
        except Exception as exc:
            logger.exception("multiprocessing.Process.start() failed for a fit job.")
            raise NotPicklable(
                "Could not start a background worker process for this "
                "fit. This is an internal/deployment error, not something "
                "wrong with your inputs — please report it."
            ) from exc
        _active[job_id] = _ActiveJob(
            process=process, status_path=status_path, started=started,
            timeout=timeout, heartbeat_timeout=heartbeat_timeout,
        )

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "status_path": str(status_path),
        "result_path": str(result_path),
        "started": started,
        "timeout": timeout,
        "heartbeat_timeout": heartbeat_timeout,
    }


def check_cooldown(session_state, key: str = "_last_fit_submit",
                   cooldown: float = FIT_SUBMIT_COOLDOWN_SECONDS) -> None:
    """Raises Cooldown if this session submitted a fit less than
    `cooldown` seconds ago. A lightweight guard against double-clicks /
    accidental resubmits — MAX_CONCURRENT_FITS above is what actually
    protects the server; this only exists to avoid one user's own
    double-click quietly eating two of their own concurrency slots.
    Mutates `session_state[key]` as a side effect (records this
    submission), so call it exactly once per actual submission attempt.
    """
    last = session_state.get(key, 0.0)
    remaining = cooldown - (time.time() - last)
    if remaining > 0:
        raise Cooldown(f"Please wait {int(remaining) + 1}s before starting another fit.")
    session_state[key] = time.time()


def poll_job(handle: dict[str, Any]) -> dict[str, Any]:
    """Returns the latest status for a job handle from submit_fit_job():
    `{"state": "queued"|"running"|"done"|"error", "progress": float|None,
    "message": str}`.

    Also terminates the job's process if it has exceeded its timeout/
    heartbeat window (belt-and-braces alongside `_reap_if_stale()`'s own
    opportunistic sweep — a session actively polling its own job reclaims
    it immediately rather than waiting for someone else's submission).
    """
    job = _active.get(handle["job_id"])
    if job is not None:
        with _lock:
            _reap_if_stale(handle["job_id"], job)

    status = read_status(handle["status_path"])
    if status is None:
        return {"state": "queued", "progress": 0.0, "message": "Starting…"}

    if status.get("state") in ("queued", "running") and job is not None and job.reaped:
        return {
            "state": "error", "progress": status.get("progress"),
            "message": f"Fit timed out or went unresponsive and was stopped "
                       f"(no progress for over {int(job.heartbeat_timeout)}s, "
                       f"or running longer than {int(job.timeout)}s).",
        }

    if status.get("state") in ("queued", "running") and job is not None and not job.process.is_alive():
        # Process exited without ever writing state=done/error — killed by
        # something run_job()'s own try/except couldn't catch (OOM killer,
        # a segfault in a native dependency, etc.).
        return {
            "state": "error", "progress": status.get("progress"),
            "message": "The fit process terminated unexpectedly (it may "
                       "have run out of memory). Try reducing the step/"
                       "walker/grid count.",
        }

    return status


def collect_result(handle: dict[str, Any]) -> Any:
    """Unpickles a completed job's result payload. Only valid once
    poll_job() reports state == 'done'."""
    with open(handle["result_path"], "rb") as fh:
        return pickle.load(fh)


def release_job(handle: dict[str, Any]) -> None:
    """Drops the process bookkeeping for a job whose result has been
    consumed. Not strictly required for correctness (the next
    submit_fit_job() anywhere would prune it once its process exits on its
    own), but keeps the concurrency count accurate immediately instead of
    only at the next unrelated submission.
    """
    with _lock:
        _active.pop(handle["job_id"], None)


def active_count() -> int:
    """Current server-wide count of live fit jobs (queued+running are the
    same thing here — a job occupies a process, and thus a slot, from the
    moment it's submitted). Exposed for tests/diagnostics."""
    with _lock:
        _prune_active()
        return len(_active)
