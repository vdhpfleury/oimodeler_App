# tests/test_jobs.py
"""
Unit tests for services/jobs.py's generic job runner — deliberately
oimodeler-free (a tiny fake worker function stands in for
core/fit_worker.run_job) so these stay fast and don't need a real fit:
start a job, see it transition queued/running -> done, see progress
update, see a failure surface, see a hung job get reclaimed by its
timeout/heartbeat, see a crashed process surface as an error, and see the
concurrency cap reject a submission once it's full.

The fake worker (`_fake_worker` below) must stay a plain, module-level
function: `multiprocessing`'s `spawn` context (see services/jobs.py's
module docstring for why spawn, not fork) pickles a Process target by
reference (module + qualified name) and re-imports it in the child, so a
closure/lambda/local function would fail — the same constraint
core/fit_worker.run_job is written against for real fits.
"""
from __future__ import annotations

import os
import pickle
import time

import pytest

from services import jobs


def _fake_worker(behavior: str, hang_seconds: float, status_path: str, result_path: str) -> None:
    from core.job_status import write_status

    write_status(status_path, state="running", progress=0.0, message="starting")

    if behavior == "ok":
        time.sleep(0.05)
        write_status(status_path, state="running", progress=0.5, message="halfway")
        time.sleep(0.05)
        with open(result_path, "wb") as fh:
            pickle.dump({"answer": 42}, fh)
        write_status(status_path, state="done", progress=1.0, message="done")
    elif behavior == "fail":
        write_status(status_path, state="error", progress=None, message="boom")
    elif behavior == "hang":
        time.sleep(hang_seconds)
        # Only reached if not terminated first — marks a bug in the test itself.
        write_status(status_path, state="done", progress=1.0, message="should have been killed")
    elif behavior == "silent_hang":
        # Writes exactly one update, then goes quiet — exercises the
        # heartbeat check (as opposed to "hang", which exercises the
        # wall-clock timeout).
        time.sleep(hang_seconds)
    elif behavior == "crash":
        os._exit(1)  # simulate an OOM-kill/segfault: no final status is ever written
    else:
        raise ValueError(behavior)


def _submit(tmp_path, behavior, hang_seconds=0.0, job_id=None, **kwargs):
    job_dir = tmp_path / (job_id or behavior)
    return jobs.submit_fit_job(
        target=_fake_worker, args=(behavior, hang_seconds),
        job_dir=job_dir, job_id=job_id, **kwargs,
    )


def _poll_until_terminal(handle, timeout=10.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = jobs.poll_job(handle)
        if status["state"] in ("done", "error"):
            return status
        time.sleep(interval)
    raise AssertionError(f"job never reached a terminal state: {status}")


@pytest.fixture(autouse=True)
def _clear_active_jobs():
    """Guards against one test's jobs bleeding into another's concurrency
    count — active_count() itself prunes finished processes, but a test
    that fails before cleanup could otherwise leave a slot occupied."""
    yield
    with jobs._lock:
        for job in jobs._active.values():
            if job.process.is_alive():
                job.process.terminate()
        jobs._active.clear()


def test_submit_and_poll_to_completion(tmp_path):
    handle = _submit(tmp_path, "ok")
    status = _poll_until_terminal(handle)
    assert status["state"] == "done"
    assert jobs.collect_result(handle) == {"answer": 42}
    jobs.release_job(handle)


def test_progress_updates_are_observed(tmp_path):
    handle = _submit(tmp_path, "ok")
    seen = set()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        status = jobs.poll_job(handle)
        if status.get("progress") is not None:
            seen.add(status["progress"])
        if status["state"] == "done":
            break
        time.sleep(0.01)
    assert 0.5 in seen or 1.0 in seen  # at least one intermediate update was observed
    jobs.release_job(handle)


def test_failure_surfaces_as_error_state(tmp_path):
    handle = _submit(tmp_path, "fail")
    status = _poll_until_terminal(handle)
    assert status["state"] == "error"
    assert status["message"] == "boom"
    jobs.release_job(handle)


def test_crashed_process_surfaces_as_error(tmp_path):
    handle = _submit(tmp_path, "crash")
    status = _poll_until_terminal(handle)
    assert status["state"] == "error"
    assert "unexpectedly" in status["message"]
    jobs.release_job(handle)


def test_timeout_terminates_a_hung_job(tmp_path):
    handle = _submit(tmp_path, "hang", hang_seconds=30.0, timeout=0.2, heartbeat_timeout=30.0)
    status = _poll_until_terminal(handle, timeout=10.0)
    assert status["state"] == "error"
    assert "timed out" in status["message"] or "unresponsive" in status["message"]
    job = jobs._active[handle["job_id"]]
    job.process.join(timeout=5.0)  # .terminate() is async (SIGTERM) — give it a moment
    assert not job.process.is_alive()
    jobs.release_job(handle)


def test_heartbeat_detects_an_unresponsive_job(tmp_path):
    handle = _submit(
        tmp_path, "silent_hang", hang_seconds=30.0,
        timeout=30.0, heartbeat_timeout=0.2,
    )
    status = _poll_until_terminal(handle, timeout=10.0)
    assert status["state"] == "error"
    job = jobs._active[handle["job_id"]]
    job.process.join(timeout=5.0)
    assert not job.process.is_alive()
    jobs.release_job(handle)


def test_concurrency_cap_rejects_once_full(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_CONCURRENT_FITS", 2)
    handles = [
        _submit(tmp_path, "hang", hang_seconds=1.0, job_id=f"cap{i}")
        for i in range(2)
    ]
    with pytest.raises(jobs.Busy):
        _submit(tmp_path, "hang", hang_seconds=1.0, job_id="cap_over")

    for h in handles:
        _poll_until_terminal(h, timeout=10.0)
        jobs.release_job(h)


def test_cooldown_rejects_a_quick_resubmit():
    session_state: dict = {}
    jobs.check_cooldown(session_state, cooldown=1.0)
    with pytest.raises(jobs.Cooldown):
        jobs.check_cooldown(session_state, cooldown=1.0)


def test_cooldown_allows_after_it_elapses():
    session_state: dict = {"_last_fit_submit": time.time() - 100}
    jobs.check_cooldown(session_state, cooldown=1.0)  # must not raise
