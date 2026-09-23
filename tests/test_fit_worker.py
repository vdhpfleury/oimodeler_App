# tests/test_fit_worker.py
"""
End-to-end tests of core/fit_worker.py against the *real* installed
oimodeler (see tests/test_filter_registry.py for the same
`pytest.importorskip("oimodeler")` convention and the bundled tutorial
data file this reuses).

Runs `run_job()` in-process (not via multiprocessing.Process — that's
services/jobs.py's concern, covered separately and fast in
tests/test_jobs.py with a fake worker) so these stay fast while still
exercising the real picklability boundary this module exists to cross:
every `params` dict below is exactly the plain, JSON-shaped input
pages/fitting.py builds from `st.session_state.MODEL[...]["components"]`,
never a live oimodeler object — and every payload written to `result.pkl`
is checked to be plain data (numbers/arrays), confirming nothing
unpicklable ever tries to cross the boundary in the first place.
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pytest

oim = pytest.importorskip("oimodeler")

from pathlib import Path

import core.fit_worker as fit_worker
from core.fit_worker import (
    reconstruct_chi2,
    reconstruct_emcee,
    reconstruct_grid,
    run_job,
)
from core.component import make_comp_dict
from core.job_status import read_status
from core.registry import build_registry
from services import jobs

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "tutorial" / "Data" / "RealData" / "MATISSE" / "HD179218" / \
    "OiXP_HD179218_MATISSE_A0-B2-C1-D0_2019-03-24.fits"


@pytest.fixture(scope="module")
def registry():
    return build_registry(oim)


def _ud_comp_dict(registry, free_d_range=(0.1, 10.0)):
    d = make_comp_dict("oimUD", "c1", registry)
    d["free_params"] = ["d"]
    d["param_ranges"]["d"] = free_d_range
    d["initial_values"]["d"] = 2.0
    d["initial_values"]["f"] = 1.0
    return d


def _paths(tmp_path):
    return str(tmp_path / "status.json"), str(tmp_path / "result.pkl")


# ── Random search ───────────────────────────────────────────────────────

def test_random_job_is_picklable_and_reports_progress(tmp_path, registry):
    status_path, result_path = _paths(tmp_path)
    params = {
        "model_comps": [_ud_comp_dict(registry)],
        "file_paths": [str(DATA_FILE)],
        "filter_specs": [],
        "n_runs": 3,
        "seed": 1,
    }
    run_job("random", params, status_path, result_path)

    status = read_status(status_path)
    assert status["state"] == "done"
    assert status["progress"] == 1.0

    with open(result_path, "rb") as fh:
        payload = pickle.load(fh)
    assert set(payload) == {"best_params", "best_chi2", "history"}
    assert len(payload["history"]) == 3
    assert isinstance(payload["best_chi2"], float)
    # Every piece of the payload is plain data — nothing oimodeler-specific
    # (the whole point: this is what's allowed to cross the process
    # boundary back to the parent).
    pickle.dumps(payload)


# ── scipy chi2 minimization ─────────────────────────────────────────────

def test_chi2_job_and_reconstruction(tmp_path, registry):
    status_path, result_path = _paths(tmp_path)
    params = {
        "model_comps": [_ud_comp_dict(registry)],
        "file_paths": [str(DATA_FILE)],
        "filter_specs": [],
        "dtypes": ["VIS2DATA"],
    }
    run_job("chi2", params, status_path, result_path)
    assert read_status(status_path)["state"] == "done"

    with open(result_path, "rb") as fh:
        payload = pickle.load(fh)
    assert set(payload) == {"values", "errors", "chi2_final"}

    lmfit = reconstruct_chi2(oim, registry, params, payload)
    assert lmfit.simulator.chi2r == pytest.approx(payload["chi2_final"], rel=1e-6)
    fitted_d = list(lmfit.freeParams.values())[0].value
    assert fitted_d == pytest.approx(payload["values"][0])


def test_chi2_job_failure_surfaces_error_status(tmp_path, registry):
    status_path, result_path = _paths(tmp_path)
    params = {
        "model_comps": [],  # build_oim_model() -> None -> AttributeError inside oimFitterMinimize
        "file_paths": [str(DATA_FILE)],
        "filter_specs": [],
        "dtypes": ["VIS2DATA"],
    }
    run_job("chi2", params, status_path, result_path)
    status = read_status(status_path)
    assert status["state"] == "error"
    assert status["message"]
    assert not Path(result_path).exists()


# ── Grid search ──────────────────────────────────────────────────────────

def test_grid_job_and_reconstruction(tmp_path, registry):
    status_path, result_path = _paths(tmp_path)
    params = {
        "model_comps": [_ud_comp_dict(registry)],
        "file_paths": [str(DATA_FILE)],
        "filter_specs": [],
        "dtypes": ["VIS2DATA"],
        # oimModel.getFreeParameters() keys components as
        # "<index><name>_<class>_<param>" (e.g. "c1_UD_d"), not the bare
        # param name — see core/component.py's create_instance().
        "axes": [{"name": "c1_UD_d", "lo": 0.5, "hi": 2.5, "n": 3}],
    }
    run_job("grid", params, status_path, result_path)
    status = read_status(status_path)
    assert status["state"] == "done"

    with open(result_path, "rb") as fh:
        payload = pickle.load(fh)
    assert payload["chi2rMap"].shape == (3,)

    gfit = reconstruct_grid(oim, registry, params, payload)
    assert gfit.chi2rMap.shape == (3,)
    np.testing.assert_array_equal(gfit.chi2rMap, payload["chi2rMap"])
    assert gfit.simulator.chi2r == pytest.approx(float(np.min(payload["chi2rMap"])), rel=1e-6)


# ── Emcee ──────────────────────────────────────────────────────────────

def test_emcee_job_and_reconstruction(tmp_path, registry):
    status_path, result_path = _paths(tmp_path)
    sampler_path = tmp_path / "sampler.txt"
    params = {
        "model_comps": [_ud_comp_dict(registry)],
        "file_paths": [str(DATA_FILE)],
        "filter_specs": [],
        "dtypes": ["VIS2DATA"],
        "nwalkers": 4,
        "nsteps": 5,
        "init": "random",
        "sampler_path": str(sampler_path),
    }
    run_job("emcee", params, status_path, result_path)
    status = read_status(status_path)
    assert status["state"] == "done"
    assert sampler_path.exists()

    with open(result_path, "rb") as fh:
        payload = pickle.load(fh)
    assert "chi2_final" in payload

    emfit = reconstruct_emcee(oim, registry, params, payload)
    assert emfit.sampler.iteration == 5
    assert np.isfinite(emfit.simulator.chi2r)


# ── Full boundary: real multiprocessing.Process (services/jobs.py), not an
# in-process call — the actual deployment path pages/fitting.py uses ────────

def test_random_job_through_a_real_spawned_process(tmp_path, registry):
    """The one test in this file that actually crosses a real process
    boundary (spawn context, see services/jobs.py) end to end — everything
    else above calls run_job() in-process to stay fast. This is what
    confirms `params` really is picklable into a fresh interpreter and
    the result really does pickle back out through result.pkl.
    """
    params = {
        "model_comps": [_ud_comp_dict(registry)],
        "file_paths": [str(DATA_FILE)],
        "filter_specs": [],
        "n_runs": 3,
        "seed": 1,
    }
    handle = jobs.submit_fit_job(
        target=fit_worker.run_job, args=("random", params), job_dir=tmp_path,
    )
    deadline = time.time() + 60.0
    status = jobs.poll_job(handle)
    while status["state"] not in ("done", "error") and time.time() < deadline:
        time.sleep(0.1)
        status = jobs.poll_job(handle)

    assert status["state"] == "done", status
    payload = jobs.collect_result(handle)
    assert len(payload["history"]) == 3
    jobs.release_job(handle)
