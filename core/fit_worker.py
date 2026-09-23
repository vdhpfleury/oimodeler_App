# core/fit_worker.py
"""
Process-isolated fit execution.

This module has two jobs:

1. `run_job()` — the function services/jobs.py spawns as a
   `multiprocessing.Process` target. It runs in a *bare* subprocess with
   no Streamlit runtime at all (no st.session_state, no st.cache_resource,
   nothing) — only oimodeler/numpy/scipy. It rebuilds everything it needs
   (oimodeler module, component registry, oimData, oimModel) from plain,
   picklable inputs, because the live oimodeler objects the app already
   builds are NOT picklable and so cannot simply be handed to a
   `multiprocessing.Process` as-is: reproduced directly against the
   installed oimodeler — `oim.oimData(...)` holds an open astropy FITS
   file handle (`TypeError: cannot pickle '_io.BufferedReader' object`),
   and anything that carries a reference to it transitively fails the same
   way (`oimSimulator`, `oimFitterEmcee`, ...). Plain `oimModel`/component
   instances (no data attached) pickle fine, but aren't reused across the
   boundary here either, to keep exactly one code path.

2. `reconstruct_chi2()` / `reconstruct_grid()` / `reconstruct_emcee()` —
   called back on the *Streamlit* side (pages/fitting.py), once a job's
   status file reports `done`, to turn the worker's plain-data result
   payload (numbers/arrays only — never a live oimData/simulator/fitter)
   back into a real, fully-functional oimodeler fitter object, so
   everything downstream of `st.session_state.<method>_result` (tables,
   plots, "Refine results", the results zip) keeps working completely
   unchanged. This works because rebuilding data+model+fitter from the
   same inputs the worker used is deterministic and, on its own (without
   the expensive optimize/sample/grid loop), cheap — the fitter classes'
   own `prepare()` never does that expensive work, only `run()`/`_run()`
   does, which is never called again here. Emcee's case is the simplest:
   the worker's actual sampling result already lives entirely in the
   `samplerFile` HDF5 backend on disk (see pages/fitting.py's existing
   per-job sampler path convention) — reattaching a fresh
   `oimFitterEmcee` to that same file and calling `getResults()` is
   *exactly* what `run_emcee_with_progress()` itself does at the end of a
   run, just replayed from disk instead of from a live sampler object.

Random search needs neither: `core.fitting.random_search()` already
returns only plain dicts/floats/lists (`best_params`, `best_chi2`,
`history`), so its worker payload IS the final result — no reconstruction
step at all.

Pure Python, no Streamlit import (project convention — see core/fitting.py).
"""
from __future__ import annotations

import pickle
import traceback
from pathlib import Path
from typing import Any, Callable

from core.component import ComponentConfig
from core.fitting import (
    random_search,
    run_emcee_with_progress,
    run_grid_search_with_progress,
)
from core.job_status import write_status
from core.model_builder import build_oim_model
from core.registry import build_registry


def _load_oim():
    import oimodeler as oim  # noqa: PLC0415 — only ever imported inside a worker process
    return oim


def build_data(oim, file_paths: list[str], filter_specs: list[dict]):
    """Rebuilds a filtered oimData from plain inputs — the same shape
    services/data_service.py's get_active_data() produces, but
    Streamlit-free (no st.cache_resource, no st.session_state) so it can
    run both inside a worker process and, for reconstruction, on the
    Streamlit side without going through the cache layer (the point here
    is determinism from `params`, not caching).
    """
    data = oim.oimData(file_paths[0]) if len(file_paths) == 1 else oim.oimData(list(file_paths))
    filters = []
    for spec in filter_specs:
        filter_cls = getattr(oim, spec["filter_class"], None)
        if filter_cls is None:
            continue  # unknown to this oimodeler version — same tolerance as data_service.py
        kwargs = {k: v for k, v in spec.get("kwargs", {}).items() if v is not None}
        filters.append(filter_cls(**kwargs))
    if filters:
        data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True
    return data


def _build_configs(registry: dict, comp_dicts: list[dict]) -> list[ComponentConfig]:
    return [
        ComponentConfig(
            component_type=c['type'], registry=registry, name=c['name'],
            initial_values=c['initial_values'], param_ranges=c['param_ranges'],
            free_params=c['free_params'], interpolators=c.get('interpolators', {}),
            normalizations=c.get('normalizations', {}),
        )
        for c in comp_dicts
    ]


# ── Worker-side: one function per fit kind, all with the same shape
# (oim, registry, data, params, progress_cb) -> picklable payload dict ─────

def _run_random(oim, registry, data, params: dict, progress_cb) -> dict:
    configs = _build_configs(registry, params['model_comps'])
    best_params, best_chi2, history = random_search(
        oim, data, configs,
        n_runs=params['n_runs'], seed=params.get('seed'),
        progress_callback=progress_cb,
    )
    return {'best_params': best_params, 'best_chi2': best_chi2, 'history': history}


def _run_chi2(oim, registry, data, params: dict, progress_cb) -> dict:
    model = build_oim_model(oim, registry, params['model_comps'])
    lmfit = oim.oimFitterMinimize(data, model, dataTypes=params['dtypes'])
    lmfit.prepare()
    progress_cb(0.05)
    lmfit.run()
    progress_cb(1.0)
    values = [p.value for p in lmfit.freeParams.values()]
    errors = [p.error for p in lmfit.freeParams.values()]
    return {'values': values, 'errors': errors, 'chi2_final': float(lmfit.simulator.chi2r)}


def _run_grid(oim, registry, data, params: dict, progress_cb) -> dict:
    model = build_oim_model(oim, registry, params['model_comps'])
    free_params = model.getFreeParameters()
    axes = params['axes']
    gfit = oim.oimFitterRegularGrid(data, model, dataTypes=params['dtypes'])
    gfit.prepare(
        params=[free_params[a['name']] for a in axes],
        min=[a['lo'] for a in axes],
        max=[a['hi'] for a in axes],
        steps=[(a['hi'] - a['lo']) / (a['n'] - 1) for a in axes],
    )
    run_grid_search_with_progress(gfit, progress_callback=progress_cb)
    return {'chi2rMap': gfit.chi2rMap, 'chi2_final': float(gfit.simulator.chi2r)}


def _run_emcee(oim, registry, data, params: dict, progress_cb) -> dict:
    model = build_oim_model(oim, registry, params['model_comps'])
    emfit = oim.oimFitterEmcee(data, model, nwalkers=params['nwalkers'], dataTypes=params['dtypes'])
    emfit.prepare(init=params['init'], samplerFile=params['sampler_path'])
    run_emcee_with_progress(emfit, params['nsteps'], progress_callback=progress_cb)
    return {'chi2_final': float(emfit.simulator.chi2r)}


_RUNNERS: dict[str, Callable] = {
    'random': _run_random,
    'chi2':   _run_chi2,
    'grid':   _run_grid,
    'emcee':  _run_emcee,
}


def run_job(kind: str, params: dict, status_path: str, result_path: str) -> None:
    """Entry point run inside the worker process (see services/jobs.py's
    submit_fit_job()). Never returns a value across the process boundary —
    `multiprocessing.Process` discards return values, so the result is
    pickled to `result_path` instead, and the outcome (done/error) is
    written to `status_path` (core/job_status.py) for the parent to poll.
    """
    status_path = Path(status_path)
    result_path = Path(result_path)

    def progress_cb(v: float) -> None:
        write_status(status_path, state="running", progress=float(v))

    try:
        write_status(status_path, state="running", progress=0.0, message="Preparing model & data…")
        oim = _load_oim()
        registry = build_registry(oim)
        data = build_data(oim, params['file_paths'], params.get('filter_specs', []))
        runner = _RUNNERS[kind]
        result = runner(oim, registry, data, params, progress_cb)
        with open(result_path, "wb") as fh:
            pickle.dump(result, fh)
        write_status(status_path, state="done", progress=1.0, message="Complete.")
    except Exception as exc:
        # Only the plain message reaches the browser (pages/fitting.py's
        # st.error(f"... error: {exc}") pattern) — the full traceback goes
        # to the worker process' own stderr only, never to status.json
        # (see docs/security_audit_2026-09.md V12 — exception-text
        # information disclosure).
        write_status(status_path, state="error", progress=None,
                     message=str(exc) or type(exc).__name__)
        traceback.print_exc()


# ── Streamlit-side: cheap, deterministic reconstruction of a real fitter
# object from a completed job's plain payload — see module docstring ───────

def compute_initial(oim, registry, params: dict) -> tuple[Any, float]:
    """Returns a fresh (unfitted) `(model, chi2r)` pair built from `params`
    — used for the "before" side of a fit's before/after comparison.
    Deliberately recomputed here (from the job's own snapshot of
    file_paths/filter_specs/model_comps) rather than reusing whatever
    model/chi2 the Streamlit page might have computed before submitting
    the job: the two can only diverge if the user changes the data/model
    selection while the job is running in the background, in which case
    this — matching what the job itself actually ran against — is the
    correct one to show next to the result.
    """
    data = build_data(oim, params['file_paths'], params.get('filter_specs', []))
    model = build_oim_model(oim, registry, params['model_comps'])
    sim = oim.oimSimulator(data=data, model=model)
    sim.compute(computeChi2=True, computeSimulatedData=False)
    return model, float(sim.chi2r)


def reconstruct_chi2(oim, registry, params: dict, payload: dict[str, Any]):
    data = build_data(oim, params['file_paths'], params.get('filter_specs', []))
    model = build_oim_model(oim, registry, params['model_comps'])
    lmfit = oim.oimFitterMinimize(data, model, dataTypes=params['dtypes'])
    lmfit.prepare()
    for p, value, error in zip(lmfit.freeParams.values(), payload['values'], payload['errors']):
        p.value = value
        p.error = error
    lmfit.simulator.compute(
        computeChi2=True, computeSimulatedData=True,
        dataTypes=params['dtypes'], cprior=lmfit.cprior,
    )
    return lmfit


def reconstruct_grid(oim, registry, params: dict, payload: dict[str, Any]):
    data = build_data(oim, params['file_paths'], params.get('filter_specs', []))
    model = build_oim_model(oim, registry, params['model_comps'])
    free_params = model.getFreeParameters()
    axes = params['axes']
    gfit = oim.oimFitterRegularGrid(data, model, dataTypes=params['dtypes'])
    gfit.prepare(
        params=[free_params[a['name']] for a in axes],
        min=[a['lo'] for a in axes],
        max=[a['hi'] for a in axes],
        steps=[(a['hi'] - a['lo']) / (a['n'] - 1) for a in axes],
    )
    gfit.chi2rMap = payload['chi2rMap']
    gfit.getResults()
    return gfit


def reconstruct_emcee(oim, registry, params: dict, payload: dict[str, Any]):
    data = build_data(oim, params['file_paths'], params.get('filter_specs', []))
    model = build_oim_model(oim, registry, params['model_comps'])
    emfit = oim.oimFitterEmcee(data, model, nwalkers=params['nwalkers'], dataTypes=params['dtypes'])
    # Always "fixed" here, regardless of the run's own `params['init']`:
    # this call's `initialParams` is only ever used to seed a *new*
    # sampling run (._run()), which never happens again during
    # reconstruction (the chain is already complete, read back from
    # `samplerFile`'s HDFBackend). "random"/"gaussian" would otherwise
    # burn a draw from numpy's *global* RNG for a value nobody uses —
    # harmless to the fit, but an avoidable cross-session side effect
    # (the same V19 class of issue, just now happening on the
    # Streamlit-process side during reconstruction instead of inside a
    # fit) now that a single global RNG is shared again by every session's
    # main-thread reconstruction step, unlike the sampling itself, which is
    # fully process-isolated.
    emfit.prepare(init="fixed", samplerFile=params['sampler_path'])
    emfit.getResults()
    return emfit
