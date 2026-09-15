# core/fitting.py
"""
Algorithme de recherche aléatoire (Random Search), et pilotage des
fitters oimodeler (Grid search / Emcee) via leur API publique afin d'en
exposer la progression (utilisé pour une barre de progression Streamlit,
passée en argument pour garder le module testable — aucun `import
streamlit` ici).
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

from core.component import ComponentConfig


def random_search(oim, data, component_configs: list[ComponentConfig],
                  n_runs: int = 100, seed: int | None = None,
                  wave_data=None,
                  progress_callback=None,
                  status_callback=None,
                  warning_callback=None) -> tuple:
    """
    Recherche aléatoire des meilleurs paramètres minimisant χ²ᵣ.

    Paramètres
    ----------
    oim               : module oimodeler
    data              : objet oimData filtré
    component_configs : liste de ComponentConfig
    n_runs            : nombre d'itérations
    seed              : graine aléatoire (None = aléatoire)
    wave_data         : longueurs d'onde pour les interpolateurs
    progress_callback : callable(float) pour la barre de progression [0,1]
    status_callback   : callable(str) pour les messages de statut
    warning_callback  : callable(str) pour les avertissements

    Retourne
    --------
    (best_params, best_chi2, history)
    """
    if seed is not None:
        np.random.seed(seed)

    best_chi2   = float('inf')
    best_params = None
    history     = []

    for run in range(n_runs):
        try:
            comps      = []
            run_params = {}
            for cfg in component_configs:
                rp = cfg.generate_random_params()
                comps.append(cfg.create_instance(oim, rp, wave_data=wave_data))
                run_params[cfg.name] = rp

            model = oim.oimModel(*comps)
            sim   = oim.oimSimulator(data=data, model=model)
            sim.compute(computeChi2=True, computeSimulatedData=False)
            chi2r = sim.chi2r

            history.append({'run': run + 1, 'chi2r': chi2r, 'params': run_params})

            if chi2r < best_chi2:
                best_chi2   = chi2r
                best_params = run_params
                if status_callback:
                    status_callback(
                        f"✓ Run {run+1}/{n_runs} – new best χ²ᵣ = {chi2r:.4f}"
                    )

        except Exception as e:
            if warning_callback:
                warning_callback(f"Run {run+1} skipped: {e}")

        if progress_callback:
            progress_callback((run + 1) / n_runs)

    return best_params, best_chi2, history


# ── Grid search progress ────────────────────────────────────────────────
# oimFitterRegularGrid.run() has no public per-step callback — the only
# per-iteration hook its `_run()` exposes is tqdm, wrapped around a plain
# `range(n)` (see oimodeler.oimFitter). We temporarily replace the `tqdm`
# symbol that module resolves with a thin subclass that also drives a UI
# progress_callback, instead of reimplementing oimFitterRegularGrid's loop
# by reaching into its internal attributes (gridSize/gridParams/grid/
# chi2rMap are not public API and could silently break, since oimodeler is
# pulled from HEAD with no version pin).
#
# The patch target is a module-global (oimodeler.oimFitter.tqdm), shared
# by every concurrent Streamlit session in this process — so the whole
# patch/run/restore window is serialized process-wide via this lock.
# Without it, two grid searches running in different session threads could
# each install their own patch mid-run and end up observing the other's
# patched class, reporting progress to the wrong session.
_grid_progress_lock = threading.Lock()


def run_grid_search_with_progress(gfit, progress_callback=None) -> None:
    """
    Runs an already-prepared oimFitterRegularGrid (gfit.prepare(...) must
    have been called already), calling progress_callback(fraction) as the
    grid is explored. Leaves `gfit` in the same state gfit.run() would
    have (chi2rMap, simulator, etc. — untouched, only tqdm is wrapped).
    """
    import tqdm as tqdm_pkg

    fitter_module = sys.modules[type(gfit).__module__]
    original_tqdm = fitter_module.tqdm

    class _ProgressTqdm(tqdm_pkg.tqdm):
        def update(self, n=1):
            advanced = super().update(n)
            if progress_callback is not None and self.total:
                progress_callback(min(self.n / self.total, 1.0))
            return advanced

        def close(self):
            super().close()
            if progress_callback is not None and self.total:
                progress_callback(min(self.n / self.total, 1.0))

    with _grid_progress_lock:
        fitter_module.tqdm = _ProgressTqdm
        try:
            gfit.run(progress=True)
        finally:
            fitter_module.tqdm = original_tqdm


# ── Emcee progress ───────────────────────────────────────────────────────
def run_emcee_with_progress(emfit, nsteps: int, progress_callback=None) -> None:
    """
    Runs an already-prepared oimFitterEmcee (emfit.prepare(...) must have
    been called already) for `nsteps` steps, calling
    progress_callback(fraction) as the chain advances (throttled to at
    most ~10 calls/second, matching tqdm's own default mininterval, so a
    large step count doesn't flood the Streamlit websocket with deltas).

    Drives emcee's own public generator API (EnsembleSampler.sample(),
    which yields a State after every step) instead of the blocking
    emfit.run(nsteps=..., progress=True) — emcee's run_mcmc() is itself
    implemented as `for state in self.sample(initial_state,
    iterations=nsteps, **kwargs): pass`, so this observes exactly the same
    public mechanism, just with our own callback in the loop body instead
    of emcee's console tqdm bar.

    Mirrors oimFitterEmcee._run()'s own initial-state resolution, then
    calls the same public emfit.getResults() _run() would have called, so
    `emfit` ends up in the same state emfit.run() would have left it in.
    """
    if emfit.sampler.iteration == 0:
        state = emfit.initialParams
    else:
        # Resuming a sampler that already has steps: this app always
        # builds a fresh oimFitterEmcee per run (see pages/fitting.py),
        # so this branch isn't reachable today. Fall back to the public
        # entry point rather than guessing at emcee's private resume
        # state (run_mcmc() resolves `initial_state=None` via its own
        # private `_previous_state`, which isn't part of emcee's public
        # API either).
        emfit.run(nsteps=nsteps, progress=False)
        return

    last_call = 0.0
    for i, _ in enumerate(
        emfit.sampler.sample(state, iterations=nsteps, store=True), start=1
    ):
        if progress_callback is not None:
            now = time.monotonic()
            if i == nsteps or now - last_call >= 0.1:
                progress_callback(i / nsteps)
                last_call = now

    emfit.getResults()
