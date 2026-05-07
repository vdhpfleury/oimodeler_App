# core/fitting.py
"""
Algorithme de recherche aléatoire (Random Search).
Logique pure – Streamlit utilisé uniquement pour la progress bar,
passée en argument pour garder le module testable.
"""
from __future__ import annotations

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
