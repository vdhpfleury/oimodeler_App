# core/component.py
"""
Classe ComponentConfig – logique métier pure, sans dépendance Streamlit.
Testable unitairement de manière indépendante.
"""
from __future__ import annotations

import numpy as np

from config.constants import DEFAULT_PARAM_RANGES, DEFAULT_PARAM_INIT


class ComponentConfig:
    """Encapsule la configuration d'un composant oimodeler."""

    def __init__(
        self,
        component_type: str,
        registry: dict,
        name: str | None = None,
        initial_values: dict | None = None,
        param_ranges: dict | None = None,
        free_params: list | None = None,
        interpolators: dict | None = None,
    ):
        if component_type not in registry:
            raise ValueError(f"Unknown component type: {component_type}")

        self.component_type  = component_type
        self.name            = name or component_type
        self.component_class = registry[component_type]['class']
        self.param_names     = registry[component_type]['params']
        self.initial_values  = initial_values or {}
        self.interpolators   = interpolators or {}
        self.param_ranges = {
            p: (
                param_ranges.get(p)
                if param_ranges and p in param_ranges
                else DEFAULT_PARAM_RANGES.get(p, (0., 100.))
            )
            for p in self.param_names
        }
        self.free_params = (
            free_params if free_params is not None
            else [p for p in self.param_names if p not in ('x', 'y')]
        )

    # ------------------------------------------------------------------
    def _full_params(self, override: dict | None = None) -> dict:
        full = {
            p: self.initial_values.get(
                p,
                0. if p in ('x', 'y') else
                0.5 if p == 'f' else
                sum(self.param_ranges[p]) / 2,
            )
            for p in self.param_names
        }
        full.update(override or {})
        return full

    def create_instance(self, oim, param_values: dict | None = None,
                        wave_data: np.ndarray | None = None):
        """
        Instancie le composant oimodeler.

        Paramètres
        ----------
        oim : module oimodeler (passé en argument pour éviter l'import global)
        param_values : dict optionnel de surcharge des valeurs initiales
        wave_data : longueurs d'onde pour les interpolateurs blackbody
        """
        full = self._full_params(param_values)

        for p, cfg in self.interpolators.items():
            if not cfg.get('enabled', False):
                continue
            if cfg.get('type') == 'blackbody':
                wl = wave_data if wave_data is not None else np.linspace(1e-6, 5e-6, 50)
                full[p] = oim.oimInterp('starWl', temp=cfg['temp'],
                                        dist=cfg['dist'], lum=cfg['lum'], wl=wl)
            else:
                full[p] = oim.oimInterp(
                    cfg['var'],
                    **{cfg['var']: cfg['wl']},
                    values=cfg['values'],
                )

        instance = self.component_class(**full)

        for param_name, param_obj in instance.params.items():
            short = param_name.split('_')[-1]
            if short in self.param_names:
                param_obj.free = short in self.free_params
                lo, hi = self.param_ranges.get(short, (None, None))
                param_obj.min = lo
                param_obj.max = hi

        return instance

    def generate_random_params(self) -> dict:
        return {
            p: np.random.uniform(*self.param_ranges[p])
            for p in self.free_params
            if not (p in self.interpolators and self.interpolators[p].get('enabled', False))
        }


# ── Helpers sur les dicts de composants (session_state) ──────────────────

def make_comp_dict(comp_type: str, comp_name: str, registry: dict) -> dict:
    """Crée un dict de composant initialisé avec les valeurs par défaut."""
    params = registry[comp_type]['params']
    return {
        'type':           comp_type,
        'name':           comp_name,
        'params':         params.copy(),
        'initial_values': {p: DEFAULT_PARAM_INIT.get(p, 0.) for p in params},
        'param_ranges':   {p: DEFAULT_PARAM_RANGES.get(p, (0., 100.)) for p in params},
        'free_params':    [p for p in params if p not in ('x', 'y')],
        'interpolators':  {},
    }


def get_comp_by_name(components: list[dict], name: str) -> dict | None:
    """Cherche un composant par son nom dans la liste."""
    return next((c for c in components if c['name'] == name), None)
