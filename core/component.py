# core/component.py
"""
Classe ComponentConfig – logique métier pure, sans dépendance Streamlit.
Testable unitairement de manière indépendante.
"""
from __future__ import annotations

import numpy as np

from config.constants import DEFAULT_PARAM_RANGES, DEFAULT_PARAM_INIT
from core.interp_registry import get_full_layout


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
        normalizations: dict | None = None,
    ):
        if component_type not in registry:
            raise ValueError(f"Unknown component type: {component_type}")

        self.component_type  = component_type
        self.name            = name or component_type
        self.component_class = registry[component_type]['class']
        self.param_names     = registry[component_type]['params']
        self.initial_values  = initial_values or {}
        self.interpolators   = interpolators or {}
        # Normalizations (oim.oimParamNorm) are applied at the MODEL level
        # (core/model_builder.py's build_oim_model()), once every
        # component instance exists — they reference OTHER components'
        # already-built parameter objects directly, unlike interpolators
        # which are self-contained per component. Stored here only so
        # create_instance()'s callers (build_oim_model) can read it back
        # per component; create_instance() itself never touches it.
        self.normalizations  = normalizations or {}
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

    def create_instance(self, oim, param_values: dict | None = None):
        """
        Instancie le composant oimodeler.

        Paramètres
        ----------
        oim : module oimodeler (passé en argument pour éviter l'import global)
        param_values : dict optionnel de surcharge des valeurs initiales
        """
        full = self._full_params(param_values)
        if 'dim' in full:
            # 'dim' is an integer pixel/grid-resolution parameter for several
            # oimodeler component classes (e.g. oimTempGrad, oimExpRing,
            # oimInnerRim); the UI's generic param editor stores all values
            # as float, which numpy rejects (e.g. np.linspace(..., dim)).
            full['dim'] = int(round(full['dim']))

        # Each entry is {'enabled': bool, 'macro': <oimInterp macro name>,
        # 'kwargs': {...}} — see core/interp_registry.py. `kwargs` is
        # already unit-converted (wl in metres, not µm) and validated by
        # the UI/registry before it lands here, so it can be passed
        # straight through to oim.oimInterp(). oimodeler swaps this
        # wrapper for a real oimParamInterpolator-derived instance the
        # first time the component reads the parameter (see
        # oimComponent._eval: `value.type(self.params[key], **value.kwargs)`).
        for p, cfg in self.interpolators.items():
            if not cfg.get('enabled', False):
                continue
            full[p] = oim.oimInterp(cfg['macro'], **cfg['kwargs'])

        instance = self.component_class(**full)

        for param_name, param_obj in instance.params.items():
            short = param_name.split('_')[-1]
            if short not in self.param_names:
                continue
            if self.normalizations.get(short, {}).get('enabled', False):
                # Applied at the model level instead (build_oim_model()):
                # oim.oimParamNorm references OTHER components' already-
                # built parameter objects, which don't exist yet here —
                # whatever free/min/max is set below would just be
                # discarded the moment that replacement happens.
                continue
            free = short in self.free_params
            lo, hi = self.param_ranges.get(short, (None, None))

            if isinstance(param_obj, oim.oimParamInterpolator):
                # An interpolated parameter isn't a plain oimParam anymore
                # (it's replaced by e.g. oimParamInterpolatorWl) — setting
                # .free/.min/.max directly on IT is a no-op for fitting:
                # oimodeler enumerates its OWN sub-parameters (.params,
                # one oimParam per keyframe/Gaussian/coefficient/...) as
                # the actual free dimensions. Those mix different physical
                # kinds (e.g. GaussWl's are [x0, fwhm, val0, value] —
                # x0/fwhm are wavelengths, val0/value share the
                # interpolated parameter's own unit), so a single shared
                # (free, min, max) for all of them is wrong; each needs
                # its own bounds, entered per sub-parameter in the
                # Interpolators tab and stored in this interpolator's
                # cfg['bounds'] = {kwarg_name: [{'free','min','max'}, ...]}
                # (see core/interp_registry.py's get_full_layout(), which
                # also marks the handful of sub-parameters oimodeler
                # itself hardcodes free=False for — those are skipped
                # here, left at oimodeler's own value, never a UI concern).
                cfg = self.interpolators[short]
                bounds = cfg.get('bounds', {})
                sub_params = param_obj.params
                idx = 0
                for kwarg_name, count, controllable in get_full_layout(cfg['macro'], cfg['kwargs']):
                    entries = bounds.get(kwarg_name, [])
                    for i in range(count):
                        if controllable and idx < len(sub_params) and i < len(entries):
                            b = entries[i]
                            sub_params[idx].free = b['free']
                            sub_params[idx].min = b['min']
                            sub_params[idx].max = b['max']
                        idx += 1
            else:
                param_obj.free = free
                param_obj.min = lo
                param_obj.max = hi

        return instance

    def generate_random_params(self) -> dict:
        return {
            p: np.random.uniform(*self.param_ranges[p])
            for p in self.free_params
            if not (p in self.interpolators and self.interpolators[p].get('enabled', False))
            and not (p in self.normalizations and self.normalizations[p].get('enabled', False))
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
        'normalizations': {},
    }


def get_comp_by_name(components: list[dict], name: str) -> dict | None:
    """Cherche un composant par son nom dans la liste."""
    return next((c for c in components if c['name'] == name), None)
