# components/param_editor.py
"""
Widget réutilisable pour l'édition des paramètres d'un composant.
Extrait de la logique UI qui était répétée dans l'original.
"""
from __future__ import annotations

import streamlit as st

from config.constants import DEFAULT_PARAM_RANGES, DEFAULT_PARAM_INIT


def render_param_editor(comp: dict) -> None:
    """
    Affiche les colonnes init / min / max / free pour chaque paramètre
    du composant et lit les valeurs depuis session_state.

    Modifie comp en place (initial_values, param_ranges, free_params).
    """
    n_cols     = min(len(comp['params']), 10)
    param_cols = st.columns(n_cols)

    for i, param in enumerate(comp['params']):
        lo_def, hi_def = DEFAULT_PARAM_RANGES.get(param, (0., 100.))
        init_def       = DEFAULT_PARAM_INIT.get(param, (lo_def + hi_def) / 2)
        cur_init       = comp['initial_values'].get(param, init_def)
        cur_lo, cur_hi = comp['param_ranges'].get(param, (lo_def, hi_def))
        cur_free       = param in comp['free_params']

        with param_cols[i % n_cols]:
            st.markdown(f"**{param}**")
            st.number_input("init", value=float(cur_init),
                            key=f"{comp['name']}_{param}_init", format="%.4g")
            st.number_input("min",  value=float(cur_lo),
                            key=f"{comp['name']}_{param}_min",  format="%.4g")
            st.number_input("max",  value=float(cur_hi),
                            key=f"{comp['name']}_{param}_max",  format="%.4g")
            st.checkbox("free", value=cur_free,
                        key=f"{comp['name']}_{param}_free")

    _read_widget_values(comp)


def _read_widget_values(comp: dict) -> None:
    """Lit les valeurs des widgets et les stocke dans le dict composant."""
    for param in comp['params']:
        k_init = f"{comp['name']}_{param}_init"
        k_min  = f"{comp['name']}_{param}_min"
        k_max  = f"{comp['name']}_{param}_max"
        k_free = f"{comp['name']}_{param}_free"

        if k_init in st.session_state:
            comp['initial_values'][param] = st.session_state[k_init]
        if k_min in st.session_state and k_max in st.session_state:
            comp['param_ranges'][param] = (
                st.session_state[k_min],
                st.session_state[k_max],
            )
        if k_free in st.session_state:
            is_free = st.session_state[k_free]
            if is_free and param not in comp['free_params']:
                comp['free_params'].append(param)
            elif not is_free and param in comp['free_params']:
                comp['free_params'].remove(param)


def read_all_widgets(components: list[dict]) -> None:
    """Lit les widgets pour tous les composants de la liste."""
    for comp in components:
        _read_widget_values(comp)
