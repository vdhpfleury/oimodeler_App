# services/session.py
"""
Initialisation centralisée du session_state.

Appelée UNE SEULE FOIS dans app.py avant le rendu de toute page.
Toutes les pages importent ce module pour accéder aux clés de session.

Bonnes pratiques appliquées :
- Les gros objets (oimData, oimModel) ne sont PAS stockés dans session_state.
  On y stocke uniquement des chemins de fichiers et des dicts légers.
- copy.deepcopy() sur les valeurs par défaut pour éviter les références partagées.
"""
from __future__ import annotations

import copy
import streamlit as st

# ── Clés de session et leurs valeurs par défaut ────────────────────────────
# ⚠️  NE JAMAIS stocker ici :
#     - des objets oimData / oimModel  (→ utiliser services/data_service.py)
#     - des DataFrames de plusieurs GB (→ stocker le chemin, recharger via cache)

_DEFAULTS: dict = {
    # Composants en cours de configuration (dicts légers)
    'components':           [],
    'active_comp_name':     None,

    # Bibliothèque de modèles sauvegardés (dicts sérialisables uniquement)
    'MODEL':                {},

    # Fichiers OIFITS chargés : { nom_fichier: chemin_tmp }
    # L'objet oimData réel est dans services/data_service.load_oifits()
    'loaded_files':         {},   # str → str  (nom → chemin /tmp/...)
    'selected_file':        None, # nom du fichier actif (compat. modelling/fitting)
    'selected_files':       [],   # liste des noms sélectionnés dans l'étape II

    # Paramètres de filtrage spectral actifs
    'filter_expr':          '',
    'filter_bin_L':         1,
    'filter_bin_N':         1,
    'filter_norm_L':        False,
    'filter_norm_N':        False,

    # Résultats d'optimisation (Random search)
    'optimization_done':    False,
    'best_params':          None,
    'best_chi2':            None,
    'history':              [],
    # ⚠️ best_model stocke le dict de composants, PAS l'objet oimModel
    'best_model_comps':     None,

    # Résultats χ² (stocke uniquement les métriques et dicts, pas les objets)
    'chi2_result':          None,

    # Résultats Emcee (idem)
    'emcee_result':         None,
}


def init_session_state() -> None:
    """
    Initialise toutes les clés de session_state avec leurs valeurs par défaut.
    Idempotente : n'écrase pas les valeurs déjà présentes.
    Doit être appelée en tout premier dans app.py, avant st.set_page_config.

    Usage
    -----
    # app.py
    from services.session import init_session_state
    init_session_state()
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            # deepcopy pour éviter que deux sessions partagent la même liste/dict
            st.session_state[key] = copy.deepcopy(default)


def reset_session_state() -> None:
    """
    Réinitialise toutes les clés à leurs valeurs par défaut.
    Utile pour un bouton "Reset" dans l'UI.
    """
    for key, default in _DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(default)