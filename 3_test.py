# app.py
"""
Point d'entrée OIModeler.

Responsabilités de ce fichier (et seulement celles-ci) :
1. st.set_page_config()         ← doit être le PREMIER appel Streamlit
2. init_session_state()         ← initialise toutes les clés de session
3. Vérification oimodeler       ← erreur explicite si absent
4. Rendu de la navigation       ← tabs ou st.navigation()

Tout le reste est délégué aux pages/ et aux couches inférieures.
"""
# app.py
import sys
from pathlib import Path

# Ajoute le dossier racine de l'app au sys.path
sys.path.insert(0, str(Path(__file__).parent))


import matplotlib.pyplot as plt
import streamlit as st

# Resets pyplot's global figure registry at the start of every rerun —
# st.tabs() executes ALL tabs' render() every rerun (only the DOM display
# of inactive tabs is hidden, their Python code still runs), so a figure
# left open by one page (e.g. an exception raised between plt.subplots()
# and the matching plt.close()) would otherwise silently become "the
# current figure" for an unrelated plt.colorbar()/plt.figure() call on a
# later rerun — the concrete cause of a reported
# "Adding colorbar to a different Figure" warning and, from the same
# stale-figure-reference family, Streamlit's MediaFileStorageError.
plt.close('all')

# ── 1. Configuration de la page (DOIT être le premier appel Streamlit) ────
st.set_page_config(
    page_title="OIModeler",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Initialisation centralisée du session_state ────────────────────────
from services.session import init_session_state  # noqa: E402
init_session_state()

# ── 3. Vérification de oimodeler (lazy import via service) ────────────────
try:
    from services.data_service import get_oim
    get_oim()  # déclenche le chargement une seule fois
except ImportError:
    st.error("oimodeler is not installed. Install it with: pip install oimodeler")
    st.stop()

# ── 4. Navigation / tabs ──────────────────────────────────────────────────
#
# Option A : tabs dans une seule page (comportement actuel)
# Option B : pages/ avec st.navigation() (recommandé pour les grandes apps)
#
# Option A conservée ici pour compatibilité avec l'existant :

from pages.overview    import render as render_overview     # noqa: E402
from pages.explorer    import render as render_explorer     # noqa: E402
from pages.data        import render as render_data         # noqa: E402
from pages.modelling   import render as render_modelling    # noqa: E402
from pages.fitting     import render as render_fitting      # noqa: E402


st.image("./images/logo.png")

# Navigation par onglets rendue via st.segmented_control plutôt que
# st.tabs() : st.tabs() exécute le code Python des 5 onglets à CHAQUE
# rerun (seul l'affichage DOM des onglets inactifs est masqué côté
# frontend), ce qui provoquait un recalcul systématique des 5 pages —
# y compris figures matplotlib et chargements de données — à chaque
# interaction, même dans un onglet non consulté. Avec
# segmented_control, st.session_state.active_tab sélectionne l'onglet
# et seule la fonction render() correspondante est appelée.
_TAB_LABELS = [
    "📋 Overview",
    "🔬 Component Explorer",
    "📂 Data",
    "⚙️ Modelling",
    "📐 Fitting",
]
_TAB_RENDERERS = {
    "📋 Overview":            render_overview,
    "🔬 Component Explorer":  render_explorer,
    "📂 Data":                render_data,
    "⚙️ Modelling":           render_modelling,
    "📐 Fitting":              render_fitting,
}

# Pas de `default=` : la clé 'active_tab' est déjà initialisée par
# init_session_state(), st.segmented_control la lit directement via `key`.
active_tab = st.segmented_control(
    "Navigation",
    options=_TAB_LABELS,
    label_visibility="collapsed",
    key="active_tab",
)
# segmented_control renvoie None si l'utilisateur déselectionne l'onglet
# actif (deuxième clic dessus) : on retombe sur "Overview" plutôt que de
# rendre une page vide, et on resynchronise la session_state pour le
# prochain rerun.
if active_tab is None:
    active_tab = "📋 Overview"
    st.session_state.active_tab = active_tab

_TAB_RENDERERS[active_tab]()

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Optical interferometry modelling app · Built on <em>oimodeler</em>"
    "<br>"
    "version : α "
    "</div>",
    unsafe_allow_html=True,
)