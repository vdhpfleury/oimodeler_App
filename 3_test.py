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
#
# NOTE — "only render the active tab" was tried here and reverted:
# 1. st.segmented_control + rendering only the selected page: Streamlit
#    deletes a widget's session_state entry entirely whenever that widget
#    isn't instantiated on a given rerun (confirmed via isolated repro) —
#    switching tabs silently wiped every keyed widget's value (selected
#    files, applied filters, ...) on the pages left un-rendered.
# 2. st.tabs() + st.fragment per tab (keeps every tab "mounted", so no
#    state eviction): fixes #1, but st.tabs() itself never triggers a
#    rerun on switch (pure client-side CSS toggle) — a fragment only
#    reruns from an interaction inside it, so a page reading state
#    written by ANOTHER tab's fragment (e.g. Fitting reading a model just
#    saved on Modelling) can show stale content until the user interacts
#    with something on that page. Confirmed live: switching straight to
#    Fitting after saving a model on Modelling still showed "No model
#    saved" — this app's tabs are too cross-dependent (shared
#    MODEL/loaded_files/applied_filters) for a naive per-tab fragment.
# A correct version would need each page's render() split into a cheap
# "gate" part (always run) and an expensive "compute" part (fragment-
# scoped) — real surgery across every page, not attempted here.
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

tab_home, tab_visu, tab_data, tab_model, tab_fit = st.tabs([
    "📋 Overview",
    "🔬 Component Explorer",
    "📂 Data",
    "⚙️ Modelling",
    "📐 Fitting",
])

with tab_home:
    render_overview()

with tab_visu:
    render_explorer()

with tab_data:
    render_data()

with tab_model:
    render_modelling()

with tab_fit:
    render_fitting()

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
