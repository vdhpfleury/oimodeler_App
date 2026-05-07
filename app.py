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


import streamlit as st

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
