# pages/overview.py
"""
Page "Overview" – Présentation de l'application et workflow recommandé.
Page purement statique : aucune dépendance services/ ou core/.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("""
    Interactive interface for modelling optical interferometry data
    in **OIFITS** format, built on the *oimodeler* Python library.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🔬 Component Explorer**\n\n"
                "Interactively explore and configure each geometric component. "
                "Visualize the image in real time and generate the associated Python code.")
    with c2:
        st.info("**📂 Loading & Filtering**\n\n"
                "Import OIFITS files, apply spectral filters "
                "and visualize VIS², T3PHI, FLUXDATA.")
    with c3:
        st.info("**📐 Modelling & Fitting**\n\n"
                "Configure a multi-component model, run a random exploration "
                "then refine with χ² minimization or Emcee.")

    st.markdown("#### Recommended workflow")
    st.markdown(
        "1. **Component Explorer** → choose components\n"
        "2. **Modelling** → I. Load OIFITS data\n"
        "3. **Modelling** → II. Filter & visualize\n"
        "4. **Modelling** → III. Configure & save model\n"
        "5. **Modelling** → IV. Run fitting"
    )

    st.markdown("#### Useful links")
    st.markdown("""
- oimodeler github : <a href="https://github.com/oimodeler/oimodeler">here</a>
- oimodeler read the doc : <a href="https://oimodeler.readthedocs.io/en/latest/">here</a>
""", unsafe_allow_html=True)
