# pages/overview.py
"""
Page "Overview" – Présentation de l'application et workflow recommandé.
Page purement statique : aucune dépendance services/ ou core/, aucun
accès à session_state.MODEL/données — uniquement du texte, pour ne pas
alourdir le temps de chargement de cet onglet (rendu à chaque rerun,
comme toutes les pages via st.tabs()).
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("""
    Interactive interface for modelling optical interferometry data
    in **OIFITS** format, built on the *oimodeler* Python library —
    no Python code required to build models, fit them to your data,
    and export a reproducible script once you're done.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**🔭 Component Explorer**\n\n"
                "Browse every available geometric component and preview "
                "its image and visibility live.")
    with c2:
        st.info("**📂 Data**\n\n"
                "Load OIFITS files, apply spectral/baseline filters, and "
                "inspect VIS², T3PHI, FLUXDATA and the uv-coverage.")
    with c3:
        st.info("**⚙️ Modelling**\n\n"
                "Build a multi-component model, add wavelength/time "
                "interpolators or flux normalizations, save/import it.")
    with c4:
        st.info("**📐 Fitting**\n\n"
                "Random search, χ² minimization, grid search or Emcee "
                "MCMC — with a reproducible Python script for every run.")

    st.markdown("#### Recommended workflow")
    st.markdown(
        "1. **Component Explorer** → pick the component type(s) your source needs\n"
        "2. **Data** → load OIFITS file(s), filter and inspect them\n"
        "3. **Modelling** → build the model (Basic Model tab), add "
        "interpolators/normalizations if needed, save it\n"
        "4. **Fitting** → select the saved model, choose a method and run it\n"
        "5. **Fitting** → save the best-fit model, download the reproducible "
        "script and result files"
    )

    st.markdown("#### Page by page")

    with st.expander("🔭 Component Explorer"):
        st.markdown(
            "- Pick any oimodeler component class and see its live "
            "preview image and visibility-vs-baseline curve as you move "
            "its parameters.\n"
            "- Useful to build intuition before adding a component to an "
            "actual model."
        )

    with st.expander("📂 Data"):
        st.markdown(
            "- **Load OIFITS data**: upload one or more files.\n"
            "- **Data info & filters**: per-file spectral/baseline/telescope "
            "filters (wavelength range, binning, minimum error, ...).\n"
            "- **Diagnostic plots**: uv-coverage, VIS², T3PHI, FLUXDATA, "
            "plus a fully configurable **Custom plot** (any observable — "
            "including the uv-plane — vs. spatial frequency or wavelength, "
            "colored by file/baseline/configuration/array/instrument)."
        )

    with st.expander("⚙️ Modelling"):
        st.markdown(
            "- **Basic Model**: add/remove components, edit their "
            "parameters, preview the image or VIS²/T3PHI live, save the "
            "model under a name (or reset to start over).\n"
            "- **Import model**: load a model from a `.txt`/`.csv` file — "
            "including one exported by this app's own Fitting results "
            "(interpolators and normalizations round-trip too).\n"
            "- **Interpolators**: tie a parameter to a wavelength- or "
            "time-dependent `oimInterp` (any oimodeler interpolator "
            "class), with per-sub-parameter bounds for fitting.\n"
            "- **Normalization**: tie a parameter to `norm − sum(other "
            "components' parameters)` (`oimParamNorm`) — the standard way "
            "to normalize flux across components in a multi-component SED.\n"
            "- Both **Interpolators** and **Normalization** edit a draft: "
            "nothing changes until you click that tab's own \"Save model "
            "...\" button.\n"
            "- **Model summary**: χ² and VIS²/T3PHI of any saved model "
            "against the currently loaded data.\n"
            "- **Model management**: rename or delete saved models."
        )

    with st.expander("📐 Fitting"):
        st.markdown(
            "- **Random**: draws random parameter sets within their "
            "bounds and keeps the best χ²ᵣ — a fast, rough starting point.\n"
            "- **scipy χ² Minimization**: local gradient-based refinement "
            "from the model's current parameter values.\n"
            "- **Grid search**: explores a regular 1D/2D grid over chosen "
            "parameters and maps χ²ᵣ over it.\n"
            "- **Emcee**: full MCMC exploration (default method) — walkers/"
            "corner plots, refinable after the run (discard/thin/χ² "
            "threshold) without re-sampling.\n"
            "- Every method's results panel includes a **reproducible "
            "Python script** and a downloadable zip (best-fit parameters, "
            "figures, script, activity log)."
        )

    st.markdown("#### Useful links")
    st.markdown(
        "- oimodeler github: [github.com/oimodeler/oimodeler](https://github.com/oimodeler/oimodeler)\n"
        "- oimodeler documentation: [oimodeler.readthedocs.io](https://oimodeler.readthedocs.io/en/latest/)"
    )

    st.caption("version : α")
