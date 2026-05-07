# pages/data.py
"""
Page "Data" – Chargement et filtrage des fichiers OIFITS.

Cette page ne contient QUE de la logique UI.
- Elle ne charge PAS directement les fichiers (→ services/data_service.py)
- Elle ne stocke PAS d'objets lourds dans session_state (→ chemins uniquement)
- Elle délègue les calculs à core/ et services/
"""
from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from services.data_service import get_oim, load_oifits, get_filtered_wavelengths, load_oifits_multi
from components.plots import safe_pyplot


def render() -> None:
    """Point d'entrée de la page Data, appelé depuis app.py."""
    _render_file_upload()

    if not st.session_state.loaded_files:
        st.info("Please load at least one OIFITS file.")
        return

    _render_filter_section()
    _render_observables()


# ── Section 1 : Upload ─────────────────────────────────────────────────────

def _render_file_upload() -> None:
    with st.expander("I. Load OIFITS data", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more OIFITS files",
            type=['fits', 'oifits'],
            accept_multiple_files=True,
        )
        if not uploaded_files:
            return

        for f in uploaded_files:
            if f.name in st.session_state.loaded_files:
                continue
            try:
                tmp_path = f"/tmp/{f.name}"
                with open(tmp_path, "wb") as fh:
                    fh.write(f.getbuffer())
                # ✅ On ne stocke QUE le chemin, pas l'objet oimData
                st.session_state.loaded_files[f.name] = tmp_path
                st.success(f"✓ {f.name} loaded")
            except Exception as exc:
                st.error(f"Error ({f.name}): {exc}")


# ── Section 2 : Filtrage spectral ─────────────────────────────────────────

def _render_filter_section() -> None:
    with st.expander("II. Data filtering and display", expanded=True):
        #st.write(f"load files :\n {st.session_state.loaded_files}")

        ####
        if "test_loaded_files" not in st.session_state : 
            st.session_state.test_loaded_files   = {}
            st.session_state.test_selected_files = []
            st.session_state.test_files_path = []

        test = st.multiselect("Select data to use", options=list(st.session_state.loaded_files.keys()))
        #st.write(f"multiselect files :\n {test}")

        test_filpath = ['/tmp/'+str(i) for i in test] 
        st.session_state.test_files_path = ['/tmp/'+str(i) for i in test] 

        #st.write(f"multiselect files path :\n\n {test_filpath}")

        st.session_state.test_selected_file = test

        ####



        #selected = st.selectbox(
        #    "Dataset to display",
        #    list(st.session_state.loaded_files.keys()),
        #)

        try : 
            st.session_state.selected_file = test[0]
            filepath = st.session_state.loaded_files[test[0]]
        except:
            pass

        # ── Paramètres de filtre ──────────────────────────────────────
        n_ranges = st.radio("Number of spectral ranges", [1, 2],
                            horizontal=True, key="n_wl_ranges")
        rc1, rc2 = st.columns([2, 3])

        with rc1:
            st.markdown("**Range 1**")
            c1, c2 = st.columns(2)
            with c1:
                wl1_min = st.number_input("λ min (µm)", value=2.9, step=0.1,
                                          format="%.2f", key="wl1_min")
            with c2:
                wl1_max = st.number_input("λ max (µm)", value=4.2, step=0.1,
                                          format="%.2f", key="wl1_max")

            if n_ranges == 2:
                st.markdown("**Range 2**")
                c3, c4 = st.columns(2)
                with c3:
                    wl2_min = st.number_input("λ min (µm)", value=4.45, step=0.1,
                                              format="%.2f", key="wl2_min")
                with c4:
                    wl2_max = st.number_input("λ max (µm)", value=5.0, step=0.1,
                                              format="%.2f", key="wl2_max")
            else:
                wl2_min = wl2_max = None

            st.markdown("##### Spectral binning")
            cb1, cb2 = st.columns(2)
            with cb1:
                st.markdown("**L band**")
                bin_L  = st.slider("Bin L", 1, 20, 1, key="bin_L")
                norm_L = st.toggle("Normalize σ (L)", value=False, key="norm_L")
            with cb2:
                st.markdown("**N band**")
                bin_N  = st.slider("Bin N", 1, 20, 1, key="bin_N")
                norm_N = st.toggle("Normalize σ (N)", value=False, key="norm_N")

            # ── Construction de l'expression de filtre ────────────────
            try:
                w1_lo = wl1_min * 1e-6
                w1_hi = wl1_max * 1e-6

                if n_ranges == 1:
                    expr = f"(EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})"
                else:
                    w2_lo = wl2_min * 1e-6
                    w2_hi = wl2_max * 1e-6
                    expr  = (
                        f"((EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})) & "
                        f"((EFF_WAVE<{w2_lo}) | (EFF_WAVE>{w2_hi}))"
                    )

                # Stocke les paramètres de filtre dans session_state
                st.session_state.filter_expr   = expr
                st.session_state.filter_bin_L  = bin_L
                st.session_state.filter_bin_N  = bin_N
                st.session_state.filter_norm_L = norm_L
                st.session_state.filter_norm_N = norm_N

                # ✅ Les longueurs d'onde filtrées sont cachées par data_service
                test_wls = [get_filtered_wavelengths(i, expr, bin_L, bin_N) for i in test_filpath]
                #st.write(f"filtered wls : \n\n {test_wls}")

                wls = get_filtered_wavelengths(filepath, expr, bin_L, bin_N)
                wls_arr = np.array(wls)

                if n_ranges == 2:
                    st.info(
                        f"After filtering: {len(wls)} points  |  "
                        f"Range 1: [{wl1_min:.2f}, {wl1_max:.2f}] µm  —  "
                        f"Range 2: [{wl2_min:.2f}, {wl2_max:.2f}] µm"
                    )
                else:
                    st.info(
                        f"After filtering: {len(wls)} points  |  "
                        f"λ ∈ [{wls_arr.min()*1e6:.3f}, {wls_arr.max()*1e6:.3f}] µm"
                    )

            except Exception as exc:
                st.warning(f"Cannot apply filter: {exc}")

        # ── UV coverage ───────────────────────────────────────────────
        with rc2:
            try:
                oim  = get_oim()
                data = _get_active_data_with_filter()
                fig  = plt.figure(figsize=(4,4))
                ax   = plt.subplot(projection='oimAxes')
                ax.uvplot(data, unit="cycle/mas", cunit="micron",
                          label="cmap on wavelength", lw=3, cmap="plasma")
                ax.set_title("Total UV coverage")
                safe_pyplot(st, fig)
            except Exception as exc:
                st.warning(f"UV plot error: {exc}")


# ── Section 3 : Observables ───────────────────────────────────────────────

def _render_observables() -> None:
    with st.expander("III. Observable visualization", expanded=True):



        try:
            data = _get_active_data_with_filter()
            fig, (ax1, ax2, ax3) = plt.subplots(
                ncols=3, figsize=(15, 4),
                subplot_kw={'projection': 'oimAxes'},
            )
            ax1.oiplot(data, "SPAFREQ", "VIS2DATA",
                    xunit="cycle/mas", color="byBaseline", errorbar=True)
            ax1.set_title("VIS2")
            ax1.legend(fontsize=8)

            ax2.oiplot(data, "EFF_WAVE", "T3PHI",
                    xunit="micron", color="byBaseline", errorbar=True)
            ax2.set_title("T3PHI")
            ax2.legend(fontsize=8)


            ax3.oiplot(data, "EFF_WAVE", "FLUXDATA",
                    xunit="micron", errorbar=True)
            ax3.set_title("FLUXDATA")
            ax3.legend(fontsize=8)

            plt.tight_layout()
            safe_pyplot(st, fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Observable plot error: {exc}")


# ── Helper interne ─────────────────────────────────────────────────────────

def _get_active_data_with_filter():
    """
    Retourne l'objet oimData actif avec le filtre appliqué.
    Utilise le cache de load_oifits() pour ne pas recharger le fichier.
    """
    oim      = get_oim()
    filepath = st.session_state.loaded_files.get(st.session_state.selected_file)

    #st.write(f"filepath : \n\n {filepath}")
    #st.write(f"test_selected : \n\n {st.session_state.test_selected_file}")

    if filepath is None:
        raise ValueError("No file selected.")

    data = load_oifits(filepath)

    data = load_oifits_multi(tuple(st.session_state.test_files_path ))
    #st.write(f"DATA : \n\n {data}")

    expr  = st.session_state.get('filter_expr', '')
    bin_L = st.session_state.get('filter_bin_L', 1)
    bin_N = st.session_state.get('filter_bin_N', 1)
    norm_L = st.session_state.get('filter_norm_L', False)
    norm_N = st.session_state.get('filter_norm_N', False)

    filters = []
    if expr:
        filters.append(oim.oimFlagWithExpressionFilter(expr=expr, keepOldFlag=False))
    filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_L, normalizeError=norm_L))
    filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_N, normalizeError=norm_N))
    data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True

    return data
