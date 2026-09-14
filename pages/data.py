# pages/data.py
"""
Page "Data" – Chargement et filtrage des fichiers OIFITS.

Cette page ne contient QUE de la logique UI.
- Elle ne charge PAS directement les fichiers (→ services/data_service.py)
- Elle ne stocke PAS d'objets lourds dans session_state (→ chemins uniquement)
- Elle délègue les calculs à core/ et services/

Security notes (docs/security_audit_2026-09.md):
- Uploads go through services/storage.store() (V1: no path built from a
  client filename; V3: session-scoped directory; V8: quotas + purge).
- File selection is always filtered against st.session_state.loaded_files
  before use — an allowlist, never a reconstructed path (V2).
- The spectral filter expression that reaches oimodeler's eval() sink is
  validated by core.validation.filter_expression() (V5), and the
  wavelength bounds feeding it are clamped (V4).
"""
from __future__ import annotations

import logging

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from services.data_service import (
    get_oim, get_filtered_wavelengths, load_oifits_multi, build_data_type_filters,
)
from services.storage import store, resolve_selected_paths
from core.validation import num, choice, filter_expression, InvalidInput
from components.plots import safe_pyplot
from config.constants import FITTABLE_DATA_TYPES

logger = logging.getLogger(__name__)


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
                path = store(f)
                # ✅ On ne stocke QUE le chemin (déjà assaini par storage.store),
                #    pas l'objet oimData
                st.session_state.loaded_files[f.name] = str(path)
                st.success(f"✓ {f.name} loaded")
            except ValueError as exc:
                # Rejection reason is already a safe, user-facing message
                # (bad name/extension, wrong content, quota exceeded).
                st.error(str(exc))
            except Exception:
                logger.exception("Upload failed for %s", f.name)
                st.error(f"Could not load {f.name}. Please try again.")


# ── Section 2 : Filtrage spectral ─────────────────────────────────────────

def _render_filter_section() -> None:
    with st.expander("II. Data filtering and display", expanded=True):
        raw_selected = st.multiselect(
            "Select data to use",
            options=list(st.session_state.loaded_files.keys()),
        )
        # Server-side allowlist guard (V2): multiselect returns an unknown
        # client value as-is instead of raising, so we drop anything that
        # isn't actually a key of loaded_files rather than trusting it.
        selected = [n for n in raw_selected if n in st.session_state.loaded_files]
        st.session_state.selected_files = selected

        try:
            st.session_state.selected_file = selected[0]
        except IndexError:
            # No selection yet: keep the previous value untouched — but
            # if nothing was ever selected this stays None (see session.py).
            pass

        filepath = st.session_state.loaded_files.get(st.session_state.selected_file)

        # ── Sélection des types de données par fichier ───────────────
        # Certains fichiers n'ont que V2, d'autres que la phase de clôture,
        # etc. — ceci permet de garder un fichier tout en n'utilisant que
        # certains de ses observables (voir oim.oimKeepDataTypeFilter).
        if selected:
            with st.expander("Data types to use per file", expanded=False):
                for fname in selected:
                    default = st.session_state.file_dtypes.get(fname, FITTABLE_DATA_TYPES)
                    chosen_raw = st.multiselect(
                        f"Data types — {fname}", FITTABLE_DATA_TYPES,
                        default=default, key=f"dtypes_sel_{fname}",
                    )
                    # multiselect can echo back an unrecognized client value
                    # as-is (V4) — drop anything outside the known types.
                    chosen = [t for t in chosen_raw if t in FITTABLE_DATA_TYPES]
                    st.session_state.file_dtypes[fname] = chosen

        # ── Paramètres de filtre ──────────────────────────────────────
        n_ranges_raw = st.radio("Number of spectral ranges", [1, 2],
                            horizontal=True, key="n_wl_ranges")
        n_ranges = choice(n_ranges_raw, (1, 2), "Number of spectral ranges")
        rc1, rc2 = st.columns([2, 3])

        with rc1:
            st.markdown("**Range 1**")
            c1, c2 = st.columns(2)
            with c1:
                wl1_min_raw = st.number_input("λ min (µm)", value=2.9, step=0.1,
                                          format="%.2f", key="wl1_min")
            with c2:
                wl1_max_raw = st.number_input("λ max (µm)", value=4.2, step=0.1,
                                          format="%.2f", key="wl1_max")

            if n_ranges == 2:
                st.markdown("**Range 2**")
                c3, c4 = st.columns(2)
                with c3:
                    wl2_min_raw = st.number_input("λ min (µm)", value=4.45, step=0.1,
                                              format="%.2f", key="wl2_min")
                with c4:
                    wl2_max_raw = st.number_input("λ max (µm)", value=5.0, step=0.1,
                                              format="%.2f", key="wl2_max")
            else:
                wl2_min_raw = wl2_max_raw = None

            st.markdown("##### Spectral binning")
            cb1, cb2 = st.columns(2)
            with cb1:
                st.markdown("**L band**")
                bin_L_raw = st.slider("Bin L", 1, 20, 1, key="bin_L")
                norm_L = st.toggle("Normalize σ (L)", value=False, key="norm_L")
            with cb2:
                st.markdown("**N band**")
                bin_N_raw = st.slider("Bin N", 1, 20, 1, key="bin_N")
                norm_N = st.toggle("Normalize σ (N)", value=False, key="norm_N")

            # ── Construction de l'expression de filtre ────────────────
            try:
                # Widget bounds (min_value/max_value/slider range) are
                # cosmetic only — re-validate every value server-side (V4).
                wl1_min = num(wl1_min_raw, 0.1, 30.0, "λ min (range 1)")
                wl1_max = num(wl1_max_raw, 0.1, 30.0, "λ max (range 1)")
                if wl1_min >= wl1_max:
                    raise InvalidInput("λ min must be smaller than λ max (range 1).")
                bin_L = num(bin_L_raw, 1, 20, "Bin L", integer=True)
                bin_N = num(bin_N_raw, 1, 20, "Bin N", integer=True)

                w1_lo = wl1_min * 1e-6
                w1_hi = wl1_max * 1e-6

                if n_ranges == 1:
                    expr = f"(EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})"
                else:
                    wl2_min = num(wl2_min_raw, 0.1, 30.0, "λ min (range 2)")
                    wl2_max = num(wl2_max_raw, 0.1, 30.0, "λ max (range 2)")
                    if wl2_min >= wl2_max:
                        raise InvalidInput("λ min must be smaller than λ max (range 2).")
                    w2_lo = wl2_min * 1e-6
                    w2_hi = wl2_max * 1e-6
                    expr  = (
                        f"((EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})) & "
                        f"((EFF_WAVE<{w2_lo}) | (EFF_WAVE>{w2_hi}))"
                    )

                # Allowlist the expression before it can ever reach
                # oimodeler's oifitsFlagWithExpression eval() sink (V5).
                expr = filter_expression(expr)

                # Stocke les paramètres de filtre dans session_state
                st.session_state.filter_expr   = expr
                st.session_state.filter_bin_L  = bin_L
                st.session_state.filter_bin_N  = bin_N
                st.session_state.filter_norm_L = norm_L
                st.session_state.filter_norm_N = norm_N

                if filepath is not None:
                    # ✅ Les longueurs d'onde filtrées sont cachées par data_service
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

            except InvalidInput as exc:
                st.warning(str(exc))
            except Exception:
                logger.exception("Cannot apply spectral filter")
                st.warning("Cannot apply the current filter settings.")

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
            except Exception:
                logger.exception("UV plot failed")
                st.warning("Could not render the UV plot for the current selection.")


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

        except Exception:
            logger.exception("Observable plot failed")
            st.warning("Could not render observable plots for the current selection.")


# ── Helper interne ─────────────────────────────────────────────────────────

def _get_active_data_with_filter():
    """
    Retourne l'objet oimData actif avec le filtre appliqué.
    Utilise le cache de load_oifits_multi() pour ne pas recharger le fichier.

    Paths are resolved only via resolve_selected_paths(), i.e. only names
    already present in st.session_state.loaded_files — never by
    reconstructing a path from a widget value (V2).
    """
    oim   = get_oim()
    paths = resolve_selected_paths(st.session_state.get('selected_files', []))

    if not paths:
        raise ValueError("No file selected.")

    data = load_oifits_multi(tuple(paths))

    expr  = st.session_state.get('filter_expr', '')
    bin_L = st.session_state.get('filter_bin_L', 1)
    bin_N = st.session_state.get('filter_bin_N', 1)
    norm_L = st.session_state.get('filter_norm_L', False)
    norm_N = st.session_state.get('filter_norm_N', False)

    file_order = st.session_state.get('selected_files', []) or []
    file_dtypes = st.session_state.get('file_dtypes', {})

    filters = build_data_type_filters(file_dtypes, file_order)
    if expr:
        filters.append(oim.oimFlagWithExpressionFilter(expr=expr, keepOldFlag=True))
    filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_L, normalizeError=norm_L))
    filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_N, normalizeError=norm_N))
    data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True

    return data
