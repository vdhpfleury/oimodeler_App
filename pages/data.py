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
    get_oim, get_file_summary, get_filtered_wavelengths_for_file,
    load_oifits_multi, build_per_file_filters,
)
from services.storage import store, resolve_selected_paths
from services.activity_log import log_event
from core.validation import num, choice, choices, InvalidInput
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
                log_event("File uploaded", f.name)
            except ValueError as exc:
                # Rejection reason is already a safe, user-facing message
                # (bad name/extension, wrong content, quota exceeded).
                st.error(str(exc))
                log_event("File upload rejected", f"{f.name}: {exc}")
            except OSError:
                # Distinct from a validation rejection: the server itself
                # couldn't write the upload (e.g. its storage directory is
                # missing or not writable) — say so rather than a generic
                # "try again", which just looked like a stuck upload.
                logger.exception("Upload storage failure for %s", f.name)
                st.error(
                    f"Could not save {f.name}: the server's upload storage "
                    "is unavailable. Contact the administrator."
                )
                log_event("File upload failed", f"{f.name}: storage unavailable")
            except Exception:
                logger.exception("Upload failed for %s", f.name)
                st.error(f"Could not load {f.name}. Please try again.")
                log_event("File upload failed", f.name)


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
        if selected != st.session_state.get('_last_logged_selection'):
            log_event("Dataset selection changed", ", ".join(selected) or "(none)")
            st.session_state['_last_logged_selection'] = list(selected)
        st.session_state.selected_files = selected

        try:
            st.session_state.selected_file = selected[0]
        except IndexError:
            # No selection yet: keep the previous value untouched — but
            # if nothing was ever selected this stays None (see session.py).
            pass

        if not selected:
            st.caption("Select at least one dataset to configure its filters.")
            return

        # ── Résumés par fichier (instrument, cible, config VLTI, λ native)
        # — lecture-seule et cachée (services/data_service.get_file_summary),
        # sert à la fois au sous-titre et au préremplissage des bornes.
        summaries = {
            fname: get_file_summary(st.session_state.loaded_files[fname])
            for fname in selected
        }
        by_instrument: dict[str, list[str]] = {}
        for fname in selected:
            instr = summaries[fname].get("instrument") or "Unknown"
            by_instrument.setdefault(instr, []).append(fname)

        st.markdown("##### Per-file spectral filtering")
        st.caption(
            "Each file below is filtered independently. Wavelength ranges "
            "default to the file's own coverage (no cut) — adjust only what "
            "needs narrowing."
        )
        for fname in selected:
            _render_one_file_filter(
                fname, summaries[fname], selected,
                by_instrument[summaries[fname].get("instrument") or "Unknown"],
            )

        rc1, rc2 = st.columns([2, 3])
        with rc1:
            try:
                data = _get_active_data_with_filter()
                st.info(f"Combined selection: {len(np.unique(data.vect_wl))} wavelength points.")
            except Exception:
                logger.exception("Cannot summarize the combined filtered selection")
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


def _render_one_file_filter(fname: str, summary: dict, all_selected: list[str],
                             same_instrument: list[str]) -> None:
    """One file's independent filter block: a metadata subtitle, optional
    'apply to all' shortcuts, wavelength range(s), binning, and the data
    types to keep — plus a live point-count check of the result."""
    wl_lo_native = summary.get("wl_min_um")
    wl_hi_native = summary.get("wl_max_um")

    subtitle_bits = [
        summary.get("instrument"),
        summary.get("target"),
        (summary.get("date_obs") or "")[:10] or None,
        summary.get("vlti_config"),
    ]
    subtitle_bits = [b for b in subtitle_bits if b]
    if wl_lo_native is not None:
        subtitle_bits.append(f"{wl_lo_native:.2f}–{wl_hi_native:.2f} µm native")
    subtitle = " · ".join(subtitle_bits) if subtitle_bits else "No metadata available"

    with st.expander(f"📄 {fname}", expanded=len(all_selected) <= 3):
        st.caption(subtitle)

        instr = summary.get("instrument")
        show_instr_btn = bool(instr) and len(same_instrument) > 1
        btn_cols = st.columns(2 if show_instr_btn else 1)
        with btn_cols[0]:
            if st.button("🔁 Apply to all files", key=f"apply_all_{fname}",
                         use_container_width=True, disabled=len(all_selected) <= 1):
                _copy_filter_widget_keys(fname, all_selected)
                st.rerun()
        if show_instr_btn:
            with btn_cols[1]:
                if st.button(f"🔁 Apply to all {instr} files", key=f"apply_instr_{fname}",
                             use_container_width=True):
                    _copy_filter_widget_keys(fname, same_instrument)
                    st.rerun()

        default_lo = wl_lo_native if wl_lo_native is not None else 0.1
        default_hi = wl_hi_native if wl_hi_native is not None else 20.0

        # `value=`/`default=` is passed on every render, not just the first
        # — omitting it once a widget's key already holds a value makes
        # Streamlit's number_input silently reset to 0 instead of reading
        # session_state, so despite the "created with a default value but
        # also had its value set via the Session State API" warning this
        # triggers right after "Apply to all" copies a value in
        # (_copy_filter_widget_keys), always passing it is the only
        # combination that keeps the widget's value correct.
        use_range = st.checkbox(
            "Keep only a wavelength sub-range", value=False,
            key=f"filt_use_range_{fname}",
        )
        wl_ranges_um: list[tuple[float, float]] = []
        if use_range:
            fc1, fc2 = st.columns(2)
            with fc1:
                lo1_raw = st.number_input("λ min (µm)", value=default_lo, step=0.1,
                                          format="%.2f", key=f"filt_wl_lo_{fname}")
            with fc2:
                hi1_raw = st.number_input("λ max (µm)", value=default_hi, step=0.1,
                                          format="%.2f", key=f"filt_wl_hi_{fname}")
            use_range2 = st.checkbox(
                "Add a second range to keep", value=False,
                key=f"filt_use_range2_{fname}",
            )
            lo2_raw = hi2_raw = None
            if use_range2:
                fc3, fc4 = st.columns(2)
                with fc3:
                    lo2_raw = st.number_input("λ min (µm) — range 2", value=default_lo, step=0.1,
                                              format="%.2f", key=f"filt_wl_lo2_{fname}")
                with fc4:
                    hi2_raw = st.number_input("λ max (µm) — range 2", value=default_hi, step=0.1,
                                              format="%.2f", key=f"filt_wl_hi2_{fname}")

        bc1, bc2 = st.columns(2)
        with bc1:
            bin_raw = st.number_input("Spectral binning", 1, 50, 1, key=f"filt_bin_{fname}")
        with bc2:
            norm_err = st.toggle("Normalize σ by bin size", value=False, key=f"filt_norm_{fname}")

        with st.expander("Data types to keep", expanded=False):
            default_dtypes = st.session_state.file_dtypes.get(fname, FITTABLE_DATA_TYPES)
            dtypes_raw = st.multiselect(
                "Data types", FITTABLE_DATA_TYPES,
                default=default_dtypes, key=f"dtypes_sel_{fname}",
            )

        try:
            wl_ranges_m: list[tuple[float, float]] = []
            if use_range:
                lo1 = num(lo1_raw, 0.1, 30.0, "λ min")
                hi1 = num(hi1_raw, 0.1, 30.0, "λ max")
                if lo1 >= hi1:
                    raise InvalidInput("λ min must be smaller than λ max.")
                wl_ranges_m.append((lo1 * 1e-6, hi1 * 1e-6))
                if use_range2:
                    lo2 = num(lo2_raw, 0.1, 30.0, "λ min (range 2)")
                    hi2 = num(hi2_raw, 0.1, 30.0, "λ max (range 2)")
                    if lo2 >= hi2:
                        raise InvalidInput("λ min must be smaller than λ max (range 2).")
                    wl_ranges_m.append((lo2 * 1e-6, hi2 * 1e-6))

            bin_size = num(bin_raw, 1, 50, "Spectral binning", integer=True)
            dtypes   = choices(dtypes_raw, FITTABLE_DATA_TYPES, "Data types")

            new_cfg = {
                "wl_ranges":     wl_ranges_m,
                "bin":           bin_size,
                "normalize_err": bool(norm_err),
            }
            # Only log when the effective filter actually changed — this
            # function reruns on every Streamlit interaction anywhere on
            # the page, not just when this file's own widgets move.
            if (st.session_state.file_filters.get(fname) != new_cfg
                    or st.session_state.file_dtypes.get(fname) != dtypes):
                ranges_txt = ", ".join(
                    f"[{lo*1e6:.2f}, {hi*1e6:.2f}]µm" for lo, hi in wl_ranges_m
                ) or "full range"
                log_event(
                    "Filter updated", f"{fname}: λ={ranges_txt}, bin={bin_size}, "
                    f"norm_err={bool(norm_err)}, dtypes={','.join(dtypes) or 'all'}",
                )
            st.session_state.file_filters[fname] = new_cfg
            st.session_state.file_dtypes[fname] = dtypes

            filepath = st.session_state.loaded_files[fname]
            wls = get_filtered_wavelengths_for_file(
                filepath, tuple(wl_ranges_m), bin_size, bool(norm_err),
            )
            if wls:
                wls_arr = np.array(wls)
                st.caption(
                    f"→ {len(wls)} wavelength points after filtering  |  "
                    f"λ ∈ [{wls_arr.min()*1e6:.3f}, {wls_arr.max()*1e6:.3f}] µm"
                )
            else:
                st.warning("This filter removes every wavelength point of this file.")
        except InvalidInput as exc:
            st.warning(str(exc))
        except Exception:
            logger.exception("Cannot apply per-file filter for %s", fname)
            st.warning("Cannot apply the current filter settings for this file.")


def _copy_filter_widget_keys(src: str, targets: list[str]) -> None:
    """Copies one file's filter *widget* values onto other files' own widget
    keys, not just the derived `file_filters` dict — a Streamlit widget only
    honors its `value=` default on the very first render for a given key, so
    writing `file_filters` alone would leave the visible widgets unchanged
    until the user touches them."""
    widget_keys = (
        "filt_use_range", "filt_wl_lo", "filt_wl_hi",
        "filt_use_range2", "filt_wl_lo2", "filt_wl_hi2",
        "filt_bin", "filt_norm", "dtypes_sel",
    )
    for target in targets:
        if target == src:
            continue
        for k in widget_keys:
            src_key = f"{k}_{src}"
            if src_key in st.session_state:
                value = st.session_state[src_key]
                st.session_state[f"{k}_{target}"] = (
                    list(value) if isinstance(value, list) else value
                )


# ── Section 3 : Observables ───────────────────────────────────────────────

def _render_observables() -> None:
    with st.expander("III. Observable visualization", expanded=True):

        try:
            data = _get_active_data_with_filter()
        except ValueError as exc:
            if str(exc) == "No file selected.":
                # Expected, frequent state right after an upload and before
                # the user has picked anything in "Select data to use" above
                # — not a real error, so no full traceback in the server log.
                st.info("Select a dataset above to preview its observables.")
                return
            logger.exception("Observable plot failed")
            st.warning("Could not render observable plots for the current selection.")
            return
        except Exception:
            logger.exception("Observable plot failed")
            st.warning("Could not render observable plots for the current selection.")
            return

        # Not every OIFITS file carries every observable — e.g. many MATISSE
        # exports have no OI_FLUX table, so FLUXDATA is absent. Each panel is
        # independently optional: one missing/unsupported observable must
        # not blank out the other two, which oimodeler's own oiplot() would
        # otherwise do since it raises before any panel is drawn.
        fig, (ax1, ax2, ax3) = plt.subplots(
            ncols=3, figsize=(15, 4),
            subplot_kw={'projection': 'oimAxes'},
        )
        panels = (
            (ax1, "SPAFREQ", "VIS2DATA", "cycle/mas", "VIS2", dict(color="byBaseline")),
            (ax2, "EFF_WAVE", "T3PHI",   "micron",    "T3PHI", dict(color="byBaseline")),
            (ax3, "EFF_WAVE", "FLUXDATA","micron",    "FLUXDATA", {}),
        )
        any_ok = False
        for ax, xname, yname, xunit, title, kwargs in panels:
            try:
                ax.oiplot(data, xname, yname, xunit=xunit, errorbar=True, **kwargs)
                ax.set_title(title)
                ax.legend(fontsize=8)
                any_ok = True
            except Exception:
                logger.info("Observable %s unavailable for the current selection", yname)
                ax.set_title(f"{title} (not available)")
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

        if any_ok:
            plt.tight_layout()
            safe_pyplot(st, fig, use_container_width=True)
        else:
            plt.close(fig)
            st.warning("Could not render observable plots for the current selection.")


# ── Helper interne ─────────────────────────────────────────────────────────

def _get_active_data_with_filter():
    """
    Retourne l'objet oimData actif avec le filtre appliqué — chaque fichier
    filtré indépendamment des autres (voir _render_one_file_filter() et
    services/data_service.build_per_file_filters()).
    Utilise le cache de load_oifits_multi() pour ne pas recharger le fichier.

    Paths are resolved only via resolve_selected_paths(), i.e. only names
    already present in st.session_state.loaded_files — never by
    reconstructing a path from a widget value (V2).
    """
    paths = resolve_selected_paths(st.session_state.get('selected_files', []))

    if not paths:
        raise ValueError("No file selected.")

    data = load_oifits_multi(tuple(paths))

    oim         = get_oim()
    file_order  = st.session_state.get('selected_files', []) or []
    file_filters = st.session_state.get('file_filters', {})
    file_dtypes  = st.session_state.get('file_dtypes', {})

    filters = build_per_file_filters(file_filters, file_dtypes, file_order)
    data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True

    return data
