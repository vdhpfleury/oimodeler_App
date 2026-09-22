# pages/data.py
"""
Page "Data" – Chargement, filtrage générique et diagnostic des fichiers OIFITS.

Cette page ne contient QUE de la logique UI.
- Elle ne charge PAS directement les fichiers (→ services/data_service.py)
- Elle ne stocke PAS d'objets lourds dans session_state (→ chemins et specs
  de filtre sérialisables uniquement)
- Elle délègue les calculs à core/ et services/

Filter workbench
-----------------
N'importe quel filtre du registre core/filter_registry.py (miroir vérifié
des classes réellement installées dans oimodeler — voir ce module) peut
être appliqué à n'importe quel sous-ensemble de fichiers/arrays/types de
données. Les filtres appliqués sont partagés par TOUTE la session — Data,
Fitting et Modelling consomment tous la même pile via
services/data_service.get_active_data() — et empilés par signature
(classe, targets, arr) : réappliquer le même filtre sur la même
cible remplace l'entrée existante plutôt que d'en empiler une copie.

Security notes (docs/security_audit_2026-09.md):
- Uploads go through services/storage.store() (V1: no path built from a
  client filename; V3: session-scoped directory; V8: quotas + purge).
- File selection is always filtered against st.session_state.loaded_files
  before use — an allowlist, never a reconstructed path (V2).
- oimFlagWithExpressionFilter (raw-text expression reaching oimodeler's
  eval() sink) is deliberately NOT in the filter registry — see
  core/filter_registry.py's docstring (V5).
- Every filter parameter is re-validated server-side (core/validation.py)
  before it can reach a filter constructor — widget bounds are never
  trusted on their own (V4).
"""
from __future__ import annotations

import io
import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from services.data_service import get_oim, get_filter_metadata, get_active_data
from services.storage import store
from services.activity_log import log_event
from core.validation import num, choice, choices, InvalidInput
from core.filter_registry import FILTER_REGISTRY
from core.oifits_meta import get_available_values, summarize_oimdata
from components.plots import safe_pyplot

logger = logging.getLogger(__name__)

_WL_MIN_UM, _WL_MAX_UM = 0.1, 30.0


def render() -> None:
    """Point d'entrée de la page Data, appelé depuis app.py."""
    _render_file_upload()

    if not st.session_state.loaded_files:
        st.info("Please load at least one OIFITS file.")
        return

    _render_filter_section()
    _render_diagnostics()


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


# ── Section 2 : Data info + filter workbench ───────────────────────────────

def _render_filter_section() -> None:
    with st.expander("II. Data info & filters", expanded=True):
        # Explicit key: keeps the selection stable and immune to Streamlit
        # regenerating an implicit auto-key for this widget on re-render;
        # also lets us sanitize a stale entry below without touching the
        # multiselect's own default.
        options = list(st.session_state.loaded_files.keys())
        stale = [n for n in st.session_state.get('data_selected_files', []) if n not in options]
        if stale:
            st.session_state.data_selected_files = [
                n for n in st.session_state.data_selected_files if n not in stale
            ]
        raw_selected = st.multiselect(
            "Select data to use",
            options=options,
            key="data_selected_files",
        )
        # Server-side allowlist guard (V2): multiselect returns an unknown
        # client value as-is instead of raising, so we drop anything that
        # isn't actually a key of loaded_files rather than trusting it.
        selected = [n for n in raw_selected if n in st.session_state.loaded_files]

        if selected != st.session_state.get('_last_logged_selection'):
            log_event("Dataset selection changed", ", ".join(selected) or "(none)")
            st.session_state['_last_logged_selection'] = list(selected)
            # Applied filters target files by INDEX into this exact list
            # (see FILTER_REGISTRY's "target" parameter) — a changed
            # selection can silently point an old filter at the wrong
            # file, so clear the stack rather than risk that.
            if st.session_state.applied_filters:
                st.session_state.applied_filters = []
                log_event("Filters reset", "dataset selection changed")

        st.session_state.selected_files = selected
        try:
            st.session_state.selected_file = selected[0]
        except IndexError:
            pass

        if not selected:
            st.caption("Select at least one dataset to see its info and configure filters.")
            return

        metadata = pd.concat(
            [
                get_filter_metadata(st.session_state.loaded_files[fname], fname)
                for fname in selected
            ],
            ignore_index=True,
        )

        col_info, col_filter = st.columns(2)

        with col_info:
            st.markdown("##### oimData info")
            try:
                data = get_active_data(selected)
                for file_label, df in summarize_oimdata(data, selected):
                    st.caption(file_label)
                    if df.empty:
                        st.caption("No observable arrays left after filtering.")
                    else:
                        st.dataframe(df, hide_index=True, use_container_width=True)
            except Exception:
                logger.exception("Cannot summarize the current selection")
                st.warning("Cannot load/summarize the current selection.")

            n_active = len(st.session_state.applied_filters)
            if n_active:
                st.caption(f"{n_active} filter(s) currently applied.")
                if st.button("↩️ Reset filters", type="primary", use_container_width=True):
                    st.session_state.applied_filters = []
                    log_event("Filters reset", "manual")
                    st.rerun()

        with col_filter:
            st.markdown("##### Add a filter")
            _render_filter_picker(selected, metadata)


def _render_filter_picker(file_names: list[str], metadata: pd.DataFrame) -> None:
    filter_name = st.selectbox(
        "Filter type", list(FILTER_REGISTRY.keys()), key="filter_picker_name",
    )
    filter_info = FILTER_REGISTRY[filter_name]
    st.caption(filter_info["description"])

    try:
        kwargs = _render_filter_form(filter_name, filter_info, metadata, file_names)
    except InvalidInput as exc:
        st.warning(str(exc))
        return

    if st.button("➕ Apply filter", type="primary", use_container_width=True,
                 key="apply_filter_btn"):
        oim = get_oim()
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            getattr(oim, filter_name)(**clean_kwargs)
        except Exception as exc:
            st.error(f"Could not build {filter_name}: {exc}")
            return

        signature = list(_filter_signature(filter_name, kwargs))
        applied = [
            e for e in st.session_state.applied_filters
            if list(e["signature"]) != signature
        ]
        applied.append({
            "filter_class": filter_name, "kwargs": kwargs, "signature": signature,
        })
        st.session_state.applied_filters = applied
        log_event(
            "Filter applied",
            f"{filter_name} targets={kwargs.get('targets')} arr={kwargs.get('arr')}",
        )
        st.rerun()


def _filter_signature(filter_name: str, kwargs: dict) -> tuple:
    """Identifies a filter by (class, targeted files, targeted arrays) so
    re-applying it to the same target replaces the existing entry instead
    of stacking a duplicate. Other parameters are free to change — they
    simply update the existing entry for that signature."""
    targets = kwargs.get("targets")
    targets_key = tuple(sorted(targets)) if isinstance(targets, list) else targets
    arr = kwargs.get("arr")
    arr_key = tuple(sorted(arr)) if isinstance(arr, list) else arr
    return (filter_name, targets_key, arr_key)


def _render_filter_form(filter_name: str, filter_info: dict,
                        metadata: pd.DataFrame, file_names: list[str]) -> dict:
    """Renders every parameter of the selected filter in declaration
    order, so "targets" resolves before "arr" (which restricts available
    arrays to the targeted files) and "arr" resolves before "dataType"
    (which restricts data types to the selected arrays)."""
    kwargs: dict = {}
    target_indices: list[int] | None = None
    selected_arrays: list[str] | None = None

    for param_name, param in filter_info["parameters"].items():
        ptype = param["type"]
        key = f"{filter_name}__{param_name}"

        if ptype == "target":
            value = _render_target_param(key, param, file_names)
            kwargs[param_name] = value
            target_indices = value
            continue

        available = get_available_values(metadata, file_names, target_indices, selected_arrays)
        value = _render_param(key, param, available)
        kwargs[param_name] = value
        if ptype == "array" and value:
            selected_arrays = value

    return kwargs


def _render_target_param(key: str, param: dict, file_names: list[str]) -> list[int] | None:
    """"targets" is common to every filter. Per the oimodeler docs it
    takes "all" or a list of indices into oimData's file list — NOT an
    astronomical target/object name. Returns None (→ omitted from
    kwargs, letting the class default to "all") when nothing/"All files"
    is picked."""
    if not file_names:
        st.caption(f"{param['label']}: load data to select target files.")
        return None

    options = ["All files"] + [f"{i} — {name}" for i, name in enumerate(file_names)]
    selection = st.multiselect(param["label"], options, default=["All files"], key=key)

    if not selection or "All files" in selection:
        return None

    try:
        indices = sorted({int(item.split(" — ")[0]) for item in selection})
    except (ValueError, IndexError):
        raise InvalidInput(f"{param['label']}: invalid selection.")
    if any(not (0 <= i < len(file_names)) for i in indices):
        raise InvalidInput(f"{param['label']}: file index out of range.")
    return indices


_CHOICE_AVAILABLE_KEY = {
    "insname": "insnames", "dataType": "dataTypes",
    "baseline": "baselines", "telescope": "telescopes",
}


def _render_param(key: str, param: dict, available: dict):
    """Renders one non-"target" parameter's widget(s) and returns its
    validated value, or None to omit it from kwargs entirely (letting the
    filter class use its own default — almost always "all"). Raises
    InvalidInput on an out-of-bounds/invalid value (V4: widget bounds are
    cosmetic only, never server-enforced by Streamlit itself)."""
    ptype, label = param["type"], param["label"]

    if ptype == "array":
        options = available["arrays"]
        if not options:
            st.caption(f"{label}: load data to see the available values.")
            return None
        raw = st.multiselect(label, options, key=key)
        return choices(raw, options, label) or None

    if ptype in _CHOICE_AVAILABLE_KEY:
        options = available[_CHOICE_AVAILABLE_KEY[ptype]]
        if not options:
            st.caption(f"{label}: load data to see the available values.")
            return None
        raw = st.multiselect(label, options, key=key)
        return choices(raw, options, label) or None

    if ptype == "wavelength_range":
        default_lo, default_hi = available["wl_range"]
        default_lo = default_lo if default_lo is not None else _WL_MIN_UM
        default_hi = default_hi if default_hi is not None else _WL_MAX_UM
        c1, c2 = st.columns(2)
        with c1:
            lo_raw = st.number_input(f"{label} min (µm)", value=float(default_lo),
                                     step=0.1, format="%.3f", key=f"{key}_min")
        with c2:
            hi_raw = st.number_input(f"{label} max (µm)", value=float(default_hi),
                                     step=0.1, format="%.3f", key=f"{key}_max")
        lo = num(lo_raw, _WL_MIN_UM, _WL_MAX_UM, f"{label} min")
        hi = num(hi_raw, _WL_MIN_UM, _WL_MAX_UM, f"{label} max")
        if lo >= hi:
            raise InvalidInput(f"{label}: min must be smaller than max.")
        return [lo * 1e-6, hi * 1e-6]

    if ptype == "float_um":
        raw = st.number_input(f"{label} (µm)", value=float(param.get("default", 0.0)),
                              step=0.1, format="%.4f", key=key)
        value_um = num(raw, param["min"], param["max"], label)
        return value_um * 1e-6

    if ptype == "array_float_um":
        raw_text = st.text_input(f"{label}", key=key)
        if not raw_text.strip():
            return None
        try:
            values_um = [float(x) for x in raw_text.split(",") if x.strip()]
        except ValueError:
            raise InvalidInput(f"{label}: enter comma-separated numbers.")
        if len(values_um) > param.get("max_len", 200):
            raise InvalidInput(f"{label}: too many values (max {param.get('max_len', 200)}).")
        validated = [num(v, param["min"], param["max"], label) for v in values_um]
        return [v * 1e-6 for v in validated]

    if ptype == "bool":
        return st.checkbox(label, value=bool(param.get("default", False)), key=key)

    if ptype == "int":
        raw = st.number_input(label, value=int(param.get("default", param["min"])),
                              min_value=param["min"], max_value=param["max"], step=1, key=key)
        return num(raw, param["min"], param["max"], label, integer=True)

    if ptype == "float":
        raw = st.number_input(label, value=float(param.get("default", param["min"])),
                              min_value=float(param["min"]), max_value=float(param["max"]), key=key)
        return num(raw, param["min"], param["max"], label)

    if ptype == "optional_float":
        use_it = st.checkbox(f"Set {label}", value=False, key=f"{key}_use")
        if not use_it:
            return None
        raw = st.number_input(label, value=float(param["min"]),
                              min_value=float(param["min"]), max_value=float(param["max"]), key=key)
        return num(raw, param["min"], param["max"], label)

    if ptype == "select":
        options = param["options"]
        default = param.get("default", options[0])
        raw = st.selectbox(label, options, index=options.index(default),
                           key=key, help=param.get("help"))
        return choice(raw, options, label)

    if ptype in ("per_datatype_float", "per_datatype_optional_float"):
        # Cross-parameter: depends on the sibling "dataType" multiselect,
        # already rendered (and stored under its own key) earlier in this
        # same form — FILTER_REGISTRY declares dataType before values/
        # relThreshold specifically so this lookup finds a real value.
        # oimodeler's setMinimumError() requires values/relThreshold to
        # be either a scalar (broadcast to every data type) or a list of
        # EXACTLY the same length as dataType — a shorter list raises
        # IndexError deep inside oimUtils.setMinimumError as soon as the
        # filter is applied. Always emitting one entry per selected data
        # type (never a bare scalar) sidesteps that mismatch entirely.
        filter_key_prefix = key.rsplit("__", 1)[0]
        selected_types = st.session_state.get(f"{filter_key_prefix}__dataType", [])
        if not selected_types:
            st.caption(f"{label}: select a data type above first.")
            return None

        cols = st.columns(min(len(selected_types), 4))
        result = []
        for i, dtype in enumerate(selected_types):
            with cols[i % len(cols)]:
                st.caption(dtype)
                if ptype == "per_datatype_optional_float":
                    use_it = st.checkbox("Set", value=False, key=f"{key}_{dtype}_use")
                    if not use_it:
                        result.append(None)
                        continue
                raw = st.number_input(
                    label, value=float(param.get("default", param["min"])),
                    min_value=float(param["min"]), max_value=float(param["max"]),
                    key=f"{key}_{dtype}", label_visibility="collapsed",
                )
                result.append(num(raw, param["min"], param["max"], label))
        return result

    if ptype == "diff_err_range":
        # Cross-parameter: depends on the sibling "rangeType" select,
        # already rendered (and stored under its own key) earlier in this
        # same form — FILTER_REGISTRY declares rangeType before ranges.
        filter_key_prefix = key.rsplit("__", 1)[0]
        range_type = st.session_state.get(f"{filter_key_prefix}__rangeType", "index")
        c1, c2 = st.columns(2)
        if range_type == "wavelength":
            default_lo, default_hi = available["wl_range"]
            default_lo = default_lo if default_lo is not None else _WL_MIN_UM
            default_hi = default_hi if default_hi is not None else _WL_MAX_UM
            with c1:
                lo_raw = st.number_input(f"{label} min (µm)", value=float(default_lo), key=f"{key}_lo")
            with c2:
                hi_raw = st.number_input(f"{label} max (µm)", value=float(default_hi), key=f"{key}_hi")
            lo = num(lo_raw, _WL_MIN_UM, _WL_MAX_UM, f"{label} min")
            hi = num(hi_raw, _WL_MIN_UM, _WL_MAX_UM, f"{label} max")
            if lo >= hi:
                raise InvalidInput(f"{label}: min must be smaller than max.")
            return [[lo * 1e-6, hi * 1e-6]]
        else:
            with c1:
                lo_raw = st.number_input(f"{label} min (channel index)", value=0,
                                         min_value=0, max_value=100000, key=f"{key}_lo")
            with c2:
                hi_raw = st.number_input(f"{label} max (channel index)", value=5,
                                         min_value=0, max_value=100000, key=f"{key}_hi")
            lo = num(lo_raw, 0, 100000, f"{label} min", integer=True)
            hi = num(hi_raw, 0, 100000, f"{label} max", integer=True)
            if lo >= hi:
                raise InvalidInput(f"{label}: min must be smaller than max.")
            return [[lo, hi]]

    # Unreachable for every type currently declared in FILTER_REGISTRY.
    return st.text_input(label, key=key) or None


# ── Section 3 : Diagnostic plots ────────────────────────────────────────────

def _render_diagnostics() -> None:
    with st.expander("III. Diagnostic plots", expanded=True):
        try:
            data = get_active_data(st.session_state.get('selected_files', []))
        except ValueError as exc:
            if str(exc) == "No file selected.":
                # Expected, frequent state right after an upload and before
                # the user has picked anything in "Select data to use" above
                # — not a real error, so no full traceback in the server log.
                st.info("Select a dataset above to preview its observables.")
            else:
                logger.exception("Diagnostic plot failed")
                st.warning("Could not render diagnostic plots for the current selection.")
            return
        except Exception:
            logger.exception("Diagnostic plot failed")
            st.warning("Could not render diagnostic plots for the current selection.")
            return

        color_options = ["byFile", "byBaseline", "byConfiguration", "byArrname"]
        color_raw = st.selectbox("Color by", color_options, key="diag_color_choice")
        color_choice = choice(color_raw, color_options, "Color by")

        # Each panel is independently optional: one missing/unsupported
        # observable (e.g. no OI_FLUX table, common on some MATISSE
        # exports) must not blank out the others.
        row1c1, row1c2 = st.columns(2)
        row2c1, row2c2 = st.columns(2)

        with row1c1:
            st.caption("UV coverage")
            _render_diagnostic_plot(lambda ax: ax.uvplot(data, color=color_choice))
        with row1c2:
            st.caption("Squared visibility (VIS2DATA)")
            _render_diagnostic_plot(lambda ax: _oiplot(
                ax, data, "SPAFREQ", "VIS2DATA", "cycle/mas", color_choice))
        with row2c1:
            st.caption("Closure phase (T3PHI)")
            _render_diagnostic_plot(lambda ax: _oiplot(
                ax, data, "SPAFREQ", "T3PHI", "cycle/rad", color_choice))
        with row2c2:
            st.caption("Total flux (FLUXDATA)")
            _render_diagnostic_plot(lambda ax: _oiplot(
                ax, data, "EFF_WAVE", "FLUXDATA", "micron", color_choice))

        st.markdown("##### Custom plot")
        _render_custom_plot(data)


def _oiplot(ax, data, xname: str, yname: str, xunit: str, color: str) -> None:
    ax.oiplot(data, xname, yname, xunit=xunit, color=color, errorbar=True)
    ax.legend(fontsize=6)


def _render_diagnostic_plot(build_plot, download_filename: str | None = None,
                            key: str | None = None) -> None:
    """Builds and displays one oimAxes-based diagnostic plot. Any error
    while building it (observable not present in the loaded/filtered
    data, empty selection, ...) is shown as a caption instead of
    crashing the app or blanking out the other panels."""
    fig = plt.figure(figsize=(5, 4))
    try:
        ax = fig.add_subplot(projection="oimAxes")
        build_plot(ax)
    except Exception as exc:
        plt.close(fig)
        st.caption(f"Not available: {exc}")
        return

    png_bytes = None
    if download_filename:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        png_bytes = buf.getvalue()

    safe_pyplot(st, fig)

    if png_bytes:
        st.download_button(
            "Download plot (PNG)", data=png_bytes,
            file_name=download_filename, mime="image/png", key=key,
        )


def _render_custom_plot(data) -> None:
    x_options = ["SPAFREQ", "EFF_WAVE"]
    # "UV plane" isn't a real oimPlot yname — selecting it switches the
    # whole panel to ax.uvplot(data, color=...) instead of ax.oiplot(...),
    # so X quantity/unit (meaningless for a uv-coverage plot) are hidden.
    observable_options = ["UV plane", "VIS2DATA", "VISAMP", "VISPHI", "T3AMP", "T3PHI", "FLUXDATA"]
    xunit_options = {
        "SPAFREQ":  ["cycle/mas", "cycle/rad", "cycle/arcsec", "m", "km"],
        "EFF_WAVE": ["micron", "nm", "m", "Angstrom"],
    }
    # byInsname (a real oimPlot color mode) was missing here. A continuous
    # colormap by wavelength isn't offered: oimodeler's oiplot only
    # supports these categorical color modes (getColorIndices) — no
    # continuous-by-EFF_WAVE mode exists to delegate to.
    color_options  = ["byFile", "byBaseline", "byConfiguration", "byArrname", "byInsname"]
    marker_options = ["none", ".", "o", "+", "x", "s", "^"]

    col_params, col_plot = st.columns([1, 2])
    with col_params:
        observable_raw = st.selectbox("Observable", observable_options, key="custom_y_quantity")
        is_uv          = observable_raw == "UV plane"
        if not is_uv:
            x_quantity_raw = st.selectbox("X quantity", x_options, key="custom_x_quantity")
            xunit_raw      = st.selectbox(
                "X unit", xunit_options[choice(x_quantity_raw, x_options, "X quantity")],
                key="custom_xunit",
            )
        color_raw      = st.selectbox("Color by", color_options, key="custom_color_choice")
        if not is_uv:
            marker_raw     = st.selectbox("Marker", marker_options, key="custom_marker")
            linewidth_raw  = st.number_input("Line width", value=1.0, min_value=0.0, max_value=10.0,
                                             step=0.5, key="custom_linewidth")
            alpha          = st.slider("Alpha", 0.0, 1.0, 1.0, key="custom_alpha")
            errorbar       = st.checkbox("Error bars", value=True, key="custom_errorbar")
            logscale       = st.checkbox("Log scale (Y)", value=False, key="custom_logscale")
        show_grid      = st.checkbox("Show grid", value=False, key="custom_showgrid")

    try:
        color = choice(color_raw, color_options, "Color by")
        if not is_uv:
            x_quantity = choice(x_quantity_raw, x_options, "X quantity")
            xunit      = choice(xunit_raw, xunit_options[x_quantity], "X unit")
            observable = choice(observable_raw, observable_options, "Observable")
            marker     = choice(marker_raw, marker_options, "Marker")
            linewidth  = num(linewidth_raw, 0.0, 10.0, "Line width")
    except InvalidInput as exc:
        with col_plot:
            st.warning(str(exc))
        return

    def build_custom_plot(ax):
        if is_uv:
            ax.uvplot(data, color=color)
        else:
            ax.oiplot(
                data, x_quantity, observable, xunit=xunit, color=color,
                errorbar=errorbar, marker=None if marker == "none" else marker,
                lw=linewidth, alpha=alpha,
            )
            if logscale:
                ax.set_yscale("log")
            ax.legend(fontsize=6)
        if show_grid:
            ax.grid(True, alpha=0.3)

    with col_plot:
        _render_diagnostic_plot(build_custom_plot, download_filename="custom_plot.png",
                                key="dl_custom_plot")
