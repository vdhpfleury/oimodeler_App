# pages/modelling.py
"""
Page "Modelling" – Configuration, sauvegarde et gestion des modèles.

Tabs :
    1. Basic Model      – ajout/édition de composants + preview
    2. Load CSV         – import d'un modèle depuis un CSV de résultats
    3. Interpolators    – configuration des oimInterp
    4. Model summary    – visualisation χ² + VIS²/T3PHI
    5. Model management – renommer / supprimer des modèles

Dépendances :
    services/data_service.py  → get_oim(), get_registry(), load_oifits()
    core/component.py         → make_comp_dict(), get_comp_by_name()
    core/model_builder.py     → build_oim_model(), generate_model_image_preview(),
                                 generate_model_v2_t3phi_preview()
    core/csv_import.py        → parse_csv_to_model()
    components/param_editor.py→ render_param_editor(), read_all_widgets()
    components/plots.py       → safe_pyplot()
"""
from __future__ import annotations

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from services.data_service import (
    get_oim, get_registry, load_oifits_multi, build_per_file_filters,
)
from services.storage import resolve_selected_paths
from core.component import make_comp_dict, get_comp_by_name
from core.model_builder import (
    build_oim_model,
    generate_model_image_preview,
    generate_model_v2_t3phi_preview,
)
from core.csv_import import parse_csv_to_model
from core.validation import num, choice, text, InvalidInput
from components.param_editor import render_param_editor, read_all_widgets
from components.plots import safe_pyplot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def render() -> None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Basic Model",
        "Load CSV model",
        "Interpolators",
        "Model summary",
        "Model management",
    ])

    with tab1:
        _render_basic_model()
    with tab2:
        _render_csv_import()
    with tab3:
        _render_interpolators()
    with tab4:
        _render_model_summary()
    with tab5:
        _render_model_management()


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 – Basic Model
# ═══════════════════════════════════════════════════════════════════════════

def _render_basic_model() -> None:
    registry = get_registry()
    oim      = get_oim()

    col_A, col_B = st.columns([2, 2])

    # ── Colonne A : contrôles ──────────────────────────────────────────
    with col_A:
        st.markdown("##### A. Initialize model")

        # Charger un modèle existant
        if st.session_state.MODEL:
            if st.checkbox("Load an existing model", key="load_existing_cb", width=300):
                model_names = sorted(st.session_state.MODEL.keys())
                model_to_load_raw = st.selectbox(
                    "Model to load",
                    model_names,
                    key="model_to_load_sel",
                )
                if st.button("📂 Load", key="btn_load_existing"):
                    try:
                        # selectbox returns an unrecognized client value
                        # as-is instead of raising — re-validate (V4).
                        model_to_load = choice(model_to_load_raw, model_names, "Model to load")
                        loaded = st.session_state.MODEL[model_to_load]
                        st.session_state.components = [
                            {
                                **c,
                                "params": registry.get(c["type"], {}).get(
                                    "params", c.get("params", [])
                                ),
                                "interpolators": c.get("interpolators", {}),
                            }
                            for c in loaded["components"]
                        ]
                        st.session_state.active_comp_name = (
                            st.session_state.components[0]["name"]
                            if st.session_state.components else None
                        )
                        st.success(f"✅ Model **{model_to_load}** loaded for editing!")
                        st.rerun()
                    except InvalidInput as exc:
                        st.error(str(exc))

        model_name_raw = st.text_input(
            "Model name", value="", placeholder="e.g.: uniform_disk",
            key="model_name_input", width=300
        )
        try:
            model_name = text(model_name_raw, "Model name", max_len=64)
        except InvalidInput as exc:
            st.warning(str(exc))
            model_name = ""

        st.write("**Add a component**")
        comp_type_options = list(registry.keys())
        comp_type_sel_raw = st.selectbox(
            "Type", comp_type_options,
            label_visibility="collapsed",
            format_func=lambda x: f"{x} — {registry[x]['description']}",
            width=300
        )
        try:
            # selectbox returns an unrecognized client value as-is — a
            # forged component type would otherwise reach registry[...]
            # and raise an unhandled KeyError (V4).
            comp_type_sel = choice(comp_type_sel_raw, comp_type_options, "Component type")
        except InvalidInput as exc:
            st.error(str(exc))
            comp_type_sel = comp_type_options[0]
        comp_name_inp_raw = st.text_input(
            "Component name", value=comp_type_sel, key="new_comp_name", width=300
        )
        if st.button("➕ Add", type="primary"):
            try:
                comp_name_inp = text(comp_name_inp_raw, "Component name", max_len=64)
            except InvalidInput as exc:
                st.error(str(exc))
            else:
                existing = [c['name'] for c in st.session_state.components]
                final    = comp_name_inp
                if final in existing:
                    suf = 2
                    while f"{comp_name_inp}_{suf}" in existing:
                        suf += 1
                    final = f"{comp_name_inp}_{suf}"
                st.session_state.components.append(
                    make_comp_dict(comp_type_sel, final, registry)
                )
                st.session_state.active_comp_name = final
                st.rerun()

        img_graph = st.toggle(
            "Show image or graph", key="img_graphe",
        )

    # ── Colonne B : preview ────────────────────────────────────────────
    with col_B:
        if st.session_state.components:
            read_all_widgets(st.session_state.components)
            with st.spinner("Rendering …", show_time=True):
                if not img_graph:
                    with st.expander(label="Image preview parameters", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1 :
                            model_preview_img_fov_raw    = st.number_input("pixel number", value=128, key="model_preview_img_fov")
                            model_preview_img_pxsize_raw = st.number_input("pixel size in mas", value=0.15, key="model_preview_img_pxsize")
                        with col2 :
                            model_preview_img_gamma_raw = st.number_input("gamma", value=0.2, key="model_preview_img_gamma", help="power low apply on each px")
                            model_preview_img_wl_raw = st.number_input("wavelength in µm", value=3.5, key="model_preview_img_wl")
                        col3, col4 = st.columns(2)
                        with col3:
                            model_preview_clip_lo_raw = st.number_input(
                                "colormap percentile min", value=0.5, min_value=0., max_value=100.,
                                key="model_preview_clip_lo",
                                help="Clips colors below this percentile of the displayed image (independent of gamma).",
                            )
                        with col4:
                            model_preview_clip_hi_raw = st.number_input(
                                "colormap percentile max", value=99.5, min_value=0., max_value=100.,
                                key="model_preview_clip_hi",
                            )

                    try:
                        # Widget bounds are cosmetic only — these feed
                        # model.getImage()'s array allocation directly,
                        # the concrete OOM DoS vector from the audit (V6).
                        model_preview_img_fov    = num(model_preview_img_fov_raw, 16, 1024, "Pixel number", integer=True)
                        model_preview_img_pxsize = num(model_preview_img_pxsize_raw, 0.001, 10.0, "Pixel size")
                        model_preview_img_gamma  = num(model_preview_img_gamma_raw, 0.01, 5.0, "Gamma")
                        model_preview_img_wl     = num(model_preview_img_wl_raw, 0.1, 30.0, "Wavelength")

                        # Widget bounds aren't server-enforced — re-validate before use.
                        clip_lo, clip_hi = sorted((
                            min(max(float(model_preview_clip_lo_raw), 0.), 100.),
                            min(max(float(model_preview_clip_hi_raw), 0.), 100.),
                        ))

                        fig = generate_model_image_preview(
                            oim, registry, st.session_state.components, model_preview_img_fov, model_preview_img_pxsize, model_preview_img_gamma, model_preview_img_wl*1e-6,
                            clip_percentile=(clip_lo, clip_hi),
                        )
                        if fig:
                            safe_pyplot(st, fig, use_container_width=False)
                    except InvalidInput as exc:
                        st.warning(str(exc))
                else:
                    data = _get_active_data_with_filter()
                    if data is not None:
                        with st.expander(label="Graph preview parameters", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1 : 
                                st.write("$V^2$")
                                model_preview_V2_Ymax    = st.number_input(r"$Y_{max}$ ", value=1., key="model_preview_V2_Ymax")
                                model_preview_V2_Ymin    = st.number_input(r"$Y_{min}$ ", value=0., key="model_preview_V2_Ymin")
                            with col2 : 
                                st.write("$T3PHI$")
                                model_preview_T3PHI_Ymax    = st.number_input(r"$Y_{max}$", value=180., key="model_preview_T3PHI_Ymax")
                                model_preview_T3PHI_Ymin    = st.number_input(r"$Y_{min}$", value=-180., key="model_preview_T3PHI_Ymin")
                                

                        fig = generate_model_v2_t3phi_preview(
                            oim, registry, st.session_state.components, data, model_preview_V2_Ymin, model_preview_V2_Ymax, model_preview_T3PHI_Ymin, model_preview_T3PHI_Ymax, 
                        )
                        if fig:
                            safe_pyplot(st, fig, use_container_width=False)
                    else:
                        st.warning("Load OIFITS data first to display this preview.")
        else:
            st.info("Add a component to see the preview.")

    # ── C. Éditeur de composant actif ──────────────────────────────────
    st.markdown("##### B. Component configuration",
                help="Select the component to configure below")

    names = [c['name'] for c in st.session_state.components]
    if not names:
        st.info("Add a component to start configuring.")
        return

    if st.session_state.active_comp_name not in names:
        st.session_state.active_comp_name = names[0]

    colD1, colD2 = st.columns([3, 1])
    with colD1:
        active_name = st.selectbox(
            "Select active component:",
            options=names, index=0,
            label_visibility="collapsed",
        )
        if active_name != st.session_state.active_comp_name:
            st.session_state.active_comp_name = active_name
            st.rerun()
        comp_active = get_comp_by_name(st.session_state.components, active_name)
        if comp_active:
            read_all_widgets(st.session_state.components)

    with colD2:
        if st.session_state.components:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state.components = [
                    c for c in st.session_state.components
                    if c['name'] != active_name
                ]
                remaining = [c['name'] for c in st.session_state.components]
                st.session_state.active_comp_name = (
                    remaining[0] if remaining else None
                )
                st.rerun()

    comp_edit = (
        get_comp_by_name(st.session_state.components,
                         st.session_state.active_comp_name)
        if st.session_state.active_comp_name else None
    )

    if comp_edit is None:
        st.info("Select a component to edit.")
    else:
        render_param_editor(comp_edit)

    # ── Sauvegarde ────────────────────────────────────────────────────
    st.write("##### C. Save model")
    if st.button("✅ Save", type="primary"):
        read_all_widgets(st.session_state.components)
        mname = model_name.strip() or "unnamed_model"
        st.session_state.MODEL[mname] = {
            'components': [
                {
                    'type':           c['type'],
                    'name':           c['name'],
                    'initial_values': c['initial_values'].copy(),
                    'param_ranges':   c['param_ranges'].copy(),
                    'free_params':    c['free_params'].copy(),
                    'interpolators':  c.get('interpolators', {}).copy(),
                }
                for c in st.session_state.components
            ]
        }
        st.success(f"✅ Model « {mname} » saved!")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 – Load CSV model
# ═══════════════════════════════════════════════════════════════════════════

def _render_csv_import() -> None:
    import pandas as pd  # noqa: PLC0415

    registry = get_registry()

    st.markdown("##### 📂 Import a model from a CSV file")

    csv_file = st.file_uploader(
        "Upload a parameter CSV (results table format)",
        type=["csv"],
        key="csv_model_uploader",
        help=(
            "Expected columns: Parameter, Value, Min, Max, Free\n"
            "Parameter format: c{n}_{TypeAbbr}_{param}  e.g.: c1_UD_d"
        ),
    )
    csv_model_name = st.text_input(
        "Name of imported model",
        placeholder="e.g.: csv_model",
        key="csv_model_name",
    )
    do_import = st.button("📥 Import & store", key="btn_csv_import")

    if csv_file is None:
        return

    try:
        csv_df = pd.read_csv(csv_file)
        with st.expander("Preview of loaded CSV", expanded=False):
            st.dataframe(csv_df, use_container_width=True)

        if do_import:
            result, err_msg = parse_csv_to_model(csv_df, registry)
            if result is None:
                st.error(f"❌ CSV import error:\n\n{err_msg}")
            else:
                target_name = csv_model_name.strip() or csv_file.name.replace(".csv", "")
                st.session_state.MODEL[target_name] = result
                st.session_state.components = [
                    dict(c) for c in result['components']
                ]
                st.session_state.active_comp_name = (
                    result['components'][0]['name']
                    if result['components'] else None
                )
                n_comp     = len(result['components'])
                comp_names = ', '.join(c['name'] for c in result['components'])
                st.success(
                    f"✅ Model **{target_name}** successfully imported "
                    f"({n_comp} component{'s' if n_comp > 1 else ''}: {comp_names})"
                )
                st.rerun()
    except Exception as exc:
        st.error(f"Cannot read CSV: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3 – Interpolators
# ═══════════════════════════════════════════════════════════════════════════

def _render_interpolators() -> None:
    oim      = get_oim()
    registry = get_registry()

    st.markdown("##### Configure oimodeler interpolators")
    st.caption(
        "Assign an `oimInterp` interpolator to a parameter of a component "
        "in an existing model. Two types: **Blackbody** (`starWl`) or "
        "**linear** (values by wavelength)."
    )

    if not st.session_state.MODEL:
        st.info("No model available. Create or import a model first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("##### A. Select the parameter to interpolate")
        interp_model_name = st.selectbox(
            "Target model",
            sorted(st.session_state.MODEL.keys()),
            key="interp_model_sel",
        )
        interp_model_data = st.session_state.MODEL[interp_model_name]
        interp_comps      = interp_model_data.get("components", [])

        if not interp_comps:
            st.warning("This model has no components.")
            return

        comp_names_interp = [c["name"] for c in interp_comps]
        interp_comp_name  = st.selectbox(
            "Component", comp_names_interp, key="interp_comp_sel",
        )
        interp_comp = next(c for c in interp_comps if c["name"] == interp_comp_name)

        _comp_type          = interp_comp.get("type", "")
        _params_from_reg    = registry.get(_comp_type, {}).get("params", [])
        _params_from_comp   = interp_comp.get("params", _params_from_reg)
        interp_params_avail = [p for p in _params_from_comp if p not in ("x", "y")]

        interp_param = st.selectbox(
            "Parameter to interpolate", interp_params_avail, key="interp_param_sel",
        )
        cur_interp = interp_comp.get("interpolators", {}).get(interp_param, {})

        # ── Résumé des interpolateurs actifs ──────────────────────────────
        st.markdown("###### Active interpolators on this component")
        interps = interp_comp.get("interpolators", {})
        if not interps:
            st.caption("No interpolator configured.")
        else:
            for p_name, cfg in interps.items():
                if not cfg.get("enabled"):
                    continue
                if cfg["type"] == "blackbody":
                    st.success(
                        f"🌟 **{p_name}** → Blackbody "
                        f"T={cfg['temp']:.0f} K, d={cfg['dist']:.0f} pc, L={cfg['lum']:.2f} L☉"
                    )
                else:
                    st.info(
                        f"📈 **{p_name}** → Linear "
                        f"({len(cfg['wl'])} points, var={cfg['var']})"
                    )
                if st.button(f"🗑️ Remove interpolator {p_name}",
                             key=f"del_interp_{p_name}"):
                    del interp_comp["interpolators"][p_name]
                    st.rerun()


        
    with col2:
        st.write("##### B. Select and set the interpolator type")

        # ── Blackbody ─────────────────────────────────────────────────────
        interp_type = st.radio(
            "Interpolator type",
            ["Blackbody (starWl)", "Linear"],
            index=0 if cur_interp.get("type") != "custom" else 1,
            horizontal=True,
            key="interp_type_radio",
        )

        if interp_type == "Blackbody (starWl)":
            st.markdown("**Blackbody parameters**")
            bb1, bb2, bb3 = st.columns(3)
            with bb1:
                bb_temp = st.number_input(
                    "Temperature (K)", value=float(cur_interp.get("temp", 5000.)),
                    min_value=100., max_value=100000., step=100.,
                    key="interp_bb_temp",
                )
            with bb2:
                bb_dist = st.number_input(
                    "Distance (pc)", value=float(cur_interp.get("dist", 140.)),
                    min_value=1., max_value=1e6, step=10.,
                    key="interp_bb_dist",
                )
            with bb3:
                bb_lum = st.number_input(
                    "Luminosity (L☉)", value=float(cur_interp.get("lum", 1.)),
                    min_value=0.001, max_value=1e6, step=0.1,
                    key="interp_bb_lum",
                )

            if st.button("✅ Apply blackbody interpolator",
                         key="btn_apply_bb", use_container_width=True):
                interp_comp.setdefault("interpolators", {})[interp_param] = {
                    "enabled": True, "type": "blackbody",
                    "temp": bb_temp, "dist": bb_dist, "lum": bb_lum,
                }
                st.success(
                    f"✅ Blackbody interpolator applied to "
                    f"**{interp_comp_name}.{interp_param}** "
                    f"(T={bb_temp:.0f} K, d={bb_dist:.0f} pc, L={bb_lum:.2f} L☉)"
                )
                st.rerun()  # ← FIX 1 : force le rafraîchissement des interpolateurs actifs

        # ── Custom spline ──────────────────────────────────────────────────
        else:
            st.markdown("**Control points (wavelength → value)**")
            st.caption(
                "Enter the wavelengths (µm) and corresponding values. "
                "oimodeler will interpolate between these points."
            )

            existing_wl     = cur_interp.get("wl",     [2e-6, 3e-6, 4e-6, 5e-6])
            existing_val    = cur_interp.get("values", [0.5, 0.8, 0.6, 0.3])
            existing_wl_um  = [w * 1e6 for w in existing_wl]

            n_pts_raw = st.number_input(
                "Number of points", min_value=2, max_value=20,
                value=len(existing_wl_um), step=1, key="interp_n_pts",
            )
            # Widget bounds are cosmetic only — this directly drives a
            # Python loop below (V4's "loop/step count" category).
            n_pts = num(n_pts_raw, 2, 20, "Number of points", integer=True)

            wl_pts  = []
            val_pts = []
            pt_cols = st.columns(min(int(n_pts), 5))
            for i in range(int(n_pts)):
                with pt_cols[i % len(pt_cols)]:
                    st.markdown(f"**pt {i+1}**")
                    wl_i = st.number_input(
                        "λ (µm)",
                        value=float(existing_wl_um[i]) if i < len(existing_wl_um) else float(1 + i),
                        min_value=0.1, max_value=20., step=0.1, format="%.2f",
                        key=f"interp_wl_{i}",
                    )
                    v_i = st.number_input(
                        "value",
                        value=float(existing_val[i]) if i < len(existing_val) else 0.5,
                        format="%.4g", key=f"interp_val_{i}",
                    )
                    wl_pts.append(wl_i * 1e-6)
                    val_pts.append(v_i)

            interp_var = st.selectbox(
                "Interpolation variable", ["wl"], index=0,
                key="interp_var_sel",
                help="Usually 'wl' for spectral dependence.",
            )

            # Aperçu de la spline
            try:
                interp_obj = oim.oimInterp(
                    interp_var,
                    **{interp_var: np.array(wl_pts)},
                    values=np.array(val_pts),
                )
                wl_fine  = np.linspace(min(wl_pts), max(wl_pts), 200)
                val_fine = np.array([interp_obj(w) for w in wl_fine])

                fig_sp, ax_sp = plt.subplots(figsize=(6, 2.5))
                ax_sp.plot(wl_fine * 1e6, val_fine, color='steelblue', lw=2, label='Spline')
                ax_sp.scatter([w * 1e6 for w in wl_pts], val_pts,
                              color='red', zorder=5, label='Control points')
                ax_sp.set_xlabel("λ (µm)")
                ax_sp.set_ylabel(interp_param)
                ax_sp.set_title(f"Interpolation {interp_param}")
                ax_sp.legend(fontsize=8)
                ax_sp.grid(True, alpha=0.3)
                plt.tight_layout()
                safe_pyplot(st, fig_sp, use_container_width=True)
            except Exception as exc:
                st.caption(f"Preview unavailable: {exc}")

            if st.button("✅ Apply custom interpolator",
                         key="btn_apply_custom", use_container_width=True):  # ← FIX 2 : dans le else
                interp_comp.setdefault("interpolators", {})[interp_param] = {
                    "enabled": True, "type": "custom",
                    "var": interp_var, "wl": wl_pts, "values": val_pts,
                }
                st.success(
                    f"✅ Custom interpolator applied to "
                    f"**{interp_comp_name}.{interp_param}** "
                    f"({int(n_pts)} points, var={interp_var})"
                )
                st.rerun()  # ← FIX 1 (bis) : même correction pour le custom


    st.write("##### C. Save as a new model")

    new_interp_name = st.text_input(
        "Save under name",
        value=f"{interp_model_name}_interp",
        key="interp_save_name",
        width=300,
    )
    if st.button("💾 Save model with interpolators",
                    key="btn_save_interp", type="primary",
                    width=300):
        saved = copy.deepcopy(interp_model_data)
        for i, c in enumerate(saved["components"]):
            if c["name"] == interp_comp_name:
                saved["components"][i]["interpolators"] = \
                    interp_comp.get("interpolators", {})
        target = new_interp_name.strip() or f"{interp_model_name}_interp"
        st.session_state.MODEL[target] = saved
        st.success(f"✅ Model **{target}** saved with interpolators!")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 4 – Model summary
# ═══════════════════════════════════════════════════════════════════════════

def _render_model_summary() -> None:
    oim      = get_oim()
    registry = get_registry()

    if not st.session_state.MODEL:
        st.info("No model available. Create or import a model first.")
        return

    data = _get_active_data_with_filter()
    if data is None:
        st.warning("Load OIFITS data first (Data tab).")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        selected = st.selectbox(
            "View a model",
            sorted(st.session_state.MODEL.keys()),
            index=0,
        )
        _model = build_oim_model(
            oim, registry,
            st.session_state.MODEL[selected]["components"],
        )
        if _model is None:
            st.error("Cannot build model.")
            return

        sim = oim.oimSimulator(data=data, model=_model)
        st.write(r"$\chi²$ : " + f"{sim.chi2r:.2f}")
        st.write(sim.model)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.write("$X$ axis")
            x_min = st.number_input("Xmin", key="Xmin_CP", value=1.)
            x_max = st.number_input("Xmax", key="Xmax_CP", value=5.)
        with col4:
            st.write("$V²_Y$")
            vis_y_min = st.number_input("Ymin", key="Ymin_Vis", value=0.)
            vis_y_max = st.number_input("Ymax", key="Ymax_Vis", value=1.)
        with col5:
            st.write("$CP_Y$")
            cp_y_min = st.number_input("Ymin", key="Ymin_CP", value=-180.)
            cp_y_max = st.number_input("Ymax", key="Ymax_CP", value=180.)

    with col2:
        try:
            fig0, ax0 = sim.plot(["VIS2DATA", "T3PHI"])
            ax0[0].set_xlim([x_min * 1e7, x_max * 1e7])
            ax0[0].set_ylim([vis_y_min, vis_y_max])
            ax0[1].set_xlim([x_min * 1e7, x_max * 1e7])
            ax0[1].set_ylim([cp_y_min, cp_y_max])
            safe_pyplot(st, fig0)
        except Exception as exc:
            st.warning(f"Plot error: {exc}")

    try:
        fig1 = sim.plotWlTemplate(
            [["VIS2DATA"], ["T3PHI"]], xunit="micron", figsize=(22, 3)
        )
        fig1.set_legends(0.5, 0.8, "$BASELINE$", ["VIS2DATA", "T3PHI"],
                         fontsize=10, ha="center")
        fig1.axes[0].set_ylim(vis_y_min, vis_y_max)
        fig1.axes[7].set_ylim(cp_y_min, cp_y_max)
        safe_pyplot(st, fig1)
    except Exception as exc:
        st.warning(f"Template plot error: {exc}")

    # ── Visibility vs baseline (East-West / North-South) ────────────────
    st.markdown("##### Visibility vs baseline")
    vb_col1, vb_col2 = st.columns([1, 2])
    with vb_col1:
        vb_bmax_raw = st.number_input(
            "Max baseline (m)", value=200., min_value=1., max_value=1000.,
            key="ms_vb_bmax",
        )
        vb_n_raw = st.number_input(
            "Number of points", value=200, min_value=10, max_value=1000,
            key="ms_vb_n",
        )
        vb_wl_raw = st.number_input(
            "Wavelength (µm)", value=3.5, min_value=0.1, max_value=20.,
            key="ms_vb_wl",
        )
    with vb_col2:
        try:
            # Widget bounds are cosmetic only — vb_n feeds an array
            # allocation directly (V6's OOM DoS category).
            vb_bmax = num(vb_bmax_raw, 1., 1000., "Max baseline")
            vb_n    = num(vb_n_raw, 10, 1000, "Number of points", integer=True)
            vb_wl   = num(vb_wl_raw, 0.1, 20.0, "Wavelength") * 1e-6

            # Same convention as the Component Explorer page: ucoord (East-
            # West) is the Fourier conjugate of the model's x/RA axis,
            # vcoord (North-South) of y/Dec.
            baselines = np.linspace(0., vb_bmax, num=vb_n)
            spf   = baselines / vb_wl
            zeros = np.zeros_like(spf)

            ccf_ew = _model.getComplexCoherentFlux(spf, zeros, wl=vb_wl)
            ccf_ns = _model.getComplexCoherentFlux(zeros, spf, wl=vb_wl)

            v_ew = np.abs(ccf_ew)
            v_ns = np.abs(ccf_ns)
            norm = v_ew[0] if v_ew[0] > 0 else 1.0
            v_ew = v_ew / norm
            v_ns = v_ns / norm

            fig_vb, ax_vb = plt.subplots(figsize=(8, 4))
            ax_vb.plot(baselines, v_ew, label='East–West', color='tab:blue')
            ax_vb.plot(baselines, v_ns, label='North–South', color='tab:orange', ls='--')
            ax_vb.set_xlabel('Baseline length (m)')
            ax_vb.set_ylabel('Normalized visibility')
            ax_vb.set_ylim(-0.02, 1.05)
            ax_vb.set_title(f'{selected}  –  λ = {vb_wl * 1e6:.2f} µm')
            ax_vb.legend()
            ax_vb.grid(alpha=0.3)
            safe_pyplot(st, fig_vb)
        except InvalidInput as exc:
            st.warning(str(exc))
        except Exception:
            logger.exception("Visibility-vs-baseline rendering failed (Model summary)")
            st.warning(
                "Could not render the visibility-vs-baseline plot for "
                "the current settings."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 5 – Model management
# ═══════════════════════════════════════════════════════════════════════════

def _render_model_management() -> None:
    if not st.session_state.MODEL:
        st.warning("No model has been defined yet.")
        return

    liste = sorted(st.session_state.MODEL.keys())
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Rename a model**")
        model_tbr_raw = st.selectbox("Select model to rename", liste, key="model_TBR")
        new_name_raw  = st.text_input("New name", placeholder="new name", key="rename_input")
        if st.button("Rename", type="primary", key="btn_rename"):
            try:
                model_tbr = choice(model_tbr_raw, liste, "Model to rename")
                new_name  = text(new_name_raw, "New name", max_len=64)
            except InvalidInput as exc:
                st.error(str(exc))
            else:
                if new_name.strip():
                    st.session_state.MODEL[new_name] = copy.deepcopy(
                        st.session_state.MODEL.pop(model_tbr)
                    )
                    st.success(f"Model **{model_tbr}** renamed to **{new_name}**")
                else:
                    st.warning("Please enter a name.")

    with col2:
        st.write("**Delete a model**")
        model_tbs_raw = st.selectbox("Select model to delete", liste, key="model_TBS")
        if st.button("Delete", type="primary", key="btn_delete"):
            try:
                model_tbs = choice(model_tbs_raw, liste, "Model to delete")
            except InvalidInput as exc:
                st.error(str(exc))
            else:
                st.session_state.MODEL.pop(model_tbs)
                st.success(f"Model **{model_tbs}** successfully deleted.")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Helper interne
# ═══════════════════════════════════════════════════════════════════════════

def _get_active_data_with_filter():
    """
    Retourne l'objet oimData actif avec le filtre appliqué — chaque fichier
    filtré indépendamment des autres (voir pages/data.py et
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

    oim          = get_oim()
    file_order   = st.session_state.get('selected_files', []) or []
    file_filters = st.session_state.get('file_filters', {})
    file_dtypes  = st.session_state.get('file_dtypes', {})

    filters = build_per_file_filters(file_filters, file_dtypes, file_order)
    data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True

    return data
