# pages/fitting.py
"""
Page "Fitting" – Random search, χ² minimization, Emcee MCMC.

Dépendances :
    services/data_service.py  → get_oim(), get_registry(), load_oifits()
    core/component.py         → ComponentConfig
    core/model_builder.py     → build_oim_model(), decompose_model_flux(),
                                 extract_model_image()
    core/fitting.py           → random_search()
    core/results.py           → get_result_df(), update_model_from_fit()
    core/code_generator.py    → generate_fitting_code()
    components/plots.py       → plot_flux_decomposition(), copy_axes_lines(),
                                 safe_pyplot()
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from services.data_service import get_oim, get_registry, load_oifits,load_oifits_multi
from core.component import ComponentConfig
from core.model_builder import (
    build_oim_model,
    decompose_model_flux,
    extract_model_image,
)
from core.fitting import random_search
from core.results import get_result_df, update_model_from_fit
from core.code_generator import generate_fitting_code
from components.plots import plot_flux_decomposition, copy_axes_lines, safe_pyplot


# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def render() -> None:
    oim      = get_oim()
    registry = get_registry()

    if not st.session_state.MODEL:
        st.warning("⚠️ No model saved. Configure and save a model (Modelling tab).")
        return

    data = _get_active_data_with_filter()
    if data is None:
        st.warning("⚠️ No OIFITS data loaded. Go to the Data tab first.")
        return

    # ── Sélecteurs ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"{len(data.data)} dataset selected:\n\n {st.session_state.test_selected_file}")
    with col2:
        model_options = sorted(st.session_state.MODEL.keys())
        model_to_use  = st.selectbox(
            "Model to use", options=model_options,
            index=len(model_options) - 1, key="fit_model",
        )
    with col3:
        methode = st.selectbox(
            "Method", ["Random", "scipy χ² Minimization", "Emcee"],
            key="fit_method",
        )

    st.markdown("---")

    if methode == "Random":
        _render_random(oim, registry, data, model_to_use)
    elif methode == "scipy χ² Minimization":
        _render_chi2(oim, registry, data, model_to_use)
    else:
        _render_emcee(oim, registry, data, model_to_use)


# ═══════════════════════════════════════════════════════════════════════════
# Random search
# ═══════════════════════════════════════════════════════════════════════════

def _render_random(oim, registry, data, model_to_use: str) -> None:
    st.markdown("##### Random search configuration")
    ca1, ca2 = st.columns(2)
    with ca1:
        n_runs   = st.number_input("Number of iterations", 10, 1000, 100, 10)
        use_seed = st.checkbox("Fixed seed", value=True)
        seed_val = st.number_input("Seed", 0, 99999, 42) if use_seed else None
    with ca2:
        rand_dtypes = st.multiselect(
            "Data to use",
            ["VIS2DATA", "T3PHI", "VISPHI", "T3AMP", "FLUXDATA"],
            default=["VIS2DATA", "T3PHI"],
        )

    model_comps = st.session_state.MODEL[model_to_use]["components"]
    if not model_comps:
        st.warning("The selected model is empty.")
        return

    with st.expander(f"Model summary « {model_to_use} »"):
        for c in model_comps:
            st.write(f"**{c['name']}** ({c['type']}) — free: {', '.join(c['free_params'])}")

    if st.button("🚀 Run random search", type="primary", use_container_width=True):
        configs = [
            ComponentConfig(
                component_type=c['type'], registry=registry, name=c['name'],
                initial_values=c['initial_values'], param_ranges=c['param_ranges'],
                free_params=c['free_params'], interpolators=c.get('interpolators', {}),
            )
            for c in model_comps
        ]
        try:
            data.useFilter = True
            progress_bar = st.progress(0)
            status_box   = st.empty()

            bp, bc, hist = random_search(
                oim, data, configs,
                n_runs=n_runs, seed=seed_val,
                progress_callback=lambda v: progress_bar.progress(v),
                status_callback=lambda s: status_box.success(s),
                warning_callback=lambda w: st.warning(w),
            )
            progress_bar.empty()
            status_box.empty()

            best_comps = [
                cfg.create_instance(oim, bp.get(cfg.name, {}))
                for cfg in configs
            ]
            st.session_state.best_model_comps = [
                {'type': c['type'], 'name': c['name'],
                 'initial_values': bp.get(c['name'], c['initial_values']),
                 'param_ranges': c['param_ranges'],
                 'free_params': c['free_params'],
                 'interpolators': c.get('interpolators', {})}
                for c in model_comps
            ]
            st.session_state.optimization_done = True
            st.session_state.best_chi2         = bc
            st.session_state.history           = hist
            # Stocke l'objet modèle temporairement pour l'affichage
            st.session_state['_random_best_model'] = oim.oimModel(*best_comps)
            st.success("✅ Optimization complete!")
            st.balloons()
        except Exception as exc:
            st.error(f"Error: {exc}")

    if not st.session_state.optimization_done:
        return

    st.markdown("### Random search results")
    st.success(f"Best χ²ᵣ: **{st.session_state.best_chi2:.4f}**")

    best_model = st.session_state.get('_random_best_model')
    if best_model:
        _, tbl = get_result_df(best_model, is_fit=False)
        st.dataframe(tbl, use_container_width=True)

        if st.button("💾 Save this best model", use_container_width=True,
                     key="save_random"):
            update_model_from_fit(
                f"Best_Random_{model_to_use}", model_to_use,
                best_model, chi2r=st.session_state.best_chi2,
            )
            st.success(f"Model **Best_Random_{model_to_use}** saved!")

    # ── Graphiques d'historique ───────────────────────────────────────
    st.markdown("##### History")
    runs  = [r['run']   for r in st.session_state.history]
    chi2s = [r['chi2r'] for r in st.session_state.history]

    cummin, cur = [], float('inf')
    for v in chi2s:
        cur = min(cur, v)
        cummin.append(cur)

    gh1, gh2 = st.columns(2)
    with gh1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.scatter(runs, chi2s, alpha=0.4, s=15, label='All runs')
        ax1.plot(runs, cummin, 'r-', lw=2, label='Best χ²ᵣ')
        ax1.set_xlabel('Run'); ax1.set_ylabel('χ²ᵣ')
        ax1.set_title('Evolution'); ax1.legend(); ax1.grid(alpha=.3)
        safe_pyplot(st, fig1)

    with gh2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.hist(chi2s, bins=30, alpha=.7, edgecolor='black')
        ax2.axvline(min(chi2s), color='r', ls='--', lw=2,
                    label=f'Min = {min(chi2s):.4f}')
        ax2.set_xlabel('χ²ᵣ'); ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution'); ax2.legend(); ax2.grid(alpha=.3)
        safe_pyplot(st, fig2)


# ═══════════════════════════════════════════════════════════════════════════
# scipy χ² Minimization
# ═══════════════════════════════════════════════════════════════════════════

def _render_chi2(oim, registry, data, model_to_use: str) -> None:
    st.markdown("### scipy χ² Minimization")
    opt_dtypes = st.multiselect(
        "Data to fit", ["VIS2DATA", "T3PHI", "FLUXDATA"],
        default=["VIS2DATA", "T3PHI"], key="chi2_dtypes",
    )

    model_chi2 = build_oim_model(oim, registry,
                                 st.session_state.MODEL[model_to_use]["components"])
    if model_chi2 is None:
        st.error("Cannot build model.")
        return

    if st.button("▶️ Run", type="primary"):
        data.useFilter = True
        try:
            model_init = copy.deepcopy(model_chi2)
            sim_init   = oim.oimSimulator(data=data, model=model_init)
            sim_init.compute(computeChi2=True, computeSimulatedData=False)
            chi2_init  = sim_init.chi2r

            lmfit = oim.oimFitterMinimize(data, model_chi2, dataTypes=opt_dtypes)
            lmfit.prepare()
            lmfit.run()
            st.balloons()

            st.session_state.chi2_result = {
                'model_initial':   model_init,
                'best_chi2_model': lmfit.simulator.model,
                'chi2_init':       chi2_init,
                'chi2_final':      lmfit.simulator.chi2r,
                'lmfit':           lmfit,
                'model_to_use':    model_to_use,
                'dtypes':          opt_dtypes,
            }
        except Exception as exc:
            st.error(f"Minimization error: {exc}")

    if st.session_state.chi2_result is None:
        return

    r = st.session_state.chi2_result

    if r['chi2_final'] > r['chi2_init']:
        st.warning(f"⚠️ Divergence: {r['chi2_init']:.2f} → {r['chi2_final']:.2f}")
    else:
        st.success(f"✅ χ²ᵣ: {r['chi2_init']:.2f} → {r['chi2_final']:.2f}")

    # ── Tableaux avant/après ──────────────────────────────────────────
    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown(f"**Before** (χ²ᵣ = {r['chi2_init']:.2f})")
        _, tbl1 = get_result_df(r['model_initial'], is_fit=False)
        st.dataframe(tbl1, use_container_width=True)
    with cr2:
        st.markdown(f"**After** (χ²ᵣ = {r['chi2_final']:.2f})")
        _, tbl2 = get_result_df(r['best_chi2_model'], is_fit=False)
        st.dataframe(tbl2, use_container_width=True)

    # ── Figure 4 panneaux ─────────────────────────────────────────────
    try:
        data.useFilter = True
        decomp = decompose_model_flux(oim, r['best_chi2_model'], data)

        ax_v2   = r['lmfit'].simulator.plotWithResiduals(
            ["VIS2DATA"], xunit="cycle/mas",
            kwargsData=dict(color="byBaseline"))[1]
        ax_t3   = r['lmfit'].simulator.plotWithResiduals(
            ["T3PHI"], xunit="cycle/mas",
            kwargsData=dict(color="byBaseline"))[1]
        d_img   = extract_model_image(oim, r['best_chi2_model'])
        fig_flux = plot_flux_decomposition(decomp, data)
        ax_flux_src = fig_flux.axes[0]
        plt.close('all')

        fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5))

        copy_axes_lines(ax_flux_src, axes_cmp[0])
        axes_cmp[0].set_title("FLUXDATA / components")
        handles, labels = axes_cmp[0].get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        axes_cmp[0].legend(seen.values(), seen.keys(), fontsize=7)

        copy_axes_lines(ax_v2[0], axes_cmp[1])
        axes_cmp[1].set_title("VIS²")

        copy_axes_lines(ax_t3[0], axes_cmp[2])
        axes_cmp[2].set_title("T3PHI")

        axes_cmp[3].imshow(d_img[0, 0] ** 0.2, cmap='hot', origin='lower')
        axes_cmp[3].set_title("Model (γ=0.2)")

        plt.tight_layout()
        safe_pyplot(st, fig_cmp, use_container_width=True)
        plt.close(fig_flux)

    except Exception as exc:
        st.warning(f"Cannot display comparison: {exc}")

    # ── Code reproductible ────────────────────────────────────────────
    with st.expander("Reproducible Python code", expanded=False):
        code = generate_fitting_code(
            method="chi2",
            result={"dtypes": r['dtypes']},
            data_filename=st.session_state.get("fit_dataset", "data.fits"),
            model_comps=st.session_state.MODEL[r["model_to_use"]]["components"],
            filter_params=_get_filter_params(),
            registry=registry,
        )
        st.code(code, language="python")

    if st.button("💾 Save best χ² model", use_container_width=True,
                 key="save_chi2", type="primary"):
        update_model_from_fit(
            f"Best_Chi2r_{r['model_to_use']}", r['model_to_use'],
            r['best_chi2_model'], chi2r=r['chi2_final'],
        )
        st.success(f"Model **Best_Chi2r_{r['model_to_use']}** saved!")


# ═══════════════════════════════════════════════════════════════════════════
# Emcee
# ═══════════════════════════════════════════════════════════════════════════

def _render_emcee(oim, registry, data, model_to_use: str) -> None:
    st.markdown("### Emcee MCMC")

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        emcee_dtypes = st.multiselect(
            "Data to fit", ["VIS2DATA", "T3PHI", "FLUXDATA"],
            default=["VIS2DATA", "T3PHI"], key="emcee_dtypes",
        )
    with ec2:
        nb_walkers = st.number_input("Walkers", 1, 64, 32, key="emcee_walkers")
    with ec3:
        nb_steps = st.number_input("Steps", 0, 40000, 1000, key="emcee_steps")
    with ec4:
        init_mode = st.selectbox(
            "Init", ['random', 'gaussian', ""], index=0, key="emcee_init",
        )

    model_emcee = build_oim_model(
        oim, registry,
        st.session_state.MODEL[model_to_use]["components"],
    )
    if model_emcee is None:
        st.error("Cannot build model.")
        return

    if st.button("▶️ Run Emcee", type="primary"):
        data.useFilter = True
        try:
            model_init = copy.deepcopy(model_emcee)
            sim_init   = oim.oimSimulator(data=data, model=model_init)
            sim_init.compute(computeChi2=True, computeSimulatedData=False)
            chi2_init  = sim_init.chi2r

            emfit = oim.oimFitterEmcee(
                data, model_emcee,
                nwalkers=nb_walkers, dataTypes=emcee_dtypes,
            )
            sampler_path = Path("/tmp/sampler_emcee.txt")
            sampler_path.unlink(missing_ok=True)
            emfit.prepare(init=init_mode, samplerFile=str(sampler_path))

            with st.spinner("MCMC running …", show_time=True):
                emfit.run(nsteps=nb_steps, progress=True)

            st.session_state.emcee_result = {
                'model_initial':    model_init,
                'best_emcee_model': emfit.simulator.model,
                'chi2_init':        chi2_init,
                'chi2_final':       emfit.simulator.chi2r,
                'lmfit':            emfit,
                'model_to_use':     model_to_use,
                'dtypes':           emcee_dtypes,
                'nwalkers':         nb_walkers,
                'nsteps':           nb_steps,
                'init':             init_mode,
            }
            st.success("✅ Emcee complete!")
            st.balloons()
        except Exception as exc:
            st.error(f"Emcee error: {exc}")

    if st.session_state.emcee_result is None:
        return

# ── Tab Code & Save ───────────────────────────────────────────────────

    er = st.session_state.emcee_result

    st.markdown(f"χ²ᵣ: **{er['chi2_init']:.2f}** → **{er['chi2_final']:.2f}**")
    st.markdown("##### Fitted parameters")
    _, tbl_em = get_result_df(er['best_emcee_model'], is_fit=False)
    st.dataframe(tbl_em, use_container_width=True, height=350)

    if st.button("💾 Save best Emcee model", use_container_width=True,
                 key="save_emcee"):
        update_model_from_fit(
            f"Best_Emcee_{er['model_to_use']}", er['model_to_use'],
            er['best_emcee_model'], chi2r=er['chi2_final'],
        )
        st.success(f"Model **Best_Emcee_{er['model_to_use']}** saved!")

    # ── Code reproductible ────────────────────────────────────────────
    with st.expander("Reproducible Python code", expanded=False):
        code = generate_fitting_code(
            method="emcee",
            result={
                "dtypes":   er['dtypes'],
                "nwalkers": er['nwalkers'],
                "nsteps":   er['nsteps'],
                "init":     er['init'],
            },
            data_filename=st.session_state.get("fit_dataset", "data.fits"),
            model_comps=st.session_state.MODEL[er["model_to_use"]]["components"],
            filter_params=_get_filter_params(),
            registry=registry,
        )
        st.code(code, language="python")


# ── Résultats ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Results display")

    tabs = st.tabs(["VIS² / T3PHI", "Model image", "FLUXDATA / components", "Walkers", "Corner plot"])
    tab_vis, tab_img, tab_flux, tab_walk, tab_corner = tabs

    # ── VIS² / T3PHI ──────────────────────────────────────────────────
    with tab_vis:
        col_p, col_g = st.columns([1, 2])
        with col_p:
            st.write("$X$ axis")
            vt_xs  = st.selectbox("X scale", ["linear", "log"], key="em_vt_xs")
            vt_xmn = st.number_input("Xmin (cycle/mas)", value=0., key="em_vt_xmn")
            vt_xmx = st.number_input("Xmax (cycle/mas)", value=5., key="em_vt_xmx")
            st.write("$V^2$")
            vt_v2_ys  = st.selectbox("Y scale VIS²", ["linear", "log"], key="em_vt_v2_ys")
            vt_v2_ymn = st.number_input("Ymin VIS²", value=0., key="em_vt_v2_ymn")
            vt_v2_ymx = st.number_input("Ymax VIS²", value=1., key="em_vt_v2_ymx")
            st.write("$T3PHI$")
            vt_t3_ymn = st.number_input("Ymin T3PHI (°)", value=-15., key="em_vt_t3_ymn")
            vt_t3_ymx = st.number_input("Ymax T3PHI (°)", value=15.,  key="em_vt_t3_ymx")
        with col_g:
            try:
                sim_plot = oim.oimSimulator(data=data, model=er['best_emcee_model'])
                sim_plot.compute(computeChi2=False, computeSimulatedData=True)
                fig_0, ax_0 = sim_plot.plot(["VIS2DATA", "T3PHI"])

                ax_0[0].set_xscale(vt_xs); ax_0[0].set_yscale(vt_v2_ys)
                ax_0[0].set_xlim(vt_xmn * 1e7, vt_xmx * 1e7)
                ax_0[0].set_ylim(vt_v2_ymn, vt_v2_ymx)
                ax_0[0].set_title("VIS²"); ax_0[0].grid(True, alpha=0.3)

                ax_0[1].set_xscale(vt_xs)
                ax_0[1].set_xlim(vt_xmn * 1e7, vt_xmx * 1e7)
                ax_0[1].set_ylim(vt_t3_ymn, vt_t3_ymx)
                ax_0[1].set_title("T3PHI"); ax_0[1].grid(True, alpha=0.3)

                safe_pyplot(st, fig_0, use_container_width=True)
            except Exception as exc:
                st.warning(f"VIS²/T3PHI: {exc}")

    # ── MODEL IMAGE ───────────────────────────────────────────────────
    with tab_img:
        col_p, col_g = st.columns([1, 2])
        with col_p:
            img_gamma = st.slider("Gamma γ", 0.05, 1.0, 0.2, 0.05, key="em_img_gamma")
            img_cmap  = st.selectbox(
                "Colormap", ["hot", "inferno", "viridis", "plasma", "gray", "afmhot"],
                key="em_img_cmap",
            )
            img_size  = st.number_input("Image size (px)", 64, 512, 128,
                                        step=64, key="em_img_size")
            img_scale = st.number_input("Scale (mas/px)", 0.1, 10., 1.,
                                        step=0.1, key="em_img_scale")
            use_wl    = st.checkbox("Filter on λ", value=False, key="em_img_use_wl")
            wl_val    = None
            if use_wl:
                wl_val = st.number_input("λ (µm)", value=3.5, step=0.1,
                                         key="em_img_wl") * 1e-6
        with col_g:
            try:
                img_data    = extract_model_image(oim, er['best_emcee_model'],
                                                  img_size=int(img_size),
                                                  img_scale=float(img_scale),
                                                  wl_value=wl_val)
                display_img = img_data[0, 0] ** img_gamma
                extent_half = img_size * img_scale / 2

                fig_img, ax_img = plt.subplots(figsize=(5, 5))
                im_plot = ax_img.imshow(
                    display_img, cmap=img_cmap, origin='lower',
                    extent=[-extent_half, extent_half, -extent_half, extent_half],
                )
                plt.colorbar(im_plot, ax=ax_img, label=f'Intensity (γ={img_gamma})')
                ax_img.set_xlabel("ΔRA (mas)"); ax_img.set_ylabel("ΔDec (mas)")
                wl_label = f" @ {wl_val*1e6:.2f} µm" if wl_val else ""
                ax_img.set_title(f"Model{wl_label}")
                safe_pyplot(st, fig_img, use_container_width=True)
            except Exception as exc:
                st.warning(f"Image: {exc}")

    # ── FLUXDATA / components ─────────────────────────────────────────
    with tab_flux:
        st.caption("No adjustable parameters.")
        try:
            data.useFilter = True
            decomp_em = decompose_model_flux(oim, er['best_emcee_model'], data)
            fig_flux_em = plot_flux_decomposition(decomp_em, data)
            safe_pyplot(st, fig_flux_em, use_container_width=True)
        except Exception as exc:
            st.warning(f"FLUXDATA: {exc}")

    # ── WALKERS ───────────────────────────────────────────────────────
    with tab_walk:
        st.caption("No adjustable parameters.")
        try:
            fw, _ = er['lmfit'].walkersPlot(chi2limfact=5)
            safe_pyplot(st, fw, use_container_width=True)
        except Exception as exc:
            st.warning(f"Walkers: {exc}")

    # ── CORNER PLOT ───────────────────────────────────────────────────
    with tab_corner:
        st.caption("No adjustable parameters.")
        try:
            fc, _ = er['lmfit'].cornerPlot(dchi2limfact=5)
            safe_pyplot(st, fc, use_container_width=True)
        except Exception as exc:
            st.warning(f"Corner: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers internes
# ═══════════════════════════════════════════════════════════════════════════

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

def _get_active_data():
    """Retourne l'objet oimData actif avec filtre, ou None si indisponible."""
    oim      = get_oim()
    filepath = st.session_state.loaded_files.get(
        st.session_state.get('selected_file')
    )
    if not filepath:
        return None
    try:
        data   = load_oifits(filepath)
        expr   = st.session_state.get('filter_expr', '')
        bin_L  = st.session_state.get('filter_bin_L', 1)
        bin_N  = st.session_state.get('filter_bin_N', 1)
        norm_L = st.session_state.get('filter_norm_L', False)
        norm_N = st.session_state.get('filter_norm_N', False)

        filters = []
        if expr:
            filters.append(
                oim.oimFlagWithExpressionFilter(expr=expr, keepOldFlag=False)
            )
        filters.append(
            oim.oimWavelengthBinningFilter(targets=0, bin=bin_L, normalizeError=norm_L)
        )
        filters.append(
            oim.oimWavelengthBinningFilter(targets=0, bin=bin_N, normalizeError=norm_N)
        )
        data.setFilter(oim.oimDataFilter(filters))
        data.useFilter = True
        return data
    except Exception:
        return None

def _get_filter_params() -> dict:
    return {
        "expr":   st.session_state.get("filter_expr", ""),
        "bin_L":  st.session_state.get("filter_bin_L", 1),
        "bin_N":  st.session_state.get("filter_bin_N", 1),
        "norm_L": st.session_state.get("filter_norm_L", False),
        "norm_N": st.session_state.get("filter_norm_N", False),
    }

