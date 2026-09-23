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
import logging
import re
import uuid

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import streamlit as st

from services.data_service import get_oim, get_registry, get_active_data
from services.activity_log import log_event, get_log_text
from services.storage import session_dir
from config.constants import (
    FITTABLE_DATA_TYPES, MAX_EMCEE_WALKERS, MAX_EMCEE_STEPS,
    MAX_GRID_AXIS_POINTS, MAX_GRID_POINTS,
)
from core.component import ComponentConfig
from core.model_builder import (
    build_oim_model,
    decompose_model_flux,
    extract_model_image,
    apply_normalizations,
)
from core.fitting import (
    random_search,
    run_grid_search_with_progress,
    run_emcee_with_progress,
)
from core.results import get_result_df, update_model_from_fit, build_results_zip
from core.code_generator import generate_fitting_code
from core.model_export import model_to_txt
from core.validation import num, choice, choices, InvalidInput
from components.plots import plot_flux_decomposition, copy_axes_lines, safe_pyplot

logger = logging.getLogger(__name__)

# Sensible default Y-axis range per FITTABLE_DATA_TYPES entry, used when
# generating a per-data-type results plot (Emcee's "Data vs model" tab).
# None means "let matplotlib auto-scale" — FLUXDATA's scale varies too
# much across datasets/instruments to guess a shared default.
_DEFAULT_Y_RANGE: dict[str, tuple[float, float] | None] = {
    "VIS2DATA": (0.0, 1.0),
    "VISAMP":   (0.0, 1.0),
    "T3AMP":    (0.0, 1.0),
    "VISPHI":   (-15.0, 15.0),
    "T3PHI":    (-15.0, 15.0),
    "FLUXDATA": None,
}


# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def render() -> None:
    oim      = get_oim()
    registry = get_registry()

    if not st.session_state.MODEL:
        st.warning("⚠️ No model saved. Configure and save a model (Modelling tab).")
        return

    try:
        data = get_active_data(st.session_state.get('selected_files', []))
    except ValueError:
        # get_active_data() raises rather than returning None on an empty
        # selection — this used to be an unguarded call whose "if data is
        # None" check below could never actually run.
        st.warning("⚠️ No OIFITS data loaded. Go to the Data tab first.")
        return

    # ── Data — Model — Fit method ───────────────────────────────────────
    st.markdown("### A — Data · Model · Fit method")

    _render_dataset_summary()

    col1, col2 = st.columns(2)
    with col1:
        model_options = sorted(st.session_state.MODEL.keys())
        model_to_use_raw = st.selectbox(
            "Model to use", options=model_options,
            index=len(model_options) - 1, key="fit_model",
        )
    with col2:
        method_options = ["Random", "scipy χ² Minimization", "Grid search", "Emcee"]
        methode_raw = st.selectbox(
            "Method", method_options,
            index=method_options.index("Emcee"), key="fit_method",
        )

    try:
        # selectbox returns an unrecognized client value as-is — both
        # feed a dict/branch lookup right below (V4).
        model_to_use = choice(model_to_use_raw, model_options, "Model to use")
        methode      = choice(methode_raw, method_options, "Method")
    except InvalidInput as exc:
        st.error(str(exc))
        return

    _render_model_summary(model_to_use)

    st.markdown("---")

    if methode == "Random":
        _render_random(oim, registry, data, model_to_use)
    elif methode == "scipy χ² Minimization":
        _render_chi2(oim, registry, data, model_to_use)
    elif methode == "Grid search":
        _render_grid(oim, registry, data, model_to_use)
    else:
        _render_emcee(oim, registry, data, model_to_use)


# ═══════════════════════════════════════════════════════════════════════════
# Random search
# ═══════════════════════════════════════════════════════════════════════════

def _render_random(oim, registry, data, model_to_use: str) -> None:
    st.markdown("##### B — Random search configuration")
    ca1, ca2 = st.columns(2)
    with ca1:
        n_runs_raw   = st.number_input("Number of iterations", 10, 1000, 100, 10)
        use_seed     = st.checkbox("Fixed seed", value=True)
        seed_val_raw = st.number_input("Seed", 0, 99999, 42) if use_seed else None
    with ca2:
        rand_dtypes_raw = st.multiselect(
            "Data to use",
            FITTABLE_DATA_TYPES,
            default=["VIS2DATA", "T3PHI"],
        )

    try:
        # Widget bounds are cosmetic only — n_runs directly drives the
        # random_search() loop count (V4's "loop/step count" category).
        n_runs   = num(n_runs_raw, 10, 1000, "Number of iterations", integer=True)
        seed_val = num(seed_val_raw, 0, 99999, "Seed", integer=True) if use_seed else None
        rand_dtypes = choices(rand_dtypes_raw, FITTABLE_DATA_TYPES, "Data to use")
    except InvalidInput as exc:
        st.warning(str(exc))
        return

    model_comps = st.session_state.MODEL[model_to_use]["components"]
    if not model_comps:
        st.warning("The selected model is empty.")
        return

    if st.button("🚀 Run random search", type="primary", use_container_width=True):
        log_event(
            "Fit run started",
            f"Random model={model_to_use} n_runs={n_runs} dtypes={','.join(rand_dtypes)}",
        )
        configs = [
            ComponentConfig(
                component_type=c['type'], registry=registry, name=c['name'],
                initial_values=c['initial_values'], param_ranges=c['param_ranges'],
                free_params=c['free_params'], interpolators=c.get('interpolators', {}),
                normalizations=c.get('normalizations', {}),
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
            apply_normalizations(oim, configs, best_comps)
            st.session_state.best_model_comps = [
                {'type': c['type'], 'name': c['name'],
                 'initial_values': bp.get(c['name'], c['initial_values']),
                 'param_ranges': c['param_ranges'],
                 'free_params': c['free_params'],
                 'interpolators': c.get('interpolators', {}),
                 'normalizations': c.get('normalizations', {})}
                for c in model_comps
            ]
            st.session_state.optimization_done = True
            st.session_state.best_chi2         = bc
            st.session_state.history           = hist
            # Stocke l'objet modèle temporairement pour l'affichage
            st.session_state['_random_best_model'] = oim.oimModel(*best_comps)
            st.success("✅ Optimization complete!")
            log_event("Fit run completed", f"Random best_chi2r={bc:.4f}")
            st.balloons()
        except Exception as exc:
            st.error(f"Error: {exc}")
            log_event("Fit run failed", f"Random: {exc}")

    if not st.session_state.optimization_done:
        return

    st.markdown("### C — Random search results")
    st.success(f"Best χ²ᵣ: **{st.session_state.best_chi2:.4f}**")

    best_model = st.session_state.get('_random_best_model')
    if best_model:
        _, tbl = get_result_df(best_model, is_fit=False)
        st.dataframe(tbl, use_container_width=True)

        if st.button("💾 Save this best model", type="primary", use_container_width=True,
                     key="save_random"):
            update_model_from_fit(
                f"Best_Random_{model_to_use}", model_to_use,
                best_model, chi2r=st.session_state.best_chi2,
            )
            st.success(f"Model **Best_Random_{model_to_use}** saved!")
            log_event("Best model saved", f"Best_Random_{model_to_use}")

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
    st.markdown("### B — scipy χ² Minimization")
    opt_dtypes_raw = st.multiselect(
        "Data to fit", FITTABLE_DATA_TYPES,
        default=["VIS2DATA", "T3PHI"], key="chi2_dtypes",
    )
    try:
        opt_dtypes = choices(opt_dtypes_raw, FITTABLE_DATA_TYPES, "Data to fit")
    except InvalidInput as exc:
        st.warning(str(exc))
        return

    model_chi2 = build_oim_model(oim, registry,
                                 st.session_state.MODEL[model_to_use]["components"])
    if model_chi2 is None:
        st.error("Cannot build model.")
        return

    if st.button("▶️ Run", type="primary", use_container_width=True):
        log_event(
            "Fit run started",
            f"chi2 model={model_to_use} dtypes={','.join(opt_dtypes)}",
        )
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
            log_event(
                "Fit run completed",
                f"chi2 chi2r={chi2_init:.4f}->{lmfit.simulator.chi2r:.4f}",
            )
        except Exception as exc:
            st.error(f"Minimization error: {exc}")
            log_event("Fit run failed", f"chi2: {exc}")

    # ── Code reproductible — right after Run, visible without a click,
    # so it can be copied whether or not the fit has been run yet ───────
    with st.expander("Reproducible Python code", expanded=True):
        st.code(
            generate_fitting_code(
                method="chi2",
                result={"dtypes": opt_dtypes},
                data_filenames=st.session_state.get("selected_files", []),
                model_comps=st.session_state.MODEL[model_to_use]["components"],
                applied_filters=st.session_state.get("applied_filters", []),
                registry=registry,
            ),
            language="python",
        )

    if st.session_state.chi2_result is None:
        return

    r = st.session_state.chi2_result

    st.markdown("### C — Results")
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
    with st.expander("Model image parameters", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            chi2_clip_lo = st.number_input("Model image colormap percentile min", 0., 100., 0.,
                                           key="chi2_img_clip_lo")
        with cc2:
            chi2_clip_hi = st.number_input("Model image colormap percentile max", 0., 100., 100.,
                                           key="chi2_img_clip_hi")
    # Widget bounds aren't server-enforced — re-validate before use.
    chi2_clip_lo, chi2_clip_hi = sorted((
        min(max(float(chi2_clip_lo), 0.), 100.),
        min(max(float(chi2_clip_hi), 0.), 100.),
    ))

    try:
        data.useFilter = True
        decomp = decompose_model_flux(oim, r['best_chi2_model'], data)

        # One panel per data type actually used for this fit (r['dtypes'])
        # — previously hardcoded to VIS2DATA + T3PHI regardless of what
        # "Data to fit" was actually set to, so e.g. a VISAMP-only fit
        # still only ever showed VIS²/T3PHI (neither part of the fit).
        dtype_axes_src = [
            (dtype, r['lmfit'].simulator.plotWithResiduals(
                [dtype], xunit="cycle/mas", kwargsData=dict(color="byBaseline"))[1][0])
            for dtype in r['dtypes']
        ]
        d_img   = extract_model_image(oim, r['best_chi2_model'])
        fig_flux = plot_flux_decomposition(decomp, data)
        ax_flux_src = fig_flux.axes[0]
        plt.close('all')

        n_panels = 2 + len(dtype_axes_src)  # FLUXDATA + one per dtype + model image
        fig_cmp, axes_cmp = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

        copy_axes_lines(ax_flux_src, axes_cmp[0])
        axes_cmp[0].set_title("FLUXDATA / components")
        handles, labels = axes_cmp[0].get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        axes_cmp[0].legend(seen.values(), seen.keys(), fontsize=7)

        for i, (dtype, ax_src) in enumerate(dtype_axes_src, start=1):
            copy_axes_lines(ax_src, axes_cmp[i])
            axes_cmp[i].set_title(dtype)

        # Astronomical convention: RA increases to the left (East left).
        d_extent_half = d_img.shape[-1] * 0.05 / 2  # matches extract_model_image() default img_scale
        d_img_disp = d_img[0, 0] ** 1.0
        d_vmin, d_vmax = np.percentile(d_img_disp, [chi2_clip_lo, chi2_clip_hi])
        axes_cmp[-1].imshow(
            d_img_disp, cmap='hot', origin='lower',
            extent=[d_extent_half, -d_extent_half, -d_extent_half, d_extent_half],
            vmin=d_vmin, vmax=d_vmax,
        )
        axes_cmp[-1].set_xlabel("ΔRA (mas)")
        axes_cmp[-1].set_ylabel("ΔDec (mas)")
        axes_cmp[-1].set_title("Model")

        plt.tight_layout()
        safe_pyplot(st, fig_cmp, use_container_width=True)
        plt.close(fig_flux)

    except Exception as exc:
        st.warning(f"Cannot display comparison: {exc}")

    if st.button("💾 Save best χ² model", use_container_width=True,
                 key="save_chi2", type="primary"):
        update_model_from_fit(
            f"Best_Chi2r_{r['model_to_use']}", r['model_to_use'],
            r['best_chi2_model'], chi2r=r['chi2_final'],
        )
        st.success(f"Model **Best_Chi2r_{r['model_to_use']}** saved!")
        log_event("Best model saved", f"Best_Chi2r_{r['model_to_use']}")


# ═══════════════════════════════════════════════════════════════════════════
# Grid search — regular grid χ² exploration
# https://oimodeler.readthedocs.io/en/latest/fitter.html#regular-grid-exploration
# ═══════════════════════════════════════════════════════════════════════════

def _render_grid(oim, registry, data, model_to_use: str) -> None:
    st.markdown("### B — Grid search (χ² exploration)")

    grid_dtypes_raw = st.multiselect(
        "Data to fit", FITTABLE_DATA_TYPES,
        default=["VIS2DATA", "T3PHI"], key="grid_dtypes",
    )
    try:
        grid_dtypes = choices(grid_dtypes_raw, FITTABLE_DATA_TYPES, "Data to fit")
    except InvalidInput as exc:
        st.warning(str(exc))
        return

    model_comps = st.session_state.MODEL[model_to_use]["components"]
    if not model_comps:
        st.warning("The selected model is empty.")
        return

    model_grid = build_oim_model(oim, registry, model_comps)
    if model_grid is None:
        st.error("Cannot build model.")
        return

    free_params = model_grid.getFreeParameters()
    if not free_params:
        st.warning("The current model has no free parameters to explore.")
        return
    free_names = sorted(free_params.keys())

    ndim_options = [1, 2] if len(free_names) >= 2 else [1]
    ndim_raw = st.radio(
        "Grid dimensions", ndim_options, horizontal=True, key="grid_ndim",
        help="oimodeler's regular grid explorer supports any dimensionality; "
             "this app limits it to 1D/2D to keep a single grid search bounded.",
    )
    try:
        ndim = choice(ndim_raw, tuple(ndim_options), "Grid dimensions")
    except InvalidInput as exc:
        st.warning(str(exc))
        return
    if ndim == 2 and len(free_names) < 2:
        st.warning(
            "The current model has only one free parameter — a 2D grid "
            "needs at least two."
        )
        return

    axis_raw = []
    gcols = st.columns(ndim)
    for i in range(ndim):
        with gcols[i]:
            st.markdown(f"**Axis {i + 1}**")
            # Default each axis to a distinct free parameter (axis 2 to the
            # 2nd one, etc.) — st.selectbox otherwise defaults every axis to
            # index 0, so two axes would start on the same parameter and
            # immediately fail the "must use different parameters" check
            # below with no obvious way to notice why the button is gone.
            name_raw = st.selectbox(
                "Parameter", free_names, key=f"grid_param_{i}",
                index=min(i, len(free_names) - 1),
            )
            p = free_params.get(name_raw)
            lo_default = float(p.min) if p is not None and p.min is not None else 0.0
            hi_default = float(p.max) if p is not None and p.max is not None else 1.0
            lo_raw = st.number_input("Min", value=lo_default, key=f"grid_lo_{i}")
            hi_raw = st.number_input("Max", value=hi_default, key=f"grid_hi_{i}")
            n_raw  = st.number_input(
                "Grid points", 2, MAX_GRID_AXIS_POINTS, 20, key=f"grid_n_{i}",
            )
            axis_raw.append({'name_raw': name_raw, 'lo_raw': lo_raw,
                             'hi_raw': hi_raw, 'n_raw': n_raw})

    try:
        # Widget bounds are cosmetic only — re-validate before use (V4).
        # A grid point costs one simulator.compute() call, so the *total*
        # (product across axes) is what's actually bounded, not just each
        # axis independently (V6/V7's DoS-by-large-computation category).
        axes = []
        total_points = 1
        for i, a in enumerate(axis_raw):
            name = choice(a['name_raw'], free_names, f"Axis {i + 1} parameter")
            lo   = num(a['lo_raw'], -1e12, 1e12, f"Axis {i + 1} min")
            hi   = num(a['hi_raw'], -1e12, 1e12, f"Axis {i + 1} max")
            if lo >= hi:
                raise InvalidInput(f"Axis {i + 1}: min must be smaller than max.")
            n = num(a['n_raw'], 2, MAX_GRID_AXIS_POINTS, f"Axis {i + 1} grid points",
                    integer=True)
            total_points *= n
            axes.append({'name': name, 'lo': lo, 'hi': hi, 'n': n})

        if len({a['name'] for a in axes}) != len(axes):
            raise InvalidInput("Grid axes must use different parameters.")
        if total_points > MAX_GRID_POINTS:
            raise InvalidInput(
                f"Grid too large: {total_points} points requested "
                f"(max {MAX_GRID_POINTS}). Reduce the grid points per axis."
            )
    except InvalidInput as exc:
        st.warning(str(exc))
        return

    st.caption(
        f"Grid size: {total_points} point"
        f"{'s' if total_points != 1 else ''} "
        f"({' × '.join(str(a['n']) for a in axes)})"
    )

    if st.button("🧮 Run grid search", type="primary", use_container_width=True):
        log_event(
            "Fit run started",
            f"grid model={model_to_use} dtypes={','.join(grid_dtypes)} "
            f"axes={[a['name'] for a in axes]} size={total_points}",
        )
        data.useFilter = True
        try:
            model_init = copy.deepcopy(model_grid)
            sim_init   = oim.oimSimulator(data=data, model=model_init)
            sim_init.compute(computeChi2=True, computeSimulatedData=False)
            chi2_init  = sim_init.chi2r

            gfit = oim.oimFitterRegularGrid(data, model_grid, dataTypes=grid_dtypes)
            grid_param_objs = [free_params[a['name']] for a in axes]
            gfit.prepare(
                params=grid_param_objs,
                min=[a['lo'] for a in axes],
                max=[a['hi'] for a in axes],
                steps=[(a['hi'] - a['lo']) / (a['n'] - 1) for a in axes],
            )
            progress_bar = st.progress(0)
            status_box   = st.empty()
            status_box.info("Grid search running …")
            run_grid_search_with_progress(
                gfit, progress_callback=lambda v: progress_bar.progress(v),
            )
            progress_bar.empty()
            status_box.empty()

            st.session_state.grid_result = {
                'model_initial':   model_init,
                'best_grid_model': gfit.simulator.model,
                'chi2_init':       chi2_init,
                'chi2_final':      gfit.simulator.chi2r,
                'gfit':            gfit,
                'model_to_use':    model_to_use,
                'dtypes':          grid_dtypes,
                'axes':            axes,
            }
            st.success("✅ Grid search complete!")
            log_event(
                "Fit run completed",
                f"grid chi2r={chi2_init:.4f}->{gfit.simulator.chi2r:.4f}",
            )
            st.balloons()
        except Exception as exc:
            logger.exception("Grid search failed")
            st.error(f"Grid search error: {exc}")
            log_event("Fit run failed", f"grid: {exc}")

    # ── Code reproductible — right after Run, visible without a click ───
    with st.expander("Reproducible Python code", expanded=True):
        st.code(
            generate_fitting_code(
                method="grid",
                result={"dtypes": grid_dtypes, "axes": axes},
                data_filenames=st.session_state.get("selected_files", []),
                model_comps=st.session_state.MODEL[model_to_use]["components"],
                applied_filters=st.session_state.get("applied_filters", []),
                registry=registry,
            ),
            language="python",
        )

    if st.session_state.grid_result is None:
        return

    r = st.session_state.grid_result

    st.markdown("### C — Results")
    if r['chi2_final'] > r['chi2_init']:
        st.warning(f"⚠️ Divergence: {r['chi2_init']:.2f} → {r['chi2_final']:.2f}")
    else:
        st.success(f"✅ χ²ᵣ: {r['chi2_init']:.2f} → {r['chi2_final']:.2f}")

    # ── Best grid point (table)  |  χ² map — same row ───────────────────
    col_tbl, col_map = st.columns(2)
    with col_tbl:
        st.markdown("##### Best grid point")
        _, tbl_grid = get_result_df(r['best_grid_model'], is_fit=False)
        st.dataframe(tbl_grid, use_container_width=True)

    fig_map = None
    with col_map:
        st.markdown("##### χ² map")
        is_2d      = len(r['axes']) == 2
        scale_opts = ["log", "linear"]
        with st.expander("Plot parameters", expanded=False):
            yscale_raw = st.selectbox(
                "χ² scale", scale_opts, key="grid_map_scale",
                help="Log scale (default) makes the minimum easier to spot "
                     "when a few points dominate the range.",
            )
        try:
            yscale = choice(yscale_raw, scale_opts, "χ² scale")
        except InvalidInput as exc:
            st.warning(str(exc))
            yscale = "log"
        try:
            try:
                plot_kwargs = {"norm": LogNorm()} if (is_2d and yscale == "log") else {}
                fig_map, _ax_map = r['gfit'].plotMap(
                    plotContour=is_2d, plotMinLines=True, **plot_kwargs,
                )
                if not is_2d and yscale == "log":
                    _ax_map.set_yscale("log")
            except ValueError:
                # LogNorm needs strictly positive data — falls back to
                # linear if any grid point's chi2r is exactly 0 (a
                # perfect fit) instead of losing the whole map.
                fig_map, _ax_map = r['gfit'].plotMap(
                    plotContour=is_2d, plotMinLines=True,
                )
            safe_pyplot(st, fig_map, use_container_width=True)
        except Exception:
            logger.exception("Grid map rendering failed")
            st.warning("Could not render the χ² map for this grid.")

    grid_csv = _grid_map_to_csv(r['gfit'], r['axes'])

    # ── Code reproductible (built before the buttons below, so the zip
    # download can reuse it without recomputation) ─────────────────────
    code = generate_fitting_code(
        method="grid",
        result={"dtypes": r['dtypes'], "axes": r['axes']},
        data_filenames=st.session_state.get("selected_files", []),
        model_comps=st.session_state.MODEL[r["model_to_use"]]["components"],
        applied_filters=st.session_state.get("applied_filters", []),
        registry=registry,
    )
    zip_bytes = build_results_zip(
        param_table=tbl_grid,
        code=code,
        figures={"chi2_map": fig_map},
        extra_files={
            "grid_chi2map.csv": grid_csv,
            "activity_log.txt": get_log_text(),
            **_all_models_as_txt(registry),
        },
    )
    safe_model_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(r['model_to_use']))[:100] or "model"

    # ── Save / Download — same row, below table + map ───────────────────
    col_save, col_dl = st.columns(2)
    with col_save:
        if st.button("💾 Save best grid model", type="primary", use_container_width=True, key="save_grid"):
            update_model_from_fit(
                f"Best_Grid_{r['model_to_use']}", r['model_to_use'],
                r['best_grid_model'], chi2r=r['chi2_final'],
            )
            st.success(f"Model **Best_Grid_{r['model_to_use']}** saved!")
            log_event("Best model saved", f"Best_Grid_{r['model_to_use']}")
    with col_dl:
        st.download_button(
            "📦 Download results (zip)",
            data=zip_bytes,
            file_name=f"grid_results_{safe_model_name}.zip",
            mime="application/zip",
            use_container_width=True,
            on_click=lambda: log_event("Results zip downloaded", f"grid model={r['model_to_use']}"),
            key="download_grid_zip",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Emcee
# ═══════════════════════════════════════════════════════════════════════════

def _render_emcee(oim, registry, data, model_to_use: str) -> None:
    st.markdown("### B — Emcee MCMC")

    model_emcee = build_oim_model(
        oim, registry,
        st.session_state.MODEL[model_to_use]["components"],
    )
    if model_emcee is None:
        st.error("Cannot build model.")
        return

    # Rule of thumb defaults, from the current model's free-parameter count:
    # 2*nfree+1 walkers, up to 500 steps per free parameter — both clamped
    # to the server-enforced caps below, never exceeding them. Floored at
    # 4 walkers regardless of nfree: oimFitterEmcee's default moves
    # (DEMove + DESnookerMove, see oimFitterEmcee._prepare) split the
    # ensemble into two halves each needing at least 2 walkers to draw a
    # proposal pair from — reproduced directly against oimodeler with
    # nwalkers=3 (2*1+1 for a 1-free-parameter model): "ValueError: a must
    # be greater than 0 unless no samples are taken" the instant sampling
    # starts, regardless of steps/data. 4 was confirmed to work for the
    # same model.
    nb_free         = len(model_emcee.getFreeParameters())
    default_walkers = min(max(2 * nb_free + 1, 4), MAX_EMCEE_WALKERS)
    default_steps   = min(max(500 * nb_free, 0), MAX_EMCEE_STEPS)

    # Emcee's "gaussian" init mode requires every free parameter to
    # already have a well-defined range around a sensible starting value —
    # "random" (uniform over each parameter's bounds) is the only mode
    # that works unconditionally, so it's the only one offered here.
    init_mode = "random"
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        emcee_dtypes_raw = st.multiselect(
            "Data to fit", FITTABLE_DATA_TYPES,
            default=["VIS2DATA", "T3PHI"], key="emcee_dtypes",
        )
    with ec2:
        nb_walkers_raw = st.number_input(
            "Walkers", 4, MAX_EMCEE_WALKERS, default_walkers, key="emcee_walkers",
            help=f"Default: 2×(free parameters)+1 = {default_walkers} for this model. "
                 "Minimum 4: oimodeler's default Emcee moves need the walker "
                 "ensemble to split into two halves of at least 2 each.",
        )
    with ec3:
        nb_steps_raw = st.number_input(
            "Steps", 0, MAX_EMCEE_STEPS, default_steps, key="emcee_steps",
            help=f"Default: min(500×(free parameters), {MAX_EMCEE_STEPS}) = {default_steps} for this model.",
        )

    try:
        # Widget bounds are cosmetic only. nb_walkers/nb_steps directly
        # size the MCMC run — the concrete CPU DoS vector from the audit
        # (V7's mitigation is a separate semaphore/cooldown workstream,
        # but the values themselves must still be bounded server-side, V4).
        emcee_dtypes = choices(emcee_dtypes_raw, FITTABLE_DATA_TYPES, "Data to fit")
        nb_walkers   = num(nb_walkers_raw, 4, MAX_EMCEE_WALKERS, "Walkers", integer=True)
        nb_steps     = num(nb_steps_raw, 0, MAX_EMCEE_STEPS, "Steps", integer=True)
    except InvalidInput as exc:
        st.warning(str(exc))
        return

    if st.button("▶️ Run Emcee", type="primary", use_container_width=True):
        log_event(
            "Fit run started",
            f"emcee model={model_to_use} dtypes={','.join(emcee_dtypes)} "
            f"walkers={nb_walkers} steps={nb_steps} init={init_mode}",
        )
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
            # Per-session, per-run unique path (services/storage.py's
            # session_dir()) — a single shared "/tmp/sampler_emcee.txt" used
            # by every session on the server caused two race conditions
            # under concurrent fits: one session's pre-run unlink() deleting
            # the inode another session's HDFBackend had open ("No such
            # file or directory"), and two sessions' HDFBackend opening the
            # same path at once ("file is already open for read-only").
            # Left in place after the run (not unlinked) since "Refine
            # results" below re-reads it via emfit.getResults() later in
            # the session; session_dir()'s own TTL-based purge_expired()
            # reclaims it eventually.
            sampler_path = session_dir() / f"sampler_{uuid.uuid4().hex}.txt"
            emfit.prepare(init=init_mode, samplerFile=str(sampler_path))

            progress_bar = st.progress(0)
            status_box   = st.empty()
            status_box.info("MCMC running …")
            try:
                run_emcee_with_progress(
                    emfit, nb_steps, progress_callback=lambda v: progress_bar.progress(v),
                )
            finally:
                # Always clear the progress UI, including on failure —
                # otherwise a stale "MCMC running …"/full progress bar was
                # left on screen underneath the st.error() below.
                progress_bar.empty()
                status_box.empty()

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
            log_event(
                "Fit run completed",
                f"emcee chi2r={chi2_init:.4f}->{emfit.simulator.chi2r:.4f}",
            )
            st.balloons()
        except Exception as exc:
            st.error(f"Emcee error: {exc}")
            log_event("Fit run failed", f"emcee: {exc}")

    # ── Code reproductible — right after Run, visible without a click ───
    with st.expander("Reproducible Python code", expanded=True):
        st.code(
            generate_fitting_code(
                method="emcee",
                result={
                    "dtypes":   emcee_dtypes,
                    "nwalkers": nb_walkers,
                    "nsteps":   nb_steps,
                    "init":     init_mode,
                },
                data_filenames=st.session_state.get("selected_files", []),
                model_comps=st.session_state.MODEL[model_to_use]["components"],
                applied_filters=st.session_state.get("applied_filters", []),
                registry=registry,
            ),
            language="python",
        )

    if st.session_state.emcee_result is None:
        return

# ── Tab Code & Save ───────────────────────────────────────────────────

    er = st.session_state.emcee_result

    st.markdown("### C — Results")

    with st.expander("🔧 Refine results (discard burn-in / χ² threshold)", expanded=False):
        st.caption(
            "Re-processes the existing MCMC chain — no new sampling. Matches "
            "oimodeler's own getResults()/printResults()/walkersPlot()/"
            "cornerPlot() `mode`/`discard`/`thin`/`chi2limfact` arguments. "
            "Applying refreshes the fitted parameters, model image, Data vs "
            "model and FLUXDATA tabs below, as well as the Walkers and Corner plots."
        )
        max_discard = max(er['nsteps'] - 1, 0)
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            mode_options = ["best", "mean", "median"]
            mode_raw = st.selectbox(
                "Mode", mode_options,
                index=mode_options.index(er.get('mode', 'best')),
                key="em_refine_mode",
            )
        with rc2:
            discard_raw = st.number_input(
                "Discard (burn-in steps)", 0, max_discard,
                min(er.get('discard', 0), max_discard), key="em_refine_discard",
            )
        with rc3:
            thin_raw = st.number_input(
                "Thin", 1, max(er['nsteps'], 1), er.get('thin', 1), key="em_refine_thin",
            )
        with rc4:
            chi2limfact_raw = st.number_input(
                "χ² lim factor", 0.01, 1000.0, er.get('chi2limfact', 20.0),
                key="em_refine_chi2limfact",
            )

        if st.button("🔄 Apply", type="primary", use_container_width=True, key="btn_refine_emcee"):
            try:
                mode        = choice(mode_raw, mode_options, "Mode")
                discard     = num(discard_raw, 0, max_discard, "Discard", integer=True)
                thin        = num(thin_raw, 1, max(er['nsteps'], 1), "Thin", integer=True)
                chi2limfact = num(chi2limfact_raw, 0.01, 1000.0, "χ² lim factor")
            except InvalidInput as exc:
                st.warning(str(exc))
            else:
                try:
                    er['lmfit'].getResults(mode=mode, discard=discard, thin=thin,
                                           chi2limfact=chi2limfact)
                    er['best_emcee_model'] = er['lmfit'].simulator.model
                    er['chi2_final']       = er['lmfit'].simulator.chi2r
                    er['mode'], er['discard'], er['thin'], er['chi2limfact'] = (
                        mode, discard, thin, chi2limfact,
                    )
                    log_event(
                        "Emcee results refined",
                        f"mode={mode} discard={discard} thin={thin} chi2limfact={chi2limfact}",
                    )
                    st.success("✅ Results refreshed.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not refine results: {exc}")

    st.markdown(f"χ²ᵣ: **{er['chi2_init']:.2f}** → **{er['chi2_final']:.2f}**")
    if er.get('discard') or er.get('thin', 1) != 1 or er.get('chi2limfact', 20) != 20 or er.get('mode', 'best') != 'best':
        st.caption(
            f"Refined with mode={er.get('mode', 'best')}, discard={er.get('discard', 0)}, "
            f"thin={er.get('thin', 1)}, χ² lim factor={er.get('chi2limfact', 20)}"
        )
    st.markdown("##### Fitted parameters")
    _, tbl_em = get_result_df(er['best_emcee_model'], is_fit=False)
    st.dataframe(tbl_em, use_container_width=True, height=350)

    if st.button("💾 Save best Emcee model", type="primary", use_container_width=True,
                 key="save_emcee"):
        update_model_from_fit(
            f"Best_Emcee_{er['model_to_use']}", er['model_to_use'],
            er['best_emcee_model'], chi2r=er['chi2_final'],
        )
        st.success(f"Model **Best_Emcee_{er['model_to_use']}** saved!")
        log_event("Best model saved", f"Best_Emcee_{er['model_to_use']}")

    # Full reproducible code (including any refine mode/discard/thin/
    # chi2limfact) — no longer shown here as its own expander (moved to
    # right after the Run button above); kept as a variable since the
    # results zip download below still bundles it.
    code = generate_fitting_code(
        method="emcee",
        result={
            "dtypes":      er['dtypes'],
            "nwalkers":    er['nwalkers'],
            "nsteps":      er['nsteps'],
            "init":        er['init'],
            "mode":        er.get('mode', 'best'),
            "discard":     er.get('discard', 0),
            "thin":        er.get('thin', 1),
            "chi2limfact": er.get('chi2limfact', 20),
        },
        data_filenames=st.session_state.get("selected_files", []),
        model_comps=st.session_state.MODEL[er["model_to_use"]]["components"],
        applied_filters=st.session_state.get("applied_filters", []),
        registry=registry,
    )


# ── Résultats ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Results display")

    # Populated by the tabs below, if their figure was generated successfully.
    # Reused as-is by the "Download results" zip further down — nothing is
    # regenerated.
    fig_0 = fig_img = fw = fc = None

    # Data type(s) actually used for this fit (er['dtypes']) — previously
    # this tab always plotted VIS2DATA + T3PHI regardless of what "Data to
    # fit" was set to, so e.g. a VISAMP-only fit still only ever showed
    # VIS²/T3PHI, neither of which was part of the fit.
    plot_dtypes = er.get('dtypes') or ['VIS2DATA', 'T3PHI']
    tabs = st.tabs(["Data vs model", "Model image", "FLUXDATA / components", "Walkers", "Corner plot"])
    tab_vis, tab_img, tab_flux, tab_walk, tab_corner = tabs

    # ── Data vs model ────────────────────────────────────────────────
    with tab_vis:
        st.caption(f"Data type(s) used for this fit: {', '.join(plot_dtypes)}.")
        col_p, col_g = st.columns([1, 2])
        with col_p:
            with st.expander("Axis controls", expanded=False):
                st.write("$X$ axis")
                vt_xs  = st.selectbox("X scale", ["linear", "log"], key="em_vt_xs")
                vt_xmn = st.number_input("Xmin (cycle/mas)", value=0., key="em_vt_xmn")
                vt_xmx = st.number_input("Xmax (cycle/mas)", value=5., key="em_vt_xmx")

                panel_settings = []
                for dtype in plot_dtypes:
                    st.write(f"**{dtype}**")
                    default_range = _DEFAULT_Y_RANGE.get(dtype)
                    auto = st.checkbox(
                        "Auto Y range", value=default_range is None, key=f"em_vt_{dtype}_auto",
                    )
                    yscale = st.selectbox(
                        "Y scale", ["linear", "log"], key=f"em_vt_{dtype}_yscale",
                    )
                    ymin = ymax = None
                    if not auto:
                        lo_default, hi_default = default_range or (0.0, 1.0)
                        ymin = st.number_input("Ymin", value=lo_default, key=f"em_vt_{dtype}_ymin")
                        ymax = st.number_input("Ymax", value=hi_default, key=f"em_vt_{dtype}_ymax")
                    panel_settings.append((dtype, yscale, ymin, ymax))
        with col_g:
            try:
                sim_plot = oim.oimSimulator(data=data, model=er['best_emcee_model'])
                sim_plot.compute(computeChi2=False, computeSimulatedData=True)
                fig_0, ax_0 = sim_plot.plot(plot_dtypes)

                for i, (dtype, yscale, ymin, ymax) in enumerate(panel_settings):
                    ax_0[i].set_xscale(vt_xs); ax_0[i].set_yscale(yscale)
                    ax_0[i].set_xlim(vt_xmn * 1e7, vt_xmx * 1e7)
                    if ymin is not None and ymax is not None and ymin < ymax:
                        ax_0[i].set_ylim(ymin, ymax)
                    ax_0[i].set_title(dtype); ax_0[i].grid(True, alpha=0.3)

                safe_pyplot(st, fig_0, use_container_width=True)
            except Exception as exc:
                st.warning(f"{', '.join(plot_dtypes)}: {exc}")

    # ── MODEL IMAGE ───────────────────────────────────────────────────
    with tab_img:
        img_cmap_options = ["hot", "inferno", "viridis", "plasma", "gray", "afmhot"]
        col_p, col_g = st.columns([1, 2])
        with col_p:
            with st.expander("Image parameters", expanded=False):
                img_gamma_raw = st.slider("Gamma γ", 0.05, 1.0, 1.0, 0.05, key="em_img_gamma")
                img_cmap_raw  = st.selectbox(
                    "Colormap", img_cmap_options,
                    key="em_img_cmap",
                )
                img_size_raw  = st.number_input("Image size (px)", 64, 512, 128,
                                            step=64, key="em_img_size")
                img_scale_raw = st.number_input("Scale (mas/px)", 0.1, 10., 1.,
                                            step=0.1, key="em_img_scale")
                use_wl    = st.checkbox("Filter on λ", value=False, key="em_img_use_wl")
                wl_val_raw = 3.5
                if use_wl:
                    wl_val_raw = st.number_input("λ (µm)", value=3.5, step=0.1,
                                             key="em_img_wl")
                img_clip_lo = st.number_input("Colormap percentile min", 0., 100., 0.,
                                              key="em_img_clip_lo")
                img_clip_hi = st.number_input("Colormap percentile max", 0., 100., 100.,
                                              key="em_img_clip_hi")
        with col_g:
            try:
                # Widget bounds are cosmetic only — img_size/img_scale feed
                # extract_model_image()'s array allocation directly, the
                # concrete OOM DoS vector from the audit (V6).
                img_gamma = num(img_gamma_raw, 0.05, 1.0, "Gamma")
                img_cmap  = choice(img_cmap_raw, img_cmap_options, "Colormap")
                img_size  = num(img_size_raw, 64, 512, "Image size", integer=True)
                img_scale = num(img_scale_raw, 0.1, 10.0, "Scale")
                wl_val = num(wl_val_raw, 0.1, 30.0, "Wavelength") * 1e-6

                img_data    = extract_model_image(oim, er['best_emcee_model'],
                                                  img_size=img_size,
                                                  img_scale=img_scale,
                                                  wl_value=wl_val)
                display_img = img_data[0, 0] ** img_gamma
                extent_half = img_size * img_scale / 2

                # Widget bounds aren't server-enforced — re-validate before use.
                clip_lo, clip_hi = sorted((
                    min(max(float(img_clip_lo), 0.), 100.),
                    min(max(float(img_clip_hi), 0.), 100.),
                ))
                vmin, vmax = np.percentile(display_img, [clip_lo, clip_hi])

                fig_img, ax_img = plt.subplots(figsize=(5, 5))
                # Astronomical convention: RA increases to the left (East left).
                im_plot = ax_img.imshow(
                    display_img, cmap=img_cmap, origin='lower',
                    extent=[extent_half, -extent_half, -extent_half, extent_half],
                    vmin=vmin, vmax=vmax,
                )
                # fig_img.colorbar (not plt.colorbar): see the same fix's
                # comment in pages/explorer.py — plt.colorbar() attaches to
                # pyplot's global "current figure", which drifts once
                # several tabs/figures are built in the same rerun.
                fig_img.colorbar(im_plot, ax=ax_img, label=f'Intensity (γ={img_gamma})')
                ax_img.set_xlabel("ΔRA (mas)"); ax_img.set_ylabel("ΔDec (mas)")
                wl_label = f" @ {wl_val*1e6:.2f} µm" if wl_val else ""
                ax_img.set_title(f"Model{wl_label}")
                safe_pyplot(st, fig_img, use_container_width=True)
            except InvalidInput as exc:
                st.warning(str(exc))
            except Exception:
                logger.exception("Model image rendering failed")
                st.warning("Could not render the model image for the current settings.")

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
        st.caption(
            f"Uses the same discard/thin/χ² lim factor as \"🔧 Refine results\" above "
            f"(discard={er.get('discard', 0)}, thin={er.get('thin', 1)}, "
            f"χ² lim factor={er.get('chi2limfact', 20)})."
        )
        try:
            fw, _ = er['lmfit'].walkersPlot(
                discard=er.get('discard', 0), thin=er.get('thin', 1),
                chi2limfact=er.get('chi2limfact', 20),
            )
            safe_pyplot(st, fw, use_container_width=True)
        except Exception as exc:
            st.warning(f"Walkers: {exc}")

    # ── CORNER PLOT ───────────────────────────────────────────────────
    with tab_corner:
        st.caption(
            f"Uses the same discard/thin/χ² lim factor as \"🔧 Refine results\" above "
            f"(discard={er.get('discard', 0)}, thin={er.get('thin', 1)}, "
            f"χ² lim factor={er.get('chi2limfact', 20)})."
        )
        try:
            # `chi2limfact` (not the old `dchi2limfact`, an unrecognized
            # kwarg that cornerPlot's **kwargs silently swallowed — the
            # plot always used its hardcoded default chi2limfact=20,
            # never the 5 this call intended).
            fc, _ = er['lmfit'].cornerPlot(
                discard=er.get('discard', 0), thin=er.get('thin', 1),
                chi2limfact=er.get('chi2limfact', 20),
            )
            safe_pyplot(st, fc, use_container_width=True)
        except Exception as exc:
            st.warning(f"Corner: {exc}")

    # ── Download all results as a zip ───────────────────────────────────
    st.markdown("---")
    zip_bytes = build_results_zip(
        param_table=tbl_em,
        code=code,
        figures={
            "data_fit_plot": fig_0,
            "model_image":   fig_img,
            "walkers_plot":  fw,
            "corner_plot":   fc,
        },
        extra_files={
            "activity_log.txt": get_log_text(),
            **_all_models_as_txt(registry),
        },
    )
    # model_to_use comes from a selectbox — its widget option list isn't
    # server-enforced, so sanitize before using it in a client-facing filename.
    safe_model_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(er['model_to_use']))[:100] or "model"
    st.download_button(
        "📦 Download results (zip)",
        data=zip_bytes,
        file_name=f"emcee_results_{safe_model_name}.zip",
        mime="application/zip",
        use_container_width=True,
        on_click=lambda: log_event("Results zip downloaded", f"emcee model={er['model_to_use']}"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers internes
# ═══════════════════════════════════════════════════════════════════════════

def _render_dataset_summary() -> None:
    """Lists the selected datasets and the filters currently applied —
    the same information every fit method below uses, shown once instead
    of duplicated per method. Filters are session-wide (st.session_state.
    applied_filters, see pages/data.py's filter workbench), not specific
    to this page."""
    selected = st.session_state.get('selected_files', []) or []
    st.markdown(f"**Datasets** — {len(selected)} selected")
    if not selected:
        st.caption("No dataset selected — go to the Data tab.")
        return
    for fname in selected:
        st.markdown(f"- `{fname}`")

    applied = st.session_state.get('applied_filters', [])
    if not applied:
        st.caption("Filters applied: none.")
        return
    filt_bits = []
    for entry in applied:
        targets = entry["kwargs"].get("targets")
        targets_txt = "all files" if not targets else f"files {targets}"
        filt_bits.append(f"{entry['filter_class']} ({targets_txt})")
    st.caption("Filters applied: " + " · ".join(filt_bits))


def _render_model_summary(model_to_use: str) -> None:
    """Recap, per component, of every parameter's current value, bounds,
    and free/fixed status for the model about to be fit."""
    comps = st.session_state.MODEL.get(model_to_use, {}).get("components", [])
    with st.expander(f"Model components — « {model_to_use} »", expanded=False):
        if not comps:
            st.caption("This model has no components.")
            return
        for c in comps:
            free = set(c.get("free_params", []))
            params = c.get("params", list(c["initial_values"].keys()))
            rows = []
            for p in params:
                lo, hi = c.get("param_ranges", {}).get(p, (None, None))
                interp = c.get("interpolators", {}).get(p, {}).get("enabled", False)
                rows.append({
                    "Parameter": p,
                    "Value":     c["initial_values"].get(p),
                    "Min":       lo,
                    "Max":       hi,
                    "Free":      "interpolated" if interp else (p in free),
                })
            st.markdown(f"**{c['name']}** ({c['type']})")
            st.dataframe(rows, use_container_width=True, hide_index=True)


def _grid_map_to_csv(gfit, axes: list[dict]) -> str:
    """Flattens a completed oimFitterRegularGrid's χ² map into a CSV table
    (one row per grid point: each axis' value + the resulting χ²ᵣ) — the
    raw grid data behind `chi2_map.png`, savable independently of the plot."""
    header = [a['name'] for a in axes] + ['chi2r']
    lines = [",".join(header)]
    for idx in np.ndindex(*gfit.gridSize):
        coords = [repr(float(gfit.grid[d][idx[d]])) for d in range(len(axes))]
        chi2 = repr(float(gfit.chi2rMap[idx]))
        lines.append(",".join(coords + [chi2]))
    return "\n".join(lines)


def _all_models_as_txt(registry) -> dict[str, str]:
    """Every model saved this session, in the normalized .txt format
    (core/model_export.py) importable back via Modelling > Import model —
    bundled into every result zip so a download carries the full model
    library, not just the one model this particular fit used."""
    return {
        f"models/{name}.txt": model_to_txt(model_dict, registry)
        for name, model_dict in st.session_state.MODEL.items()
    }

