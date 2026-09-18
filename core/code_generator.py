# core/code_generator.py
"""
Génération de code Python reproductible pour χ² et Emcee.
Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import numpy as np

from config.constants import DEFAULT_PARAM_RANGES
from core.interp_registry import get_full_layout

import locale
from datetime import datetime


def date():
    return datetime.now().strftime("%d/%m/%Y")


def generate_fitting_code(method: str, result: dict, data_filenames: list,
                           model_comps: list, applied_filters: list,
                           registry: dict) -> str:
    """
    Génère un script Python autonome reproduisant le fitting.

    Paramètres
    ----------
    method          : "chi2", "grid" ou "emcee"
    result          : dict contenant dtypes, nwalkers, nsteps, init (emcee)
    data_filenames  : liste des noms de fichiers OIFITS utilisés pour le fit,
                       dans l'ordre exact utilisé pour construire oimData
                       (caller passes e.g. st.session_state.selected_files –
                       no Streamlit dependency here, core/ stays pure).
    model_comps     : liste de dicts de composants
    applied_filters : liste de specs { 'filter_class': str, 'kwargs': dict }
                       — même contrat que st.session_state.applied_filters
                       (voir core/filter_registry.py + pages/data.py's filter
                       workbench et services/data_service.build_filters_from_specs,
                       ce module génère juste le code source équivalent au
                       lieu de l'exécuter).
    registry        : COMPONENT_REGISTRY
    """

    lines = []

    # ── Header ────────────────────────────────────────────────────────
    lines += [
        "import os",
        "import numpy as np",
        "import oimodeler as oim",
        "import matplotlib.pyplot as plt",
        "",
        "# ═══════════════════════════════════════════════════════════",
        f"# Fitting method : {method} - date {date()}",
        "# ═══════════════════════════════════════════════════════════",
        "",
    ]

    # ── Chargement des données ────────────────────────────────────────
    lines += [
        "# ── 1. Load data ───────────────────────────────────────────",
        "path = '[ABSOLUTE PATH TO OIFITS FOLDER - TO BE FILLED BY USER]'",
        "",
    ]

    file_vars = []
    for i, fname in enumerate(data_filenames):
        vname = f"file{i+1}"
        # os.path.join (not string concatenation) so this doesn't silently
        # produce a broken path when `path` isn't filled in with a trailing separator.
        lines.append(f'{vname} = os.path.join(path, "{fname}")')
        file_vars.append(vname)

    files_arg = "[" + ", ".join(file_vars) + "]"
    lines += [
        f"data = oim.oimData({files_arg})",
        "",
    ]

    # ── Filtres appliqués (partagés pour toute la session) ─────────────
    # Mirrors services/data_service.build_filters_from_specs(): None-valued
    # kwargs are omitted rather than passed literally, letting the filter
    # class fall back to its own default ("all") instead of crashing on
    # e.g. arr=None.
    lines += ["# ── 2. Applied filters ──────────────────────────────────"]
    filter_var_names = []

    for i, spec in enumerate(applied_filters):
        vname = f"f{i+1}"
        clean_kwargs = {k: v for k, v in spec.get("kwargs", {}).items() if v is not None}
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in clean_kwargs.items())
        lines.append(f"{vname} = oim.{spec['filter_class']}({kwargs_str})")
        filter_var_names.append(vname)

    if filter_var_names:
        lines.append(f"data.setFilter(oim.oimDataFilter([{', '.join(filter_var_names)}]))")
        lines.append("data.useFilter = True")
    else:
        lines.append("# No filter applied — data used as-is")
    lines.append("")

    # ── Construction du modèle ────────────────────────────────────────
    lines += ["# ── 3. Build model ─────────────────────────────────────"]
    comp_var_names = []

    comp_var_of = {c["name"]: f"comp{i+1}" for i, c in enumerate(model_comps)}

    for i, c in enumerate(model_comps):
        vname     = f"comp{i+1}"
        comp_type = c["type"]
        params    = registry.get(comp_type, {}).get(
            "params", c.get("params", list(c["initial_values"].keys()))
        )
        interps = c.get("interpolators", {})
        norms   = c.get("normalizations", {})

        scalar_params = {
            p: c["initial_values"].get(p, 0.)
            for p in params
            if (p not in interps or not interps[p].get("enabled", False))
            and (p not in norms or not norms[p].get("enabled", False))
        }
        param_str = ", ".join(f"{p}={v!r}" for p, v in scalar_params.items())

        for p, cfg in interps.items():
            if not cfg.get("enabled", False):
                continue
            macro     = cfg["macro"]
            kwarg_str = ", ".join(
                f"{k}=np.array({v!r})" if isinstance(v, list) else f"{k}={v!r}"
                for k, v in cfg["kwargs"].items()
            )
            lines.append(
                f"interp_{vname}_{p} = oim.oimInterp('{macro}', {kwarg_str})"
            )
            param_str += f", {p}=interp_{vname}_{p}"

        lines.append(f"{vname} = oim.{comp_type}({param_str})")
        lines.append("")
        comp_var_names.append(vname)

    comp_args = ", ".join(comp_var_names)
    lines += [f"model = oim.oimModel({comp_args})", ""]

    # oim.oimParamNorm(refs, norm=...) ties a component's parameter to
    # "norm - sum(refs)" over OTHER components' *already-built* parameter
    # objects — it can only be assigned once every component above exists,
    # so this comes after model construction (matching the oimodeler
    # example this was integrated from: `pt2.params["f"] =
    # oim.oimParamNorm(g2.params["f"])`), and before the .set() loop below
    # since oimParamNorm has no .set() of its own.
    norm_lines = []
    for c in model_comps:
        vname = comp_var_of[c["name"]]
        for p, cfg in c.get("normalizations", {}).items():
            if not cfg.get("enabled", False):
                continue
            ref_strs = [
                f'{comp_var_of[r["component"]]}.params[{r["param"]!r}]'
                for r in cfg.get("refs", [])
            ]
            norm_lines.append(
                f'{vname}.params[{p!r}] = oim.oimParamNorm('
                f'[{", ".join(ref_strs)}], norm={cfg.get("norm", 1.0)!r})'
            )
    if norm_lines:
        lines += ["# ── Flux normalization (oimParamNorm) ───────────────────"]
        lines += norm_lines
        lines.append("")

    # ── Paramètres du modèle ──────────────────────────────────────────
    lines += [
        "# ── 4. Set model parameters ────────────────────────────────",
        "",
    ]

    # Build a NAME-keyed mapping instead of zipping two independently-built
    # sequences positionally. oimModel.getParameters() (see the installed
    # oimodeler's oimModel.getParameters source) names each scalar parameter
    # "c{i+1}_{component.shortname}_{param}" (i is 0-based component index,
    # shortname is a class attribute e.g. oimUD.shortname == "UD"). This is
    # reproduced here directly from the registry's component classes, so the
    # generated script looks each parameter up by its real key instead of
    # assuming getParameters().keys() enumerates in the same order/count as
    # the registry's declared `params` list.
    #
    # Interpolated parameters (oimInterp, built in section 3 above) expand
    # into their own sub-parameters once oimodeler swaps the oimInterp
    # macro for a real oimParamInterpolator instance, keyed
    # "..._{param}_interp1", "..._interp2", ... — N is a 1-based sequential
    # index over the interpolator's FULL sub-parameter composition
    # (get_full_layout(), same registry core/component.py's
    # ComponentConfig.create_instance() uses to apply these bounds live).
    # Non-controllable slots oimodeler itself hardcodes free=False for
    # (e.g. powerlaw's x0, rangeWl's wlmin/wlmax) still occupy an index
    # but never get an entry here, mirroring create_instance() exactly.
    param_settings = {}
    for i, c in enumerate(model_comps):
        comp_type = c["type"]
        comp_cls  = registry.get(comp_type, {}).get("class")
        shortname = comp_cls.shortname.replace(" ", "_") if comp_cls is not None else comp_type
        params    = registry.get(comp_type, {}).get(
            "params", c.get("params", list(c["initial_values"].keys()))
        )
        interps = c.get("interpolators", {})
        norms   = c.get("normalizations", {})

        for p in params:
            if p in norms and norms[p].get("enabled", False):
                # oim.oimParamNorm has no .set() — it's a formula over
                # other parameters, not a fittable value of its own (see
                # the assignment emitted in section 3 above).
                continue
            if p in interps and interps[p].get("enabled", False):
                cfg = interps[p]
                bounds = cfg.get("bounds", {})
                idx = 0
                for kwarg_name, count, controllable in get_full_layout(cfg["macro"], cfg["kwargs"]):
                    entries = bounds.get(kwarg_name, [])
                    for j in range(count):
                        idx += 1
                        if controllable and j < len(entries):
                            b = entries[j]
                            key = f"c{i+1}_{shortname}_{p}_interp{idx}"
                            param_settings[key] = (
                                float(b['min']) if b.get('min') is not None else None,
                                float(b['max']) if b.get('max') is not None else None,
                                bool(b.get('free', False)),
                            )
                continue
            lo, hi = c["param_ranges"].get(p, (None, None))
            free   = p in c.get("free_params", [])
            key    = f"c{i+1}_{shortname}_{p}"
            param_settings[key] = (
                float(lo) if lo is not None else None,
                float(hi) if hi is not None else None,
                bool(free),
            )

    lines.append("param_settings = {")
    for key, (lo, hi, free) in param_settings.items():
        lines.append(f"    {key!r}: ({lo!r}, {hi!r}, {free!r}),")
    lines.append("}")
    lines += [
        "model_params = model.getParameters()",
        "for key, (lo, hi, free) in param_settings.items():",
        "    if key in model_params:",
        "        model_params[key].set(min=lo, max=hi, free=free)",
        "    else:",
        "        print(f'Warning: parameter {key} not found on model, skipping.')",
    ]
    lines.append("")

    # ── Fitting ───────────────────────────────────────────────────────
    dtypes     = result.get("dtypes", ["VIS2DATA", "T3PHI"])
    dtypes_str = repr(dtypes)

    if method == "chi2":
        lines += [
            "# ── 5. χ² minimization ─────────────────────────────────",
            f"fitter = oim.oimFitterMinimize(data, model, dataTypes={dtypes_str})",
            "fitter.prepare()",
            "fitter.run()",
            "",
            "fitter.printResults()",
            "",
            "# ── 6. Visualization ────────────────────────────────────",
            f"fig, ax = fitter.simulator.plot({dtypes_str})",
            "plt.show()",
        ]
    elif method == "grid":
        # Regular grid χ² exploration — see
        # https://oimodeler.readthedocs.io/en/latest/fitter.html#regular-grid-exploration
        axes = result.get("axes", [])
        axis_names = [a["name"] for a in axes]
        mins  = [a["lo"] for a in axes]
        maxs  = [a["hi"] for a in axes]
        steps = [
            (a["hi"] - a["lo"]) / (a["n"] - 1) if a["n"] > 1 else 0.0
            for a in axes
        ]
        lines += [
            "# ── 5. Regular grid χ² exploration ─────────────────────",
            f"fitter = oim.oimFitterRegularGrid(data, model, dataTypes={dtypes_str})",
            f"grid_param_names = {axis_names!r}",
            "model_params = model.getParameters()",
            "grid_params = [model_params[n] for n in grid_param_names]",
            f"fitter.prepare(params=grid_params, min={mins!r}, max={maxs!r}, "
            f"steps={steps!r})",
            "fitter.run(progress=True)",
            "",
            "fitter.printResults()",
            "",
            "# ── 6. Visualization ────────────────────────────────────",
            f"fig, ax = fitter.plotMap(plotContour={len(axes) == 2}, plotMinLines=True)",
            "plt.show()",
        ]
    else:  # emcee
        nwalkers    = result.get("nwalkers", 32)
        nsteps      = result.get("nsteps",   1000)
        init        = result.get("init",     "gaussian")
        # "Refine results" (mode/discard/thin/chi2limfact) re-processes the
        # already-sampled chain — no new sampling — and is forwarded as-is
        # to printResults()/walkersPlot()/cornerPlot(), matching the live
        # UI's "🔧 Refine results" expander exactly.
        mode        = result.get("mode",        "best")
        discard     = result.get("discard",     0)
        thin        = result.get("thin",        1)
        chi2limfact = result.get("chi2limfact", 20)
        # printResults()/getResults() accept `mode`; walkersPlot()/
        # cornerPlot() do not (verified against the installed oimodeler's
        # signatures) — passing it there raises deep inside matplotlib via
        # their **kwargs passthrough, so the two kwarg strings must differ.
        results_kwargs = (
            f"mode={mode!r}, discard={discard!r}, thin={thin!r}, "
            f"chi2limfact={chi2limfact!r}"
        )
        plot_kwargs = f"discard={discard!r}, thin={thin!r}, chi2limfact={chi2limfact!r}"
        lines += [
            "# ── 5. Emcee MCMC ──────────────────────────────────────",
            f"fitter = oim.oimFitterEmcee(data, model, nwalkers={nwalkers},",
            f"                            dataTypes={dtypes_str})",
            f'fitter.prepare(init="{init}")',
            f"fitter.run(nsteps={nsteps}, progress=True)",
            "",
            f"fitter.printResults({results_kwargs})",
            "",
            "# ── 6. Visualization ────────────────────────────────────",
            f"fig_w, _ = fitter.walkersPlot({plot_kwargs})",
            f"fig_c, _ = fitter.cornerPlot({plot_kwargs})",
            f"fig, ax = fitter.simulator.plot({dtypes_str})",
            "plt.show()",
        ]

    return "\n".join(lines)
