# core/code_generator.py
"""
Génération de code Python reproductible pour χ² et Emcee.
Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import numpy as np

from config.constants import DEFAULT_PARAM_RANGES

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

    for i, c in enumerate(model_comps):
        vname     = f"comp{i+1}"
        comp_type = c["type"]
        params    = registry.get(comp_type, {}).get(
            "params", c.get("params", list(c["initial_values"].keys()))
        )
        interps = c.get("interpolators", {})

        scalar_params = {
            p: c["initial_values"].get(p, 0.)
            for p in params
            if p not in interps or not interps[p].get("enabled", False)
        }
        param_str = ", ".join(f"{p}={v!r}" for p, v in scalar_params.items())

        for p, cfg in interps.items():
            if not cfg.get("enabled", False):
                continue
            if cfg["type"] == "blackbody":
                wl_var = f"wl_{vname}_{p}"
                lines += [
                    f"{wl_var} = np.linspace(1e-6, 5e-6, 200)",
                    f"interp_{vname}_{p} = oim.oimInterp('starWl', "
                    f"temp={cfg['temp']}, dist={cfg['dist']}, "
                    f"lum={cfg['lum']}, wl={wl_var})",
                ]
            else:
                wl_arr  = repr(cfg["wl"])
                val_arr = repr(cfg["values"])
                var_key = cfg.get("var", "wl")
                lines += [
                    f"interp_{vname}_{p} = oim.oimInterp('{var_key}', "
                    f"{var_key}=np.array({wl_arr}), values=np.array({val_arr}))",
                ]
            param_str += f", {p}=interp_{vname}_{p}"

        lines.append(f"{vname} = oim.{comp_type}({param_str})")
        lines.append("")
        comp_var_names.append(vname)

    comp_args = ", ".join(comp_var_names)
    lines += [f"model = oim.oimModel({comp_args})", ""]

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
    # Interpolated parameters (oimInterp, built in section 3 above) are
    # intentionally skipped here: oimodeler expands a single interpolated
    # parameter into its own sub-parameters (keyed "..._{param}_interp1",
    # "..._interp2", ...) whose count depends on the interpolator (e.g. the
    # number of wavelength control points) and isn't known until the
    # interpolator itself is built. There is no single UI-configured
    # (min, max, free) triple that applies to them positionally or by name,
    # so they keep the bounds/free status oimInterp gives them.
    param_settings = {}
    for i, c in enumerate(model_comps):
        comp_type = c["type"]
        comp_cls  = registry.get(comp_type, {}).get("class")
        shortname = comp_cls.shortname.replace(" ", "_") if comp_cls is not None else comp_type
        params    = registry.get(comp_type, {}).get(
            "params", c.get("params", list(c["initial_values"].keys()))
        )
        interps = c.get("interpolators", {})

        for p in params:
            if p in interps and interps[p].get("enabled", False):
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
            'fig, ax = fitter.simulator.plot(["VIS2DATA", "T3PHI"])',
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
        nwalkers = result.get("nwalkers", 32)
        nsteps   = result.get("nsteps",   1000)
        init     = result.get("init",     "gaussian")
        lines += [
            "# ── 5. Emcee MCMC ──────────────────────────────────────",
            f"fitter = oim.oimFitterEmcee(data, model, nwalkers={nwalkers},",
            f"                            dataTypes={dtypes_str})",
            f'fitter.prepare(init="{init}")',
            f"fitter.run(nsteps={nsteps}, progress=True)",
            "",
            "fitter.printResults()",
            "",
            "# ── 6. Visualization ────────────────────────────────────",
            "fig_w, _ = fitter.walkersPlot(chi2limfact=5)",
            "fig_c, _ = fitter.cornerPlot(dchi2limfact=5)",
            'fig, ax  = fitter.simulator.plot(["VIS2DATA", "T3PHI"])',
            "plt.show()",
        ]

    return "\n".join(lines)
