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
import streamlit as st

def date():
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
    now = datetime.now()
    date_str = now.strftime("%d - %B - %Y")
    return date_str


def generate_fitting_code(method: str, result: dict, data_filename: str,
                           model_comps: list, filter_params: dict,
                           registry: dict) -> str:
    """
    Génère un script Python autonome reproduisant le fitting.

    Paramètres
    ----------
    method        : "chi2" ou "emcee"
    result        : dict contenant dtypes, nwalkers, nsteps, init (emcee)
    data_filename : nom du fichier OIFITS
    model_comps   : liste de dicts de composants
    filter_params : dict avec expr, bin_L, bin_N, norm_L, norm_N
    registry      : COMPONENT_REGISTRY
    """

    lines = []

    # ── Header ────────────────────────────────────────────────────────
    lines += [
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
    for i, fname in enumerate(st.session_state.test_selected_file):
        vname = f"file{i+1}"
        lines.append(f'{vname} = path + "{fname}"')
        file_vars.append(vname)

    files_arg = "[" + ", ".join(file_vars) + "]"
    lines += [
        f"data = oim.oimData({files_arg})",
        "",
    ]

    # ── Filtre spectral ───────────────────────────────────────────────
    lines += ["# ── 2. Spectral filtering ──────────────────────────────"]
    expr   = filter_params.get("expr", "")
    bin_L  = filter_params.get("bin_L", 1)
    bin_N  = filter_params.get("bin_N", 1)
    norm_L = filter_params.get("norm_L", False)
    norm_N = filter_params.get("norm_N", False)

    if expr:
        lines.append(f'f_wl = oim.oimFlagWithExpressionFilter(expr="{expr}", keepOldFlag=False)')
    lines += [
        f"f_bL = oim.oimWavelengthBinningFilter(targets=0, bin={bin_L}, normalizeError={norm_L})",
        f"f_bN = oim.oimWavelengthBinningFilter(targets=0, bin={bin_N}, normalizeError={norm_N})",
    ]
    if expr:
        lines.append("data.setFilter(oim.oimDataFilter([f_wl, f_bL, f_bN]))")
    else:
        lines.append("data.setFilter(oim.oimDataFilter([f_bL, f_bN]))")
    lines += ["data.useFilter = True", ""]

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

    min_value   = []
    max_value   = []
    free_status = []
    for i, c in enumerate(model_comps):
        comp_type = c["type"]
        params    = registry.get(comp_type, {}).get(
            "params", c.get("params", list(c["initial_values"].keys()))
        )
        

        for p in params:
            lo, hi = c["param_ranges"].get(p, (None, None))
            free   = p in c.get("free_params", [])
            min_value.append(lo)
            max_value.append(hi)
            free_status.append(free)
        
    lines+=[
        f"min_value = {min_value}",
        f"max_value = {max_value}",
        f"free_status = {free_status}",
        "for i, j, k, l in zip(model.getParameters().keys(), min_value , max_value ,free_status) :",
        "\tmodel.getParameters()[i].set(min=j, max=k, free=l)",
    ]


            #param_key = f"{c['name'].replace("oim", f"c{i+1}_")}_{p}"
            #lines.append(
            #    f"model.getParameters()['{param_key}']"
            #    f".set(min={lo!r}, max={hi!r}, free={free})"
            #)
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