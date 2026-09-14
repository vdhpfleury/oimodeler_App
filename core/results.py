# core/results.py
"""
Extraction et mise à jour des résultats de fitting.
Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import io
import re
import copy
import zipfile
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st


def get_result_df(model_or_fit, is_fit: bool = False) -> tuple[float | None, pd.DataFrame]:
    """
    Retourne (chi2r, DataFrame des paramètres) depuis un modèle ou un fitter.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        if is_fit:
            model_or_fit.printResults()
            params = model_or_fit.simulator.model.getParameters()
        else:
            params = model_or_fit.getParameters()

    output = buf.getvalue()
    match  = re.search(r"chi2r\s*=\s*([0-9.eE+\-]+)", output)
    chi2r  = float(match.group(1)) if match else None

    rows = []
    for name, p in params.items():
        at_min = (p.min is not None) and abs(p.value - p.min) < 1e-10
        at_max = (p.max is not None) and abs(p.value - p.max) < 1e-10
        rows.append({
            "Parameter":   name,
            "Value":       p.value,
            "Uncertainty": p.error,
            "Min":         p.min,
            "Max":         p.max,
            "Free":        p.free,
            "At bound":    at_min or at_max,
        })
    return chi2r, pd.DataFrame(rows)


def update_model_from_fit(new_name: str, base_name: str,
                          fit_model, chi2r: float | None = None) -> dict:
    """
    Crée une copie du modèle de base mise à jour avec les paramètres du fit.
    Sauvegarde le résultat dans st.session_state.MODEL[new_name].
    """
    if base_name not in st.session_state.MODEL:
        raise KeyError(f"Base model '{base_name}' not found.")

    updated = copy.deepcopy(st.session_state.MODEL[base_name])
    params  = fit_model.getParameters()

    for full_name, p in params.items():
        parts = full_name.split("_")
        try:
            comp_index = int(parts[0][1:]) - 1
        except (ValueError, IndexError):
            continue
        if len(parts) < 3:
            continue
        param_name = "_".join(parts[2:])

        if 0 <= comp_index < len(updated["components"]):
            comp = updated["components"][comp_index]
            if param_name in comp["initial_values"]:
                comp["initial_values"][param_name] = p.value
            if param_name in comp["param_ranges"]:
                comp["param_ranges"][param_name] = (p.min, p.max)
            if p.free:
                if param_name not in comp["free_params"]:
                    comp["free_params"].append(param_name)
            else:
                if param_name in comp["free_params"]:
                    comp["free_params"].remove(param_name)

    if chi2r is not None:
        updated["chi2r"] = chi2r

    st.session_state.MODEL[new_name] = updated
    return updated


def build_results_zip(param_table: pd.DataFrame, code: str, figures: dict,
                      extra_files: dict[str, str] | None = None) -> bytes:
    """
    Empaquette les résultats d'un fit dans un zip en mémoire :
    - le tableau des meilleurs paramètres (CSV)
    - le script Python reproductible (généré par core/code_generator.py)
    - les figures matplotlib déjà générées, une par entrée de `figures`
    - tout fichier texte additionnel (ex : grille χ² brute d'un grid search)

    Ne recalcule rien : les figures sont réutilisées telles quelles.

    Paramètres
    ----------
    param_table : DataFrame des paramètres du meilleur modèle (get_result_df()).
    code : script Python reproductible (str).
    figures : dict[str, matplotlib.figure.Figure | None]
        Nom de fichier (sans extension) → figure. Une entrée absente ou à
        None est ignorée (ex : une figure qui n'a pas pu être générée).
    extra_files : dict[str, str] | None
        Nom de fichier (avec extension) → contenu texte, écrit tel quel.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("best_fit_parameters.csv", param_table.to_csv(index=False))
        zf.writestr("reproducible_fit.py", code)
        for name, fig in figures.items():
            if fig is None:
                continue
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
            zf.writestr(f"{name}.png", img_buf.getvalue())
        for filename, content in (extra_files or {}).items():
            zf.writestr(filename, content)
    return buf.getvalue()
