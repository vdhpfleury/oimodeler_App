# core/csv_import.py
"""
Import d'un modèle depuis un CSV de résultats.
Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import pandas as pd

from config.constants import DEFAULT_PARAM_RANGES, DEFAULT_PARAM_INIT, SHORT_TO_OIM


def _build_shortname_map(registry: dict) -> dict[str, str]:
    """Mappe le vrai `shortname` oimodeler de chaque classe (celui qui
    apparaît réellement dans model.getParameters()'s keys, ex. "Bckg" pour
    oimBackground, "eUD" pour oimEllipse) vers le nom complet de la classe.

    Dérivé du registre (donc de la version d'oimodeler réellement
    installée) plutôt que codé en dur : SHORT_TO_OIM ne correspondait pas
    aux vrais shortname pour plusieurs types (ex. "Bg"≠"Bckg") — un CSV/TXT
    exporté avec le vrai getParameters() ne se réimportait pas pour eux.

    Deux vraies collisions existent dans oimodeler même (deux classes
    partageant le même shortname) : "IR" pour oimIRing ET oimAEIRing,
    "SKER" pour oimESKRing ET oimESKGRing — le nom de classe le plus court
    (donc la variante "de base") gagne, un choix arbitraire mais
    déterministe. Le vrai type reste préservé pour tout modèle construit
    dans cette session (session_state.MODEL stocke le nom de classe
    complet) ; seule une réimportation externe utilisant l'abréviation
    ambiguë "IR"/"SKER" peut résoudre vers le mauvais des deux.
    """
    mapping = {}
    for full_name, info in sorted(registry.items(), key=lambda kv: len(kv[0])):
        cls = info.get('class')
        shortname = getattr(cls, 'shortname', None) if cls else None
        if shortname and shortname not in mapping:
            mapping[shortname] = full_name
    return mapping


def _resolve_comp_type(registry: dict, abbreviation: str) -> str | None:
    """Résout l'abréviation CSV/TXT vers le nom complet oimodeler.

    Ordre de résolution : (1) vrai shortname oimodeler (cas exact — c'est
    ce que produit réellement model.getParameters()) ; (2) SHORT_TO_OIM,
    conservé pour la compatibilité avec d'anciens exports ; (3) un nom de
    classe du registre se terminant par l'abréviation, en dernier recours.
    """
    shortname_map = _build_shortname_map(registry)
    if abbreviation in shortname_map:
        return shortname_map[abbreviation]
    if abbreviation in SHORT_TO_OIM and SHORT_TO_OIM[abbreviation] in registry:
        return SHORT_TO_OIM[abbreviation]
    abbr_lower = abbreviation.lower()
    for full_name in registry:
        if full_name.lower().endswith(abbr_lower):
            return full_name
    return None


def parse_csv_to_model(df: pd.DataFrame, registry: dict) -> tuple[dict | None, str]:
    """
    Convertit un DataFrame CSV en structure de modèle compatible session_state.MODEL.

    Colonnes attendues (insensibles à la casse) :
        Parameter | Value | Uncertainty | Min | Max | Free | At bound

    Format du champ Parameter : c{index}_{TypeAbbr}_{param}
    Exemple : c1_Pt_f, c2_EG_fwhm, c3_UD_d

    Retourne (model_dict, "") en cas de succès, ou (None, message_erreur).
    """
    # ── Normalisation des noms de colonnes ────────────────────────────
    rename_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('paramètre', 'parametre', 'parameter', 'param'):
            rename_map[col] = 'Paramètre'
        elif cl in ('valeur', 'value', 'val'):
            rename_map[col] = 'Valeur'
        elif cl in ('incertitude', 'uncertainty', 'error', 'erreur'):
            rename_map[col] = 'Incertitude'
        elif cl == 'min':
            rename_map[col] = 'Min'
        elif cl == 'max':
            rename_map[col] = 'Max'
        elif cl in ('libre', 'free'):
            rename_map[col] = 'Libre'
        elif cl in ('au bord', 'at bound', 'aubord', 'atbound'):
            rename_map[col] = 'Au bord'
    df = df.rename(columns=rename_map)

    required = {'Paramètre', 'Valeur', 'Min', 'Max', 'Libre'}
    missing  = required - set(df.columns)
    if missing:
        return None, f"Missing columns in CSV: {', '.join(missing)}"

    # ── Parsing ligne par ligne ───────────────────────────────────────
    comp_data: dict[int, dict] = {}

    for _, row in df.iterrows():
        param_full = str(row['Paramètre']).strip()
        parts = param_full.split('_')
        if len(parts) < 3:
            return None, (
                f"Invalid parameter format: « {param_full} »\n"
                f"Expected: c{{n}}_{{Type}}_{{param}}  (e.g.: c1_UD_d)"
            )
        try:
            comp_idx = int(parts[0][1:])
        except ValueError:
            return None, f"Unreadable component index in « {param_full} »"

        type_abbr  = parts[1]
        param_name = '_'.join(parts[2:])

        def _flt(val, fallback=0.):
            try:
                return float(val)
            except (TypeError, ValueError):
                return fallback

        def _bool(val) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            return str(val).strip().lower() in ('true', '1', 'oui', 'yes', 'libre')

        value = _flt(row['Valeur'])
        lo    = _flt(row['Min'],  DEFAULT_PARAM_RANGES.get(param_name, (-1e9, 1e9))[0])
        hi    = _flt(row['Max'],  DEFAULT_PARAM_RANGES.get(param_name, (-1e9, 1e9))[1])
        free  = _bool(row.get('Libre', False))

        if comp_idx not in comp_data:
            comp_data[comp_idx] = {'type_abbr': type_abbr, 'params': {}}
        comp_data[comp_idx]['params'][param_name] = {
            'value': value, 'min': lo, 'max': hi, 'free': free,
        }

    # ── Construction de la liste de composants ────────────────────────
    components = []
    for idx in sorted(comp_data.keys()):
        cd        = comp_data[idx]
        type_abbr = cd['type_abbr']
        oim_type  = _resolve_comp_type(registry, type_abbr)
        if oim_type is None:
            return None, (
                f"Unknown component type: « {type_abbr} » (component c{idx}).\n"
                f"Recognized types: {', '.join(SHORT_TO_OIM.keys())}"
            )

        param_names  = registry[oim_type]['params']
        init_values  = {}
        param_ranges = {}
        free_params  = []

        for p in param_names:
            if p in cd['params']:
                pd_row = cd['params'][p]
                init_values[p]  = pd_row['value']
                param_ranges[p] = (pd_row['min'], pd_row['max'])
                if pd_row['free']:
                    free_params.append(p)
            else:
                init_values[p]  = DEFAULT_PARAM_INIT.get(p, 0.)
                param_ranges[p] = DEFAULT_PARAM_RANGES.get(p, (0., 100.))

        components.append({
            'type':           oim_type,
            'name':           f"c{idx}_{type_abbr}",
            'params':         param_names.copy(),
            'initial_values': init_values,
            'param_ranges':   param_ranges,
            'free_params':    free_params,
            'interpolators':  {},
        })

    if not components:
        return None, "No component found in the CSV."

    return {'components': components}, ""
