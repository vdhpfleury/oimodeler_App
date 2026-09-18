# core/csv_import.py
"""
Import d'un modèle depuis un CSV de résultats.
Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from config.constants import DEFAULT_PARAM_RANGES, DEFAULT_PARAM_INIT, SHORT_TO_OIM
from core.model_export import INTERPOLATOR_SECTION_MARKER

# Matches the "..._interpN" suffix oimModel.getParameters() gives an
# interpolated parameter's sub-parameters (see core/interp_registry.py).
# Used only to detect — and warn about, rather than silently drop — rows
# from a file that has no INTERPOLATOR_SECTION_MARKER section (typically
# one written by EXTERNAL_WRITER_SNIPPET from a real model.getParameters(),
# which flattens interpolators with no macro/kwargs/bounds attached).
_INTERP_SUFFIX_RE = re.compile(r'^(.*)_interp\d+$')


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


def split_interpolator_section(raw_text: str) -> tuple[str, str]:
    """Sépare un fichier modèle TXT en (tableau_plat, section_interpolateurs).

    Si INTERPOLATOR_SECTION_MARKER est absent (CSV/TXT ordinaire, ou fichier
    produit par EXTERNAL_WRITER_SNIPPET), retourne (raw_text, "") — le
    tableau plat est alors le fichier entier, comportement inchangé.
    """
    if INTERPOLATOR_SECTION_MARKER not in raw_text:
        return raw_text, ""
    main, _, rest = raw_text.partition(INTERPOLATOR_SECTION_MARKER)
    return main, rest


def parse_interpolator_section(section_text: str) -> dict[tuple[int, str], dict]:
    """Parse la section INTERPOLATOR_SECTION_MARKER (voir model_to_txt()).

    Une ligne par paramètre interpolé : "c{idx}_{TypeAbbr}_{param}\tmacro\t
    kwargs_json\tbounds_json". Retourne {(comp_idx, param_name): {'enabled':
    True, 'macro':..., 'kwargs':..., 'bounds':...}}, indexé exactement comme
    les clés du tableau plat pour que parse_csv_to_model() les recolle au
    bon composant/paramètre.
    """
    result: dict[tuple[int, str], dict] = {}
    for line in section_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        key, macro, kwargs_json, bounds_json = parts[0], parts[1], parts[2], parts[3]
        key_parts = key.split('_')
        if len(key_parts) < 3:
            continue
        try:
            comp_idx = int(key_parts[0][1:])
        except ValueError:
            continue
        param_name = '_'.join(key_parts[2:])
        try:
            kwargs = json.loads(kwargs_json)
            bounds = json.loads(bounds_json)
        except json.JSONDecodeError:
            continue
        result[(comp_idx, param_name)] = {
            'enabled': True, 'macro': macro, 'kwargs': kwargs, 'bounds': bounds,
        }
    return result


def parse_csv_to_model(
    df: pd.DataFrame, registry: dict, interp_rows: dict | None = None,
) -> tuple[dict | None, str, list[str]]:
    """
    Convertit un DataFrame CSV en structure de modèle compatible session_state.MODEL.

    Colonnes attendues (insensibles à la casse) :
        Parameter | Value | Uncertainty | Min | Max | Free | At bound

    Format du champ Parameter : c{index}_{TypeAbbr}_{param}
    Exemple : c1_Pt_f, c2_EG_fwhm, c3_UD_d

    interp_rows : sortie de parse_interpolator_section(), ou None — les
    paramètres interpolés qu'il désigne sont reconstruits dans
    components[i]['interpolators'] plutôt que dans initial_values/
    param_ranges/free_params (qui restent à leurs valeurs par défaut pour
    ces clés, ComponentConfig.create_instance() ne les utilisant de toute
    façon jamais pour un paramètre interpolé — seul cfg['bounds'] compte).

    Retourne (model_dict, "", warnings) en cas de succès, ou
    (None, message_erreur, []) en cas d'échec. `warnings` signale les
    lignes "..._interpN" rencontrées SANS section d'interpolateurs
    correspondante (typiquement un fichier écrit par
    EXTERNAL_WRITER_SNIPPET, qui aplatit les interpolateurs sans laisser
    de trace de leur macro/kwargs/bornes) — leur valeur brute est ignorée
    plutôt que silencieusement mal assignée à un paramètre homonyme.
    """
    interp_rows = interp_rows or {}
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
        return None, f"Missing columns in CSV: {', '.join(missing)}", []

    # ── Parsing ligne par ligne ───────────────────────────────────────
    comp_data: dict[int, dict] = {}

    for _, row in df.iterrows():
        param_full = str(row['Paramètre']).strip()
        parts = param_full.split('_')
        if len(parts) < 3:
            return None, (
                f"Invalid parameter format: « {param_full} »\n"
                f"Expected: c{{n}}_{{Type}}_{{param}}  (e.g.: c1_UD_d)"
            ), []
        try:
            comp_idx = int(parts[0][1:])
        except ValueError:
            return None, f"Unreadable component index in « {param_full} »", []

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
    warnings: list[str] = []
    for idx in sorted(comp_data.keys()):
        cd        = comp_data[idx]
        type_abbr = cd['type_abbr']
        oim_type  = _resolve_comp_type(registry, type_abbr)
        if oim_type is None:
            return None, (
                f"Unknown component type: « {type_abbr} » (component c{idx}).\n"
                f"Recognized types: {', '.join(SHORT_TO_OIM.keys())}"
            ), []

        param_names   = registry[oim_type]['params']
        init_values   = {}
        param_ranges  = {}
        free_params   = []
        interpolators = {}

        for p in param_names:
            ir = interp_rows.get((idx, p))
            if ir is not None:
                interpolators[p] = ir
                init_values[p]  = DEFAULT_PARAM_INIT.get(p, 0.)
                param_ranges[p] = DEFAULT_PARAM_RANGES.get(p, (0., 100.))
                continue
            if p in cd['params']:
                pd_row = cd['params'][p]
                init_values[p]  = pd_row['value']
                param_ranges[p] = (pd_row['min'], pd_row['max'])
                if pd_row['free']:
                    free_params.append(p)
            else:
                init_values[p]  = DEFAULT_PARAM_INIT.get(p, 0.)
                param_ranges[p] = DEFAULT_PARAM_RANGES.get(p, (0., 100.))

        # Rows named "{base}_interpN" that DIDN'T resolve through
        # interp_rows above (no INTERPOLATOR_SECTION_MARKER section for
        # them — e.g. a file written by EXTERNAL_WRITER_SNIPPET from a
        # real model.getParameters(), which flattens interpolators with
        # no macro/kwargs/bounds attached) are otherwise silently dropped:
        # `param_name` never matches any real `p` in param_names. Warn
        # instead, naming the component/parameter, rather than pretending
        # the import fully succeeded.
        orphaned_bases = set()
        for raw_name in cd['params']:
            if raw_name in param_names:
                continue
            m = _INTERP_SUFFIX_RE.match(raw_name)
            if m and m.group(1) in param_names:
                orphaned_bases.add(m.group(1))
        for base in sorted(orphaned_bases):
            warnings.append(
                f"Component c{idx} ({type_abbr}): parameter "
                f"'{base}' looks interpolated in the source file "
                f"(found '{base}_interpN' rows), but this file has no "
                f"interpolator metadata section — its value was "
                f"ignored. Re-export with this app's model export, or "
                f"configure an interpolator for '{base}' on this "
                f"component manually after import."
            )

        components.append({
            'type':           oim_type,
            'name':           f"c{idx}_{type_abbr}",
            'params':         param_names.copy(),
            'initial_values': init_values,
            'param_ranges':   param_ranges,
            'free_params':    free_params,
            'interpolators':  interpolators,
        })

    if not components:
        return None, "No component found in the CSV.", []

    return {'components': components}, "", warnings
