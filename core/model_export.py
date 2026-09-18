# core/model_export.py
"""
Export d'un modèle vers le format texte normalisé importable par la page
Modelling (voir core/csv_import.parse_csv_to_model — même colonnes, juste
séparées par des tabulations plutôt que des virgules, donc aucun nouveau
parseur n'est nécessaire : pd.read_csv(sep=None, engine='python') détecte
le séparateur automatiquement).

Deux façons de produire un tel fichier :
- model_to_txt() : utilisée PAR l'app elle-même pour exporter tout modèle
  sauvegardé en session (st.session_state.MODEL) — voir pages/fitting.py's
  zip de résultats.
- EXTERNAL_WRITER_SNIPPET : le code source d'une fonction autonome que
  l'utilisateur copie-colle dans SON PROPRE script oimodeler (hors de
  cette app) pour exporter n'importe quel modèle qu'il y construit, au
  même format, afin de le réimporter ensuite ici. Ne prend que le nom du
  modèle en argument — elle suppose une variable `model` déjà en portée,
  comme le code reproductible généré ailleurs dans l'app (core/code_generator.py).

Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import json

_HEADER = "Parameter\tValue\tUncertainty\tMin\tMax\tFree"

# Appended after the flat parameter table, on its own line, when the model
# has at least one enabled interpolator. Everything after this marker is
# an oimodeler_App-specific extension (one tab-separated row per
# interpolated parameter: key, macro, kwargs as JSON, bounds as JSON) that
# core.csv_import.split_interpolator_section()/parse_interpolator_section()
# read back to fully reconstruct interpolators on import. A plain CSV
# reader (or EXTERNAL_WRITER_SNIPPET, which never writes this section)
# simply never produces it, and pandas never sees it either — the app
# splits it off *before* handing the rest to pd.read_csv.
INTERPOLATOR_SECTION_MARKER = "# OIMODELER_APP_INTERPOLATORS"

# Same idea as INTERPOLATOR_SECTION_MARKER, for oim.oimParamNorm
# normalizations (core/normalization.py): one row per normalized
# parameter — key, norm, refs as JSON (a list of OTHER components' own
# "c{idx}_{TypeAbbr}" key prefixes + param names, not their free-form
# display names, so a reference stays resolvable after re-import even
# though parse_csv_to_model() always reassigns component names to that
# same "c{idx}_{TypeAbbr}" form regardless of what they were called here).
NORMALIZATION_SECTION_MARKER = "# OIMODELER_APP_NORMALIZATIONS"


def model_to_txt(model_dict: dict, registry: dict) -> str:
    """Sérialise un modèle sauvegardé (forme de session_state.MODEL[nom],
    { 'components': [...] }) au format tabulé Parameter/Value/Uncertainty/
    Min/Max/Free — le même que produit EXTERNAL_WRITER_SNIPPET à partir
    d'un vrai model.getParameters(), donc les deux sont mutuellement
    réimportables par la page Modelling.

    Les paramètres interpolés (oimInterp) ou normalisés (oimParamNorm)
    n'ont pas de triplet (valeur, min, max) unique représentable dans ce
    tableau plat — ils en sont donc omis, mais une section supplémentaire
    par mécanisme (voir INTERPOLATOR_SECTION_MARKER /
    NORMALIZATION_SECTION_MARKER) enregistre de quoi les reconstruire,
    afin que core.csv_import les recolle à l'import plutôt que de les
    perdre silencieusement.
    """
    components = model_dict.get('components', [])

    def _shortname(comp_type: str) -> str:
        comp_cls = registry.get(comp_type, {}).get('class')
        return getattr(comp_cls, 'shortname', comp_type) if comp_cls else comp_type

    key_prefix_of = {
        c['name']: f"c{i+1}_{_shortname(c['type'])}"
        for i, c in enumerate(components)
    }

    lines = [_HEADER]
    interp_lines = []
    norm_lines = []
    for c in components:
        prefix = key_prefix_of[c['name']]
        params = registry.get(c['type'], {}).get(
            'params', c.get('params', list(c['initial_values'].keys()))
        )
        interps = c.get('interpolators', {})
        norms   = c.get('normalizations', {})

        for p in params:
            key = f"{prefix}_{p}"
            if p in norms and norms[p].get('enabled', False):
                cfg = norms[p]
                refs = [
                    {"component": key_prefix_of.get(r['component'], r['component']),
                     "param": r['param']}
                    for r in cfg.get('refs', [])
                ]
                norm_lines.append(
                    f"{key}\t{cfg.get('norm', 1.0)!r}\t{json.dumps(refs)}"
                )
                continue
            if p in interps and interps[p].get('enabled', False):
                cfg = interps[p]
                interp_lines.append(
                    f"{key}\t{cfg['macro']}\t{json.dumps(cfg['kwargs'])}\t"
                    f"{json.dumps(cfg.get('bounds', {}))}"
                )
                continue
            value  = c['initial_values'].get(p, 0.)
            lo, hi = c.get('param_ranges', {}).get(p, (float('-inf'), float('inf')))
            free   = p in c.get('free_params', [])
            lines.append(f"{key}\t{value!r}\t\t{lo!r}\t{hi!r}\t{free}")

    out = "\n".join(lines) + "\n"
    if interp_lines:
        out += "\n" + INTERPOLATOR_SECTION_MARKER + "\n" + "\n".join(interp_lines) + "\n"
    if norm_lines:
        out += "\n" + NORMALIZATION_SECTION_MARKER + "\n" + "\n".join(norm_lines) + "\n"
    return out


# Shown as a copyable code block in Modelling > Import model, for use in
# the user's OWN oimodeler scripts (not executed by this app itself).
# Keep this in sync with model_to_txt()'s column order/format above —
# tests/test_model_export.py checks a file it writes round-trips through
# core.csv_import.parse_csv_to_model.
EXTERNAL_WRITER_SNIPPET = '''\
def write_model_to_txt(model_name):
    """Writes `model`'s current parameters to '<model_name>.txt', in the
    format oimodeler_App's Modelling > Import model tab reads back."""
    params = model.getParameters()
    with open(f"{model_name}.txt", "w") as f:
        f.write("Parameter\\tValue\\tUncertainty\\tMin\\tMax\\tFree\\n")
        for key, p in params.items():
            if not hasattr(p, "value"):
                # A derived parameter (e.g. oim.oimParamNorm) has no
                # independent value/bounds of its own — this plain format
                # has no way to represent it, so it's skipped rather than
                # crashing on the missing attribute. Reconfigure it after
                # import in the app's Normalization tab.
                continue
            # float(...) first: oimParam.min/max are numpy scalars (e.g.
            # numpy.float16), whose own repr (numpy>=2.0) prints as
            # "np.float16(-inf)" instead of a plain, re-parseable "-inf".
            value = float(p.value)
            error = float(p.error)
            lo    = float(p.min)
            hi    = float(p.max)
            f.write(f"{key}\\t{value!r}\\t{error!r}\\t{lo!r}\\t{hi!r}\\t{p.free}\\n")
'''
