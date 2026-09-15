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

_HEADER = "Parameter\tValue\tUncertainty\tMin\tMax\tFree"


def model_to_txt(model_dict: dict, registry: dict) -> str:
    """Sérialise un modèle sauvegardé (forme de session_state.MODEL[nom],
    { 'components': [...] }) au format tabulé Parameter/Value/Uncertainty/
    Min/Max/Free — le même que produit EXTERNAL_WRITER_SNIPPET à partir
    d'un vrai model.getParameters(), donc les deux sont mutuellement
    réimportables par la page Modelling.

    Les paramètres interpolés (oimInterp) sont omis : il n'existe pas de
    triplet (valeur, min, max) unique qui les représente dans ce format
    plat — même limitation, documentée de la même façon, que
    core/code_generator.py pour le code reproductible.
    """
    lines = [_HEADER]
    for i, c in enumerate(model_dict.get('components', [])):
        comp_type = c['type']
        comp_cls  = registry.get(comp_type, {}).get('class')
        shortname = getattr(comp_cls, 'shortname', comp_type) if comp_cls else comp_type
        params    = registry.get(comp_type, {}).get(
            'params', c.get('params', list(c['initial_values'].keys()))
        )
        interps = c.get('interpolators', {})

        for p in params:
            if p in interps and interps[p].get('enabled', False):
                continue
            value  = c['initial_values'].get(p, 0.)
            lo, hi = c.get('param_ranges', {}).get(p, (float('-inf'), float('inf')))
            free   = p in c.get('free_params', [])
            key    = f"c{i+1}_{shortname}_{p}"
            lines.append(f"{key}\t{value!r}\t\t{lo!r}\t{hi!r}\t{free}")

    return "\n".join(lines) + "\n"


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
            # float(...) first: oimParam.min/max are numpy scalars (e.g.
            # numpy.float16), whose own repr (numpy>=2.0) prints as
            # "np.float16(-inf)" instead of a plain, re-parseable "-inf".
            value = float(p.value)
            error = float(p.error)
            lo    = float(p.min)
            hi    = float(p.max)
            f.write(f"{key}\\t{value!r}\\t{error!r}\\t{lo!r}\\t{hi!r}\\t{p.free}\\n")
'''
