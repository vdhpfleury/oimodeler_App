# services/data_service.py
"""
Couche service : objets lourds partagés entre pages et utilisateurs.

Règles :
- st.cache_resource  → objets NON sérialisables, partagés entre tous les users
                       (oimodeler module, connexions, registry)
- st.cache_data      → données sérialisables avec TTL et max_entries contrôlés

IMPORTANT : les objets retournés par cache_resource sont partagés entre
tous les utilisateurs sur le même worker. Ne jamais les muter directement.
"""
from __future__ import annotations

import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════
# 1. Module oimodeler (lazy import – chargé une seule fois)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_oim():
    """
    Importe oimodeler une seule fois pour toute la durée de vie du serveur.
    Partagé entre toutes les pages et tous les utilisateurs.

    Pourquoi cache_resource et pas import module-level ?
    → Le lazy import via cache_resource évite de charger oimodeler lors du
      démarrage de pages qui n'en ont pas besoin, et garantit une instance unique.
    """
    import oimodeler as oim  # noqa: PLC0415
    return oim


# ═══════════════════════════════════════════════════════════════════════════
# 2. Registre des composants (dépend de oim, construit une seule fois)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_registry() -> dict:
    """
    Construit et met en cache le registre des composants.
    Dépend de get_oim() donc bénéficie du même cycle de vie.
    """
    from core.registry import build_registry  # noqa: PLC0415
    return build_registry(get_oim())


# ═══════════════════════════════════════════════════════════════════════════
# 3. Chargement des fichiers OIFITS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(ttl=3600, max_entries=20)
def load_oifits(filepath: str):
    """
    Charge un fichier OIFITS et le met en cache par chemin de fichier.

    Paramètres
    ----------
    filepath : str
        Chemin absolu vers le fichier .fits / .oifits sur le disque temporaire.

    Notes
    -----
    - cache_resource : l'objet oimData n'est pas sérialisable (pickle).
    - ttl=3600 : expire après 1 heure pour libérer la RAM automatiquement.
    - max_entries=20 : évite la croissance illimitée du cache en session longue.
    - NE PAS muter l'objet retourné directement : appliquer les filtres sur
      une copie ou via setFilter() (qui modifie l'état interne mais est idempotent).
    """
    oim = get_oim()
    return oim.oimData(filepath)


@st.cache_resource(ttl=3600, max_entries=20)
def load_oifits_multi(filepaths: tuple):
    """
    Charge et fusionne plusieurs fichiers OIFITS en un seul objet oimData.
    La clé de cache est le tuple ordonné des chemins — tout changement de
    sélection invalide automatiquement le cache.

    Paramètres
    ----------
    filepaths : tuple
        Tuple de chemins absolus (utiliser tuple, pas list, pour le cache).
    """
    oim = get_oim()
    if len(filepaths) == 1:
        return oim.oimData(filepaths[0])
    return oim.oimData(list(filepaths))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Filtres génériques (registre core/filter_registry.py) — session-wide,
#    partagés par les pages Data / Fitting / Modelling
# ═══════════════════════════════════════════════════════════════════════════

def build_filters_from_specs(applied_filters: list[dict]):
    """
    Construit les instances de filtre oimodeler réelles à partir de leurs
    spécifications sérialisables (voir services/session.py's
    'applied_filters' : { 'filter_class': str, 'kwargs': dict, ... }).

    Ne stocke jamais d'instance de filtre "vivante" dans session_state —
    seules ces specs (JSON-sérialisables) le sont ; les instances réelles
    sont reconstruites ici à la demande, cohérent avec la règle du projet
    de ne jamais garder d'objet oimodeler lourd/non sérialisable en session.

    Toute clé de kwargs dont la valeur est None est omise plutôt que
    passée telle quelle : la plupart des filtres itèrent sur
    self.params['arr']/['targets'] sans vérifier None (ils attendent soit
    "all" soit une liste) — passer explicitement None casserait le filtre
    au lieu de laisser la classe utiliser son propre défaut ("all").
    """
    oim = get_oim()
    filters = []
    for spec in applied_filters:
        filter_cls = getattr(oim, spec["filter_class"], None)
        if filter_cls is None:
            continue  # filtre inconnu de cette version d'oimodeler — ignoré
        kwargs = {k: v for k, v in spec.get("kwargs", {}).items() if v is not None}
        filters.append(filter_cls(**kwargs))
    return filters


@st.cache_data(ttl=3600, max_entries=50)
def get_filter_metadata(filepath: str, display_name: str):
    """
    Table de métadonnées par ligne (cible/array/datatype/baseline/
    télescopes/plage λ) d'UN fichier OIFITS — alimente les widgets en
    cascade du sélecteur de filtre générique (Data page).

    cache_data (pas cache_resource) : ne retourne qu'un DataFrame de
    types simples, indépendant de tout filtre appliqué ensuite.
    """
    from core.oifits_meta import extract_filter_metadata  # noqa: PLC0415
    return extract_filter_metadata(filepath, display_name)


def get_active_data(selected_files: list[str]):
    """
    Point d'entrée UNIQUE pour obtenir l'objet oimData actif, filtres
    appliqués — utilisé identiquement par les pages Data, Fitting et
    Modelling puisque les filtres sont désormais partagés pour toute la
    session (st.session_state.applied_filters), plutôt que reconstruits
    indépendamment par chaque page.

    Lève ValueError("No file selected.") si `selected_files` est vide,
    après résolution via l'allowlist loaded_files (V2) — jamais de
    reconstruction de chemin depuis une valeur de widget.
    """
    from services.storage import resolve_selected_paths  # noqa: PLC0415

    paths = resolve_selected_paths(selected_files)
    if not paths:
        raise ValueError("No file selected.")

    data = load_oifits_multi(tuple(paths))

    oim = get_oim()
    filters = build_filters_from_specs(st.session_state.get("applied_filters", []))
    data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True

    return data