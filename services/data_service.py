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
# 4. Application d'un filtre spectral (résultat mis en cache)
# ═══════════════════════════════════════════════════════════════════════════

def build_per_file_filters(
    file_filters: dict[str, dict], file_dtypes: dict[str, list[str]],
    file_order: list[str],
):
    """
    Construit, pour chaque fichier sélectionné, ses propres filtres
    (plage(s) de longueur d'onde, binning spectral, types de données gardés)
    — chacun ciblé UNIQUEMENT sur son propre index dans file_order via
    `targets=[idx]`, jamais `targets=idx` (un entier nu) ni `targets="all"` :
    oimDataFilterComponent.applyFilter() enveloppe un entier nu en
    `[idx]` de la même façon, donc un `targets=0` appliqué "par erreur"
    à toute la sélection ne cible en réalité QUE le premier fichier — c'est
    exactement le bug qui rendait le filtrage global inopérant sur tout
    fichier après le premier avant ce module.

    Paramètres
    ----------
    file_filters : dict
        { nom_fichier: {'wl_ranges': [(lo_m, hi_m), ...], 'bin': int,
                         'normalize_err': bool} }, en mètres (unité native
        oimodeler) — voir pages/data.py pour la construction depuis l'UI (µm).
        Une entrée absente, ou avec 'wl_ranges' vide et 'bin' <= 1, ne filtre
        pas ce fichier (no-op).
    file_dtypes : dict
        { nom_fichier: [types de données gardés (VIS2DATA, VISAMP, …)] }
    file_order : list[str]
        Noms de fichiers dans l'ordre exact utilisé pour construire oimData
        (déterminant l'index cible des filtres).
    """
    from config.constants import FITTABLE_DATA_TYPES  # noqa: PLC0415

    oim = get_oim()
    all_types = set(FITTABLE_DATA_TYPES)
    filters = []
    for idx, fname in enumerate(file_order):
        cfg = file_filters.get(fname, {})

        wl_ranges = cfg.get('wl_ranges') or []
        if wl_ranges:
            filters.append(oim.oimWavelengthRangeFilter(
                targets=[idx], wlRange=[list(r) for r in wl_ranges], method="cut",
            ))

        bin_size = cfg.get('bin', 1)
        if bin_size and bin_size > 1:
            filters.append(oim.oimWavelengthBinningFilter(
                targets=[idx], bin=bin_size,
                normalizeError=cfg.get('normalize_err', False),
            ))

        selected = file_dtypes.get(fname)
        if selected is not None and set(selected) < all_types:
            # Pas de filtre créé si tout est sélectionné (no-op) — même
            # convention que ci-dessus pour les plages/binning.
            filters.append(oim.oimKeepDataTypeFilter(dataType=list(selected), targets=[idx]))

    return filters


@st.cache_data(ttl=3600, max_entries=50)
def get_file_summary(filepath: str) -> dict:
    """
    Résumé lecture-seule (instrument, cible, date, config VLTI, couverture
    spectrale native) d'un fichier OIFITS — mis en cache par chemin.

    cache_data (pas cache_resource) : ne retourne que des types sérialisables
    (str/float/int/None), lus une fois pour toutes indépendamment de tout
    filtre appliqué ensuite sur les données.
    """
    from core.oifits_meta import read_file_summary  # noqa: PLC0415
    return read_file_summary(filepath)


@st.cache_data(ttl=600, max_entries=100)
def get_filtered_wavelengths_for_file(
    filepath: str, wl_ranges: tuple, bin_size: int, normalize_err: bool,
) -> list[float]:
    """
    Retourne les longueurs d'onde uniques d'UN SEUL fichier après application
    de son propre filtre (plage(s) + binning) — le contrôle de bonne
    application affiché sous chaque bloc de filtre par fichier (Data page).
    Mis en cache par combinaison (filepath, paramètres de filtre).

    Un objet oimData single-fichier n'a besoin d'aucun `targets` explicite
    (il n'y a qu'un seul fichier à l'index 0) — inutile de reproduire ici le
    ciblage par index utilisé par build_per_file_filters() pour la
    sélection multi-fichiers réelle.

    Utilise cache_data (sérialisable) car on ne retourne que des floats.
    """
    import numpy as np  # noqa: PLC0415
    oim  = get_oim()
    data = load_oifits(filepath)

    filters = []
    if wl_ranges:
        filters.append(oim.oimWavelengthRangeFilter(
            wlRange=[list(r) for r in wl_ranges], method="cut",
        ))
    if bin_size and bin_size > 1:
        filters.append(oim.oimWavelengthBinningFilter(
            bin=bin_size, normalizeError=normalize_err,
        ))

    if filters:
        data.setFilter(oim.oimDataFilter(filters))
        data.useFilter = True
    else:
        data.useFilter = False

    return list(np.unique(data.vect_wl))