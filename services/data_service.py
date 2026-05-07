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

@st.cache_data(ttl=600, max_entries=50)
def get_filtered_wavelengths(filepath: str, expr: str, bin_L: int, bin_N: int) -> list[float]:
    """
    Retourne les longueurs d'onde uniques après filtrage.
    Mis en cache par combinaison (filepath, paramètres de filtre).

    Utilise cache_data (sérialisable) car on ne retourne que des floats.
    """
    import numpy as np  # noqa: PLC0415
    oim  = get_oim()
    data = load_oifits(filepath)

    filters = []
    if expr:
        filters.append(oim.oimFlagWithExpressionFilter(expr=expr, keepOldFlag=False))
    if bin_L > 1:
        filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_L, normalizeError=False))
    if bin_N > 1:
        filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_N, normalizeError=False))

    if filters:
        data.setFilter(oim.oimDataFilter(filters))

    return list(np.unique(data.vect_wl))