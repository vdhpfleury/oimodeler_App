# config/constants.py
"""
Constantes globales de l'application OIModeler.
Ce module est purement déclaratif : aucune dépendance Streamlit ou oimodeler.
"""

# ── Couleurs et styles pour les graphes multi-composants ──────────────────
COMP_COLORS: list[str] = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'cyan']
COMP_STYLES: list[str] = ['-', '--', ':', '-.', '-', '--', ':']

# ── Paramètres par défaut ─────────────────────────────────────────────────
DEFAULT_PARAM_RANGES: dict[str, tuple] = {
    'x': (-50., 50.), 'y': (-50., 50.), 'f': (0., 1.),
    'd': (0., 100.), 'din': (0., 80.), 'dout': (0., 100.),
    'fwhm': (0., 100.), 'elong': (1., 5.), 'pa': (-180., 180.),
    'skw': (0., 5.), 'skwPa': (-180., 180.), 'w': (0., 20.),
    'a': (0., 1.), 'a1': (0., 1.), 'a2': (0., 1.),
    'dx': (0., 50.), 'dy': (0., 50.), 'hlr': (0., 50.),
    'flor': (0., 1.), 'la': (0., 50.), 'fh': (0., 1.),
    'fs': (0., 1.), 'fc': (0., 1.), 'kc': (0., 10.),
    'ks': (0., 10.), 'wl0': (0., 10e-6), 'lkr': (0., 1.),
}

DEFAULT_PARAM_INIT: dict[str, float] = {
    'x': 0., 'y': 0., 'f': 0.5, 'd': 10., 'din': 5., 'dout': 20.,
    'fwhm': 5., 'elong': 1.5, 'pa': 0., 'skw': 0.5, 'skwPa': 0.,
    'w': 5., 'a': 0.5, 'a1': 0.3, 'a2': 0.2, 'dx': 10., 'dy': 10.,
    'hlr': 5., 'flor': 0.5, 'la': 5., 'fh': 0.5, 'fs': 0.5, 'fc': 0.5,
    'kc': 1., 'ks': 1., 'wl0': 3e-6, 'lkr': 0.5,
}

# ── Mapping abréviation CSV → nom complet oimodeler ──────────────────────
SHORT_TO_OIM: dict[str, str] = {
    'Pt': 'oimPt', 'Bg': 'oimBackground', 'UD': 'oimUD',
    'El': 'oimEllipse', 'Ga': 'oimGauss', 'EG': 'oimEGauss',
    'IR': 'oimIRing', 'EIR': 'oimEIRing', 'Ri': 'oimRing',
    'Ri2': 'oimRing2', 'ERi': 'oimERing', 'ERi2': 'oimERing2',
    'ESKIR': 'oimESKIRing', 'ESKGR': 'oimESKGRing', 'ESKRi': 'oimESKRing',
    'Lo': 'oimLorentz', 'ELo': 'oimELorentz', 'LLDD': 'oimLinearLDD',
    'QLDD': 'oimQuadLDD', 'PLLDD': 'oimPowerLawLDD', 'SqLDD': 'oimSqrtLDD',
    'AEIR': 'oimAEIRing', 'Box': 'oimBox', 'GL': 'oimGaussLorentz',
    'SHGL': 'oimStarHaloGaussLorentz', 'SHIR': 'oimStarHaloIRing',
}
