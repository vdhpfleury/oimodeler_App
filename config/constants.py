# config/constants.py
"""
Constantes globales de l'application OIModeler.
"""

# ── Couleurs et styles pour les graphes multi-composants ──────────────────
COMP_COLORS: list[str] = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'cyan']
COMP_STYLES: list[str] = ['-', '--', ':', '-.', '-', '--', ':']

# ── Paramètres par défaut ─────────────────────────────────────────────────
DEFAULT_PARAM_RANGES: dict[str, tuple] = {
    'x': (-50., 50.), 'y': (-50., 50.), 'f': (0., 1.),
    'd': (0., 100.), 'din': (0., 80.), 'dout': (0., 100.),
    'fwhm': (0., 100.), 'elong': (1., 3.), 'pa': (0., 180.),
    'skw': (0., 1.), 'skwPa': (0., 180.), 'w': (0., 20.),
    'a': (0., 1.), 'a1': (0., 1.), 'a2': (0., 1.),
    'dx': (0., 50.), 'dy': (0., 50.), 'hlr': (0., 50.),
    'flor': (0., 1.), 'la': (0., 50.), 'fh': (0., 1.),
    'fs': (0., 1.), 'fc': (0., 1.), 'kc': (0., 10.),
    'ks': (0., 10.), 'wl0': (0., 10e-6), 'lkr': (0., 1.),
    'dim': (16, 1024), 'P': (0.01, 100), 'width': (0.001, 1000),
    # oimTempGrad / oim4CLDD / oimExpRing / oimInnerRim
    # rin's lower bound is >0, not 0: oimodeler's default radial grid is
    # logarithmic, and oimTempGrad raises ValueError("Logarithmic grid
    # requires rin > 0.") for rin<=0 on every model evaluation.
    'rin': (0.01, 50.), 'rout': (0., 200.), 'r0': (0., 50.),
    'T0': (10., 3000.), 'Mdust': (0., 5.), 'q': (-1., 0.), 'p': (-3., 3.),
    'kappa_abs': (0., 10.), 'dist': (0., 10000.),
    'a3': (0., 1.), 'a4': (0., 1.), 'h': (0., 10.), 'incl': (0., 90.),
}

DEFAULT_PARAM_INIT: dict[str, float] = {
    'x': 0., 'y': 0., 'f': 0.5, 'd': 10., 'din': 5., 'dout': 20.,
    'fwhm': 5., 'elong': 1.5, 'pa': 0., 'skw': 0.5, 'skwPa': 0.,
    'w': 5., 'a': 0.5, 'a1': 0.3, 'a2': 0.2, 'dx': 10., 'dy': 10.,
    'hlr': 5., 'flor': 0.5, 'la': 5., 'fh': 0.5, 'fs': 0.5, 'fc': 0.5,
    'kc': 1., 'ks': 1., 'wl0': 3e-6, 'lkr': 0.5, 'dim': 128, 'P': 0.1, 'width': 0.1,
    'rin': 0.5, 'rout': 5., 'r0': 1., 'T0': 300., 'Mdust': 0.2,
    'q': -0.5, 'p': 0., 'kappa_abs': 1., 'dist': 100.,
    'a3': 0.1, 'a4': 0.1, 'h': 1., 'incl': 30.,
}

# ── Types de données OIFITS utilisables pour le fit ───────────────────────
# oim.oimSimulator.compute()/oimFitterMinimize/oimFitterEmcee accept these
# generically via their `dataTypes` argument — no per-type special-casing.
FITTABLE_DATA_TYPES: list[str] = [
    'VIS2DATA', 'VISAMP', 'VISPHI', 'T3AMP', 'T3PHI', 'FLUXDATA',
]

# ── Bornes serveur pour les méthodes de fit (V4/V6/V7 — widget bounds are
# cosmetic only, these are what's actually enforced) ───────────────────────
MAX_EMCEE_WALKERS:   int = 64
MAX_EMCEE_STEPS:      int = 40000
# A grid point costs one oimSimulator.compute() call, same order as one
# Random-search iteration (capped at 1000) — cap the *total* grid (product
# across all axes) accordingly rather than per-axis alone.
MAX_GRID_AXIS_POINTS: int = 200
MAX_GRID_POINTS:      int = 5000

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
    'Gen_comp':'oimSpiral',
}
