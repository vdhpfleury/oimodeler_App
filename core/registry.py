# core/registry.py
"""
Registre des composants oimodeler.
Construit une seule fois au démarrage, sans dépendance Streamlit.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (nom de la classe oimodeler, params, description). `oimodeler` est tiré
# depuis HEAD sans pin de commit (voir docs/security_audit_2026-09.md V14) :
# une classe listée ici peut ne pas exister dans la version réellement
# installée. build_registry() ignore silencieusement (avec un warning loggé)
# toute entrée absente plutôt que de faire planter toute l'application.
_COMPONENT_SPECS: list[tuple[str, list[str], str]] = [
    ('oimPt',                   ['x','y','f'],                                                                     'Point source (star)'),
    ('oimBackground',           ['x','y','f'],                                                                     'Uniform background'),
    ('oimUD',                   ['x','y','f','d'],                                                                 'Uniform disk'),
    ('oimEllipse',              ['x','y','f','elong','pa','d'],                                                    'Uniform ellipse'),
    ('oimGauss',                ['x','y','f','fwhm'],                                                              'Gaussian disk'),
    ('oimEGauss',               ['x','y','f','elong','pa','fwhm'],                                                 'Gaussian ellipse'),
    ('oimIRing',                ['x','y','f','d'],                                                                 'Infinitesimal ring'),
    ('oimEIRing',               ['x','y','f','elong','pa','d'],                                                    'Infinitesimal elliptic ring'),
    ('oimRing',                 ['x','y','f','din','dout'],                                                        'Ring'),
    ('oimRing2',                ['x','y','f','d','w'],                                                             'IRing convolved with UD'),
    ('oimERing',                ['x','y','f','elong','pa','din','dout'],                                           'Elliptic ring'),
    ('oimERing2',               ['x','y','f','elong','pa','d','w'],                                                'Elliptic ring 2'),
    ('oimESKIRing',             ['x','y','f','elong','pa','d','skw','skwPa'],                                      'Asymmetric infinitesimal elliptic ring'),
    ('oimESKGRing',             ['x','y','f','elong','pa','d','fwhm','skw','skwPa'],                               'Asymmetric Gaussian elliptic ring'),
    ('oimESKRing',              ['x','y','f','elong','pa','din','dout','skw','skwPa'],                             'Asymmetric elliptic ring'),
    ('oimLorentz',              ['x','y','f','fwhm'],                                                              'Pseudo-Lorentzian'),
    ('oimELorentz',             ['x','y','f','elong','pa','fwhm'],                                                 'Elliptic pseudo-Lorentzian'),
    ('oimLinearLDD',            ['x','y','f','d','a'],                                                             'Linear limb darkening'),
    ('oimQuadLDD',              ['x','y','f','d','a1','a2'],                                                       'Quadratic limb darkening'),
    ('oimPowerLawLDD',          ['x','y','f','d','a'],                                                             'Power-law limb darkening'),
    ('oimSqrtLDD',              ['x','y','f','d','a1','a2'],                                                       'Square-root limb darkening'),
    ('oimAEIRing',              ['x','y','f','elong','pa','d','skw','skwPa'],                                      'Asymmetric infinitesimal elliptic ring (2)'),
    ('oimBox',                  ['x','y','f','dx','dy'],                                                           'Rectangular box'),
    ('oimGaussLorentz',         ['x','y','f','elong','pa','hlr','flor'],                                           'Gauss-Lorentzian'),
    ('oimStarHaloGaussLorentz', ['x','y','f','elong','pa','la','flor','fh','fs','fc','kc','ks','wl0'],             'Star + Gauss-Lorentz halo'),
    ('oimStarHaloIRing',        ['x','y','f','elong','pa','la','flor','fh','fs','fc','kc','ks','wl0','lkr','skw','skwPa'], 'Star + ring halo'),
    ('oimSpiral',               ['x','y','f','elong','pa','fwhm','P','width'],                                     'Spiral'),
    ('oimTempGrad',             ['x','y','f','dim','elong','pa','rin','rout','r0','T0','Mdust','q','p','kappa_abs','dist'], 'Temperature-gradient disk (radial dust density + temperature power laws)'),
    ('oim4CLDD',                ['x','y','f','d','a1','a2','a3','a4'],                                             '4-coefficient limb darkening'),
    ('oimExpRing',              ['x','y','f','elong','pa','dim','d','fwhm'],                                       'Exponential ring (radial profile)'),
    ('oimInnerRim',             ['x','y','f','pa','dim','d','h','incl'],                                           'Puffed-up inner rim (image-based)'),
]


def build_registry(oim) -> dict[str, dict]:
    """
    Construit le COMPONENT_REGISTRY à partir du module oimodeler.
    Appelé une seule fois via services/data_service.get_registry().

    Toute classe listée dans _COMPONENT_SPECS mais absente de `oim` (version
    d'oimodeler plus ancienne/différente de celle attendue) est ignorée avec
    un warning, plutôt que de lever une AttributeError qui empêcherait toute
    l'application de démarrer.
    """
    registry: dict[str, dict] = {}
    for name, params, description in _COMPONENT_SPECS:
        cls = getattr(oim, name, None)
        if cls is None:
            logger.warning(
                "oimodeler has no '%s' — skipping it in the component "
                "registry (installed oimodeler version differs from the "
                "one this list was written against).",
                name,
            )
            continue
        registry[name] = {'class': cls, 'params': params, 'description': description}
    return registry
