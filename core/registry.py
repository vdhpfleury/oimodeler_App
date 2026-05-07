# core/registry.py
"""
Registre des composants oimodeler.
Construit une seule fois au démarrage, sans dépendance Streamlit.
"""
from __future__ import annotations


def build_registry(oim) -> dict[str, dict]:
    """
    Construit le COMPONENT_REGISTRY à partir du module oimodeler.
    Appelé une seule fois via services/data_service.get_registry().
    """
    return {
        'oimPt':                  {'class': oim.oimPt,                  'params': ['x','y','f'],                                                        'description': 'Point source (star)'},
        'oimBackground':          {'class': oim.oimBackground,          'params': ['x','y','f'],                                                        'description': 'Uniform background'},
        'oimUD':                  {'class': oim.oimUD,                  'params': ['x','y','f','d'],                                                    'description': 'Uniform disk'},
        'oimEllipse':             {'class': oim.oimEllipse,             'params': ['x','y','f','elong','pa','d'],                                       'description': 'Uniform ellipse'},
        'oimGauss':               {'class': oim.oimGauss,               'params': ['x','y','f','fwhm'],                                                'description': 'Gaussian disk'},
        'oimEGauss':              {'class': oim.oimEGauss,              'params': ['x','y','f','elong','pa','fwhm'],                                    'description': 'Gaussian ellipse'},
        'oimIRing':               {'class': oim.oimIRing,               'params': ['x','y','f','d'],                                                    'description': 'Infinitesimal ring'},
        'oimEIRing':              {'class': oim.oimEIRing,              'params': ['x','y','f','elong','pa','d'],                                       'description': 'Infinitesimal elliptic ring'},
        'oimRing':                {'class': oim.oimRing,                'params': ['x','y','f','din','dout'],                                           'description': 'Ring'},
        'oimRing2':               {'class': oim.oimRing2,               'params': ['x','y','f','d','w'],                                               'description': 'IRing convolved with UD'},
        'oimERing':               {'class': oim.oimERing,               'params': ['x','y','f','elong','pa','din','dout'],                              'description': 'Elliptic ring'},
        'oimERing2':              {'class': oim.oimERing2,              'params': ['x','y','f','elong','pa','d','w'],                                   'description': 'Elliptic ring 2'},
        'oimESKIRing':            {'class': oim.oimESKIRing,            'params': ['x','y','f','elong','pa','d','skw','skwPa'],                         'description': 'Asymmetric infinitesimal elliptic ring'},
        'oimESKGRing':            {'class': oim.oimESKGRing,            'params': ['x','y','f','elong','pa','d','fwhm','skw','skwPa'],                  'description': 'Asymmetric Gaussian elliptic ring'},
        'oimESKRing':             {'class': oim.oimESKRing,             'params': ['x','y','f','elong','pa','din','dout','skw','skwPa'],                'description': 'Asymmetric elliptic ring'},
        'oimLorentz':             {'class': oim.oimLorentz,             'params': ['x','y','f','fwhm'],                                                'description': 'Pseudo-Lorentzian'},
        'oimELorentz':            {'class': oim.oimELorentz,            'params': ['x','y','f','elong','pa','fwhm'],                                    'description': 'Elliptic pseudo-Lorentzian'},
        'oimLinearLDD':           {'class': oim.oimLinearLDD,           'params': ['x','y','f','d','a'],                                               'description': 'Linear limb darkening'},
        'oimQuadLDD':             {'class': oim.oimQuadLDD,             'params': ['x','y','f','d','a1','a2'],                                         'description': 'Quadratic limb darkening'},
        'oimPowerLawLDD':         {'class': oim.oimPowerLawLDD,         'params': ['x','y','f','d','a'],                                               'description': 'Power-law limb darkening'},
        'oimSqrtLDD':             {'class': oim.oimSqrtLDD,             'params': ['x','y','f','d','a1','a2'],                                         'description': 'Square-root limb darkening'},
        'oimAEIRing':             {'class': oim.oimAEIRing,             'params': ['x','y','f','elong','pa','d','skw','skwPa'],                         'description': 'Asymmetric infinitesimal elliptic ring (2)'},
        'oimBox':                 {'class': oim.oimBox,                 'params': ['x','y','f','dx','dy'],                                             'description': 'Rectangular box'},
        'oimGaussLorentz':        {'class': oim.oimGaussLorentz,        'params': ['x','y','f','elong','pa','hlr','flor'],                              'description': 'Gauss-Lorentzian'},
        'oimStarHaloGaussLorentz':{'class': oim.oimStarHaloGaussLorentz,'params': ['x','y','f','elong','pa','la','flor','fh','fs','fc','kc','ks','wl0'],'description': 'Star + Gauss-Lorentz halo'},
        'oimStarHaloIRing':       {'class': oim.oimStarHaloIRing,       'params': ['x','y','f','elong','pa','la','flor','fh','fs','fc','kc','ks','wl0','lkr','skw','skwPa'],'description': 'Star + ring halo'},
    }
