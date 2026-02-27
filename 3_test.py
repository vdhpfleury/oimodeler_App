"""
OIModeler – Application Streamlit
=======================================================

"""

import io
import re
import copy
import pickle
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Dépendances optionnelles ────────────────────────────────────────────────
try:
    import oimodeler as oim
    OIM_AVAILABLE = True
except ImportError:
    OIM_AVAILABLE = False
    st.error("oimodeler n'est pas installé. Installez-le avec : pip install oimodeler")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# 0.  Configuration de la page  (DOIT être le premier appel Streamlit)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OIModeler",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Registre des composants
# ═══════════════════════════════════════════════════════════════════════════
COMPONENT_REGISTRY: dict[str, dict] = {
    'oimPt':                  {'class': oim.oimPt,                  'params': ['x','y','f'],                                                        'description': 'Point source (étoile)'},
    'oimBackground':          {'class': oim.oimBackground,          'params': ['x','y','f'],                                                        'description': 'Fond uniforme'},
    'oimUD':                  {'class': oim.oimUD,                  'params': ['x','y','f','d'],                                                    'description': 'Disque uniforme'},
    'oimEllipse':             {'class': oim.oimEllipse,             'params': ['x','y','f','elong','pa','d'],                                       'description': 'Ellipse uniforme'},
    'oimGauss':               {'class': oim.oimGauss,               'params': ['x','y','f','fwhm'],                                                'description': 'Disque gaussien'},
    'oimEGauss':              {'class': oim.oimEGauss,              'params': ['x','y','f','elong','pa','fwhm'],                                    'description': 'Ellipse gaussienne'},
    'oimIRing':               {'class': oim.oimIRing,               'params': ['x','y','f','d'],                                                    'description': 'Anneau infinitésimal'},
    'oimEIRing':              {'class': oim.oimEIRing,              'params': ['x','y','f','elong','pa','d'],                                       'description': 'Anneau elliptique infinitésimal'},
    'oimRing':                {'class': oim.oimRing,                'params': ['x','y','f','din','dout'],                                           'description': 'Anneau'},
    'oimRing2':               {'class': oim.oimRing2,               'params': ['x','y','f','d','w'],                                               'description': 'IRing convolu avec UD'},
    'oimERing':               {'class': oim.oimERing,               'params': ['x','y','f','elong','pa','din','dout'],                              'description': 'Anneau elliptique'},
    'oimERing2':              {'class': oim.oimERing2,              'params': ['x','y','f','elong','pa','d','w'],                                   'description': 'Anneau elliptique 2'},
    'oimESKIRing':            {'class': oim.oimESKIRing,            'params': ['x','y','f','elong','pa','d','skw','skwPa'],                         'description': 'Anneau ell. infin. asymétrique'},
    #'oimESKGRing':            {'class': oim.oimESKGRing,            'params': ['x','y','f','elong','pa','d','fwhm','skw','skwPa'],                  'description': 'Anneau ell. gaussien asymétrique'},
    'oimESKRing':             {'class': oim.oimESKRing,             'params': ['x','y','f','elong','pa','din','dout','skw','skwPa'],                'description': 'Anneau elliptique asymétrique'},
    'oimLorentz':             {'class': oim.oimLorentz,             'params': ['x','y','f','fwhm'],                                                'description': 'Pseudo Lorentzien'},
    'oimELorentz':            {'class': oim.oimELorentz,            'params': ['x','y','f','elong','pa','fwhm'],                                    'description': 'Pseudo Lorentzien elliptique'},
    'oimLinearLDD':           {'class': oim.oimLinearLDD,           'params': ['x','y','f','d','a'],                                               'description': 'Limb darkening linéaire'},
    'oimQuadLDD':             {'class': oim.oimQuadLDD,             'params': ['x','y','f','d','a1','a2'],                                         'description': 'Limb darkening quadratique'},
    #'oimPowerLawLDD':         {'class': oim.oimPowerLawLDD,         'params': ['x','y','f','d','a'],                                               'description': 'Limb darkening loi de puissance'},
    #'oimSqrtLDD':             {'class': oim.oimSqrtLDD,             'params': ['x','y','f','d','a1','a2'],                                         'description': 'Limb darkening racine carrée'},
    #'oimAEIRing':             {'class': oim.oimAEIRing,             'params': ['x','y','f','elong','pa','d','skw','skwPa'],                         'description': 'Anneau ell. infin. asymétrique (2)'},
    'oimBox':                 {'class': oim.oimBox,                 'params': ['x','y','f','dx','dy'],                                             'description': 'Boîte rectangulaire'},
    #'oimGaussLorentz':        {'class': oim.oimGaussLorentz,        'params': ['x','y','f','elong','pa','hlr','flor'],                              'description': 'Gauss-Lorentzien'},
    #'oimStarHaloGaussLorentz':{'class': oim.oimStarHaloGaussLorentz,'params': ['x','y','f','elong','pa','la','flor','fh','fs','fc','kc','ks','wl0'],'description': 'Étoile + halo Gauss-Lorentz'},
    #'oimStarHaloIRing':       {'class': oim.oimStarHaloIRing,       'params': ['x','y','f','elong','pa','la','flor','fh','fs','fc','kc','ks','wl0','lkr','skw','skwPa'],'description': 'Étoile + halo anneau'},
}

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

# Palette de couleurs automatique par composante
COMP_COLORS = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'cyan']
COMP_STYLES = ['-', '--', ':', '-.', '-', '--', ':']

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Initialisation centralisée du session_state
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    # Composants en cours de configuration
    'components':         [],
    'active_comp_name':   None,
    # Bibliothèque de modèles enregistrés
    'MODEL':              {},
    # Données OIFITS
    'loaded_files':       {},
    'data':               None,
    # Résultats d'optimisation
    'optimization_done':  False,
    'best_params':        None,
    'best_chi2':          None,
    'history':            [],
    'best_model':         None,
    # Résultats χ²
    'chi2_result':        None,
    # Résultats Emcee
    'Emcee_Result':       None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Classes & fonctions métier
# ═══════════════════════════════════════════════════════════════════════════

class ComponentConfig:
    """Encapsule la configuration d'un composant oimodeler."""

    def __init__(self, component_type: str, name: str | None = None,
                 initial_values: dict | None = None, param_ranges: dict | None = None,
                 free_params: list | None = None, interpolators: dict | None = None):
        if component_type not in COMPONENT_REGISTRY:
            raise ValueError(f"Type de composant inconnu : {component_type}")
        self.component_type  = component_type
        self.name            = name or component_type
        self.component_class = COMPONENT_REGISTRY[component_type]['class']
        self.param_names     = COMPONENT_REGISTRY[component_type]['params']
        self.initial_values  = initial_values or {}
        self.interpolators   = interpolators or {}
        self.param_ranges    = {
            p: (param_ranges.get(p) if param_ranges and p in param_ranges
                else DEFAULT_PARAM_RANGES.get(p, (0., 100.)))
            for p in self.param_names
        }
        self.free_params = (free_params if free_params is not None
                            else [p for p in self.param_names if p not in ('x', 'y')])

    # ------------------------------------------------------------------
    def _full_params(self, override: dict | None = None) -> dict:
        full = {p: self.initial_values.get(
                    p, 0. if p in ('x', 'y') else
                       0.5 if p == 'f' else
                       sum(self.param_ranges[p]) / 2)
                for p in self.param_names}
        full.update(override or {})
        return full

    def create_instance(self, param_values: dict | None = None,
                        wave_data: np.ndarray | None = None):
        full = self._full_params(param_values)

        # Interpolateurs
        for p, cfg in self.interpolators.items():
            if not cfg.get('enabled', False):
                continue
            if cfg.get('type') == 'blackbody':
                wl = wave_data if wave_data is not None else np.linspace(1e-6, 5e-6, 50)
                full[p] = oim.oimInterp('starWl', temp=cfg['temp'],
                                        dist=cfg['dist'], lum=cfg['lum'], wl=wl)
            else:
                full[p] = oim.oimInterp(cfg['var'],
                                        **{cfg['var']: cfg['wl']},
                                        values=cfg['values'])
        instance = self.component_class(**full)

        # Appliquer libre/fixe + bornes
        for param_name, param_obj in instance.params.items():
            short = param_name.split('_')[-1]
            if short in self.param_names:
                param_obj.free = short in self.free_params
                lo, hi = self.param_ranges.get(short, (None, None))
                param_obj.min = lo
                param_obj.max = hi
        return instance

    def generate_random_params(self) -> dict:
        return {
            p: np.random.uniform(*self.param_ranges[p])
            for p in self.free_params
            if not (p in self.interpolators and self.interpolators[p].get('enabled', False))
        }


# ── Helpers session_state ────────────────────────────────────────────────

def get_comp(name: str) -> dict | None:
    return next((c for c in st.session_state.components if c['name'] == name), None)


def make_comp_dict(comp_type: str, comp_name: str) -> dict:
    params = COMPONENT_REGISTRY[comp_type]['params']
    return {
        'type':           comp_type,
        'name':           comp_name,
        'params':         params.copy(),
        'initial_values': {p: DEFAULT_PARAM_INIT.get(p, 0.) for p in params},
        'param_ranges':   {p: DEFAULT_PARAM_RANGES.get(p, (0., 100.)) for p in params},
        'free_params':    [p for p in params if p not in ('x', 'y')],
        'interpolators':  {},
    }


def read_widget_values_into_comp(comp: dict):
    """Lit les valeurs des widgets et les stocke dans le dict composant."""
    for param in comp['params']:
        k_init = f"{comp['name']}_{param}_init"
        k_min  = f"{comp['name']}_{param}_min"
        k_max  = f"{comp['name']}_{param}_max"
        k_free = f"{comp['name']}_{param}_free"
        if k_init in st.session_state:
            comp['initial_values'][param] = st.session_state[k_init]
        if k_min in st.session_state and k_max in st.session_state:
            comp['param_ranges'][param] = (st.session_state[k_min], st.session_state[k_max])
        if k_free in st.session_state:
            is_free = st.session_state[k_free]
            if is_free and param not in comp['free_params']:
                comp['free_params'].append(param)
            elif not is_free and param in comp['free_params']:
                comp['free_params'].remove(param)


# ── Génération de modèles ────────────────────────────────────────────────
def build_oim_model(comp_list: list) -> oim.oimModel | None:
    """Instancie un oimModel à partir d'une liste de dicts composant."""
    if not comp_list:
        return None
    try:
        instances = [
            ComponentConfig(
                component_type=c['type'], name=c['name'],
                initial_values=c['initial_values'], param_ranges=c['param_ranges'],
                free_params=c['free_params'],
                interpolators=c.get('interpolators', {}),   # ✅ ajout
            ).create_instance()
            for c in comp_list
        ]
        return oim.oimModel(*instances)
    except Exception as e:
        st.error(f"Erreur construction modèle : {e}")
        return None
    
def generate_model_preview(comp_list: list) -> plt.Figure | None:
    model = build_oim_model(comp_list)
    if model is None:
        return None
    try:
        im = model.getImage(64, 0.5, fromFT=True)
        N = im.shape[0]
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(im ** 0.2, cmap='hot', origin='lower', extent=[-N//2, N//2, -N//2, N//2])
        ax.set_title('Aperçu (γ=0.2)', fontsize=6)
        ax.set_xlabel('X (px)', fontsize=6)
        ax.set_ylabel('Y (px)', fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        

        return fig
    except Exception as e:
        st.error(f"Erreur aperçu : {e}")
        return None


# ── Résultats & export ───────────────────────────────────────────────────

def get_result_df(model_or_fit, is_fit: bool = False) -> tuple[float | None, pd.DataFrame]:
    """Retourne (chi2r, DataFrame des paramètres)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        if is_fit:
            model_or_fit.printResults()
            params = model_or_fit.simulator.model.getParameters()
        else:
            params = model_or_fit.getParameters()

    output = buf.getvalue()
    match  = re.search(r"chi2r\s*=\s*([0-9.eE+\-]+)", output)
    chi2r  = float(match.group(1)) if match else None

    rows = []
    for name, p in params.items():
        at_min   = (p.min is not None) and abs(p.value - p.min) < 1e-10
        at_max   = (p.max is not None) and abs(p.value - p.max) < 1e-10
        rows.append({
            "Paramètre":   name,
            "Valeur":      p.value,
            "Incertitude": p.error,
            "Min":         p.min,
            "Max":         p.max,
            "Libre":       p.free,
            "Au bord":     at_min or at_max,
        })
    return chi2r, pd.DataFrame(rows)

def update_model_from_fit(new_name: str, base_name: str, fit_model, chi2r: float | None = None) -> dict:
    if base_name not in st.session_state.MODEL:
        raise KeyError(f"Modèle base '{base_name}' introuvable.")
    updated = copy.deepcopy(st.session_state.MODEL[base_name])
    params  = fit_model.getParameters()

    for full_name, p in params.items():
        parts = full_name.split("_")
        # ✅ L'index est dans parts[0] : "c1" → 0
        try:
            comp_index = int(parts[0][1:]) - 1
        except (ValueError, IndexError):
            continue

        # ✅ Le nom du paramètre est TOUT ce qui suit "c{n}_{TypeAbbr}_"
        # ex: "c1_ESKIR_skwPa" → type_part="ESKIR", param_name="skwPa"
        # ex: "c1_UD_d"        → type_part="UD",    param_name="d"
        if len(parts) < 3:
            continue
        param_name = "_".join(parts[2:])   # ✅ au lieu de parts[-1]

        if 0 <= comp_index < len(updated["components"]):
            comp = updated["components"][comp_index]
            if param_name in comp["initial_values"]:
                comp["initial_values"][param_name] = p.value
            if param_name in comp["param_ranges"]:
                comp["param_ranges"][param_name] = (p.min, p.max)
            if p.free:
                if param_name not in comp["free_params"]:
                    comp["free_params"].append(param_name)
            else:
                if param_name in comp["free_params"]:
                    comp["free_params"].remove(param_name)

    if chi2r is not None:
        updated["chi2r"] = chi2r

    st.session_state.MODEL[new_name] = updated
    return updated
# ── Import de modèle depuis CSV ─────────────────────────────────────────

# Table de correspondance abréviation → nom complet oimodeler
# (les abréviations viennent du nommage automatique oimodeler : c1_UD_d, c2_EG_fwhm…)
_SHORT_TO_OIM: dict[str, str] = {
    'Pt':      'oimPt',
    'Bg':      'oimBackground',
    'UD':      'oimUD',
    'El':      'oimEllipse',
    'Ga':      'oimGauss',
    'EG':      'oimEGauss',
    'IR':      'oimIRing',
    'EIR':     'oimEIRing',
    'Ri':      'oimRing',
    'Ri2':     'oimRing2',
    'ERi':     'oimERing',
    'ERi2':    'oimERing2',
    'ESKIR':   'oimESKIRing',
    'ESKGR':   'oimESKGRing',
    'ESKRi':   'oimESKRing',
    'Lo':      'oimLorentz',
    'ELo':     'oimELorentz',
    'LLDD':    'oimLinearLDD',
    'QLDD':    'oimQuadLDD',
    'PLLDD':   'oimPowerLawLDD',
    'SqLDD':   'oimSqrtLDD',
    'AEIR':    'oimAEIRing',
    'Box':     'oimBox',
    'GL':      'oimGaussLorentz',
    'SHGL':    'oimStarHaloGaussLorentz',
    'SHIR':    'oimStarHaloIRing',
}


def _resolve_comp_type(abbreviation: str) -> str | None:
    """
    Tente de retrouver le type oimodeler complet depuis une abréviation CSV.
    Stratégie : correspondance exacte dans _SHORT_TO_OIM, puis recherche
    insensible à la casse sur les noms complets du registre.
    """
    # 1. Table directe
    if abbreviation in _SHORT_TO_OIM:
        return _SHORT_TO_OIM[abbreviation]
    # 2. Correspondance insensible à la casse sur les noms complets
    abbr_lower = abbreviation.lower()
    for full_name in COMPONENT_REGISTRY:
        if full_name.lower().endswith(abbr_lower):
            return full_name
    return None


def parse_csv_to_model(df: pd.DataFrame) -> dict | tuple[None, str]:
    """
    Convertit un DataFrame issu d'un CSV de résultats en structure de modèle
    compatible avec st.session_state.MODEL.

    Format attendu des colonnes (insensibles à la casse) :
        Paramètre | Valeur | Incertitude | Min | Max | Libre | Au bord

    Le champ « Paramètre » a la forme  c{index}_{ShortType}_{param}
    ex : c1_Pt_f, c2_EG_fwhm, c3_UD_d …

    Retourne soit le dict modèle, soit (None, message_erreur).
    """
    # ── Normaliser les noms de colonnes ────────────────────────────────
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
        return None, f"Colonnes manquantes dans le CSV : {', '.join(missing)}"

    # ── Parser chaque ligne ────────────────────────────────────────────
    # Regrouper par composant (c1, c2, …)
    comp_data: dict[int, dict] = {}   # index → {type_abbr, params…}

    for _, row in df.iterrows():
        param_full = str(row['Paramètre']).strip()
        # Forme attendue : c{n}_{TypeAbbr}_{paramName}
        parts = param_full.split('_')
        if len(parts) < 3:
            return None, (f"Format de paramètre invalide : « {param_full} »\n"
                          f"Attendu : c{{n}}_{{Type}}_{{param}}  (ex: c1_UD_d)")
        try:
            comp_idx = int(parts[0][1:])   # "c1" → 1
        except ValueError:
            return None, f"Index de composant illisible dans « {param_full} »"

        type_abbr  = parts[1]
        param_name = '_'.join(parts[2:])   # gère les noms composés type skwPa

        # Valeurs numériques
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
            'value': value, 'min': lo, 'max': hi, 'free': free
        }

    # ── Construire la liste de composants ──────────────────────────────
    components = []
    for idx in sorted(comp_data.keys()):
        cd         = comp_data[idx]
        type_abbr  = cd['type_abbr']
        oim_type   = _resolve_comp_type(type_abbr)
        if oim_type is None:
            return None, (f"Type de composant inconnu : « {type_abbr} » "
                          f"(composant c{idx}).\n"
                          f"Types reconnus : {', '.join(_SHORT_TO_OIM.keys())}")

        param_names = COMPONENT_REGISTRY[oim_type]['params']
        init_values = {}
        param_ranges = {}
        free_params  = []

        for p in param_names:
            if p in cd['params']:
                pd_row = cd['params'][p]
                init_values[p]  = pd_row['value']
                param_ranges[p] = (pd_row['min'], pd_row['max'])
                if pd_row['free']:
                    free_params.append(p)
            else:
                # Paramètre absent du CSV : valeurs par défaut
                init_values[p]  = DEFAULT_PARAM_INIT.get(p, 0.)
                param_ranges[p] = DEFAULT_PARAM_RANGES.get(p, (0., 100.))

        # Nom du composant : c{idx}_{type_abbr}
        comp_name = f"c{idx}_{type_abbr}"

        components.append({
            'type':           oim_type,
            'name':           comp_name,
            'params':         param_names.copy(),
            'initial_values': init_values,
            'param_ranges':   param_ranges,
            'free_params':    free_params,
            'interpolators':  {},
        })

    if not components:
        return None, "Aucun composant trouvé dans le CSV."

    return {'components': components}, ""


# ── Optimisation aléatoire ───────────────────────────────────────────────

def random_search(data, component_configs: list, n_runs: int = 100,
                  seed: int | None = None, wave_data=None) -> tuple:
    if seed is not None:
        np.random.seed(seed)
    best_chi2   = float('inf')
    best_params = None
    history     = []
    progress    = st.progress(0)
    status      = st.empty()

    for run in range(n_runs):
        try:
            comps      = []
            run_params = {}
            for cfg in component_configs:
                rp   = cfg.generate_random_params()
                comps.append(cfg.create_instance(rp, wave_data=wave_data))
                run_params[cfg.name] = rp
            model  = oim.oimModel(*comps)
            sim    = oim.oimSimulator(data=data, model=model)
            sim.compute(computeChi2=True, computeSimulatedData=False)
            chi2r  = sim.chi2r
            history.append({'run': run + 1, 'chi2r': chi2r, 'params': run_params})
            if chi2r < best_chi2:
                best_chi2   = chi2r
                best_params = run_params
                status.success(f"✓ Run {run+1}/{n_runs} – nouveau meilleur χ²ᵣ = {chi2r:.4f}")
        except Exception as e:
            st.warning(f"Run {run+1} ignoré : {e}")
        progress.progress((run + 1) / n_runs)

    progress.empty()
    status.empty()
    return best_params, best_chi2, history


# ── Utilitaires graphiques ───────────────────────────────────────────────

def plot_oimdata(oim_data, data_type: str,
                 xmin: float, xmax: float, ymin: float, ymax: float,
                 xscale: str, yscale: str, title: str) -> plt.Figure:

    fig = plt.figure(figsize=(5, 4))

    # Tracé données complètes (fond, transparent)
    ax_bg = fig.add_subplot(111, projection='oimAxes')
    try:
        oim_data.useFilter = False
        ax_bg.oiplot(oim_data, "EFF_WAVE", data_type, alpha=0.2,
                     xunit="micron", color="byBaseline", errorbar=False)
    except Exception:
        pass

    # Tracé données filtrées (par-dessus, opaque) — nouvel axe superposé
    ax_fg = fig.add_subplot(111, projection='oimAxes')
    try:
        oim_data.useFilter = True
        ax_fg.oiplot(oim_data, "EFF_WAVE", data_type, alpha=1.0,
                     xunit="micron", color="byBaseline", errorbar=False)
        ax_fg.patch.set_visible(False)   # rendre le fond transparent
    except Exception:
        pass

    ax_fg.set_xscale(xscale); ax_fg.set_yscale(yscale)
    ax_fg.set_xlim(xmin, xmax); ax_fg.set_ylim(ymin, ymax)
    ax_fg.set_title(title); ax_fg.grid(True)

    return fig

def extract_model_image(model, img_size=64, img_scale=0.5, wl_value=None) -> np.ndarray:
    """
    Extrait et somme les images de chaque composante individuellement.
    Utilise showModel(fig, ax_array, data) et ferme la figure immédiatement.
    """
    data_sum = None
    for comp in model.components:
        try:
            m_single = oim.oimModel(comp)
            if wl_value is not None:
                fig_c, _, data_c = m_single.showModel(img_size, img_scale,
                                                       wl=[wl_value], fromFT=True)
            else:
                fig_c, _, data_c = m_single.showModel(img_size, img_scale,
                                                       fromFT=True)
            plt.close(fig_c)   # fermer tout de suite pour ne pas accumuler

            if data_sum is None:
                data_sum = data_c.copy()
            else:
                data_sum += data_c
        except Exception:
            pass

    if data_sum is None:
        data_sum = np.zeros((1, 1, img_size, img_size))
    return data_sum
def copy_axes_lines(src_ax, dst_ax):
    for line in src_ax.get_lines():
        dst_ax.plot(line.get_xdata(), line.get_ydata(),
                    linestyle=line.get_linestyle(),
                    marker=line.get_marker(),
                    color=line.get_color(),
                    label=line.get_label())   # ✅ ajout
    dst_ax.set_xlabel(src_ax.get_xlabel())
    dst_ax.set_ylabel(src_ax.get_ylabel())
    dst_ax.grid(True)

def decompose_model_flux(model: oim.oimModel, data) -> dict:
    """
    Pour chaque composante du modèle, crée un oimModel individuel
    et un oimSimulator associé, puis calcule les données simulées.
    
    Retourne un dict :
        {
            'full':  {'model': oimModel, 'sim': oimSimulator},
            'c1_UD': {'model': oimModel, 'sim': oimSimulator},
            'c2_EG': {'model': oimModel, 'sim': oimSimulator},
            ...
        }
    """
    result = {}

    # ── Modèle complet ───────────────────────────────────────────────
    sim_full = oim.oimSimulator(data=data, model=model)
    sim_full.compute(computeChi2=False, computeSimulatedData=True)
    result['full'] = {'model': model, 'sim': sim_full}

    # ── Une composante à la fois ─────────────────────────────────────
    for comp in model.components:
        # Le nom de la composante oimodeler (ex: "c1_UD", "c2_EG")
        comp_name = comp.name if hasattr(comp, 'name') else type(comp).__name__

        m_single = oim.oimModel(comp)
        sim_single = oim.oimSimulator(data=data, model=m_single)
        sim_single.compute(computeChi2=False, computeSimulatedData=True)

        result[comp_name] = {'model': m_single, 'sim': sim_single}

    return result

def plot_flux_decomposition(decomp: dict, data) -> plt.Figure:
    """
    Affiche FLUXDATA pour les données, le modèle complet,
    et chaque composante individuellement.
    """
    fig = plt.figure(figsize=(9, 5))
    ax  = plt.subplot(projection='oimAxes')

    # ── Données observées ────────────────────────────────────────────
    try:
        data.useFilter = True
        ax.oiplot(data, "EFF_WAVE", "FLUXDATA",
                  errorbar=True, xunit='micron',
                  kwargs_error={"alpha": 0.3},
                  color='grey', label='Données')
    except Exception as e:
        pass

    # ── Modèle complet ───────────────────────────────────────────────
    try:
        ax.oiplot(decomp['full']['sim'].simulatedData,
                  "EFF_WAVE", "FLUXDATA",
                  xunit='micron', color='red', lw=2, label='Modèle complet')
    except Exception as e:
        pass

    # ── Composantes individuelles ────────────────────────────────────
    comp_keys = [k for k in decomp if k != 'full']
    for i, key in enumerate(comp_keys):
        color = COMP_COLORS[i % len(COMP_COLORS)]
        ls    = COMP_STYLES[i % len(COMP_STYLES)]
        try:
            ax.oiplot(decomp[key]['sim'].simulatedData,
                      "EFF_WAVE", "FLUXDATA",
                      xunit='micron', color=color, ls=ls, label=key)
        except Exception as e:
            pass

    ax.set_ylabel('Flux density (Jy)', fontsize=10)
    ax.set_xlabel(r'Wavelength (µm)', fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ── Génère code Python ───────────────────────────────────────────────
def generate_fitting_code(method: str, result: dict, data_filename: str,
                           model_comps: list, filter_params: dict) -> str:
    """
    Génère le code Python reproductible pour une minimisation χ² ou Emcee.
    """
    lines = []

    # ── En-tête ───────────────────────────────────────────────────────
    lines += [
        "import numpy as np",
        "import oimodeler as oim",
        "import matplotlib.pyplot as plt",
        "",
        "# ═══════════════════════════════════════════════════════════",
        f"# Fitting method : {method}",
        "# ═══════════════════════════════════════════════════════════",
        "",
    ]

    # ── Chargement des données ────────────────────────────────────────
    lines += [
        "# ── 1. Chargement des données ──────────────────────────────",
        f'data = oim.oimData("{data_filename}")',
        "",
    ]

    # ── Filtre spectral ───────────────────────────────────────────────
    lines += ["# ── 2. Filtrage spectral ───────────────────────────────"]
    expr      = filter_params.get("expr", "")
    bin_L     = filter_params.get("bin_L", 1)
    bin_N     = filter_params.get("bin_N", 1)
    norm_L    = filter_params.get("norm_L", False)
    norm_N    = filter_params.get("norm_N", False)

    if expr:
        lines.append(f'f_wl = oim.oimFlagWithExpressionFilter(expr="{expr}", keepOldFlag=False)')
    lines += [
        f"f_bL = oim.oimWavelengthBinningFilter(targets=0, bin={bin_L}, normalizeError={norm_L})",
        f"f_bN = oim.oimWavelengthBinningFilter(targets=0, bin={bin_N}, normalizeError={norm_N})",
    ]
    if expr:
        lines.append("data.setFilter(oim.oimDataFilter([f_wl, f_bL, f_bN]))")
    else:
        lines.append("data.setFilter(oim.oimDataFilter([f_bL, f_bN]))")
    lines += ["data.useFilter = True", ""]

    # ── Construction du modèle ────────────────────────────────────────
    lines += ["# ── 3. Construction du modèle ──────────────────────────"]
    comp_var_names = []
    for i, c in enumerate(model_comps):
        vname     = f"comp{i+1}"
        comp_type = c["type"]
        params    = COMPONENT_REGISTRY.get(comp_type, {}).get("params",
                    c.get("params", list(c["initial_values"].keys())))
        interps   = c.get("interpolators", {})

        # Paramètres scalaires (non interpolés)
        scalar_params = {
            p: c["initial_values"].get(p, 0.)
            for p in params
            if p not in interps or not interps[p].get("enabled", False)
        }
        param_str = ", ".join(f"{p}={v!r}" for p, v in scalar_params.items())

        # Interpolateurs
        for p, cfg in interps.items():
            if not cfg.get("enabled", False):
                continue
            if cfg["type"] == "blackbody":
                wl_var = f"wl_{vname}_{p}"
                lines += [
                    f"{wl_var} = np.linspace(1e-6, 5e-6, 200)  # adapter si besoin",
                    f"interp_{vname}_{p} = oim.oimInterp('starWl', "
                    f"temp={cfg['temp']}, dist={cfg['dist']}, "
                    f"lum={cfg['lum']}, wl={wl_var})",
                ]
            else:
                wl_arr  = repr(cfg["wl"])
                val_arr = repr(cfg["values"])
                var_key = cfg.get("var", "wl")
                lines += [
                    f"interp_{vname}_{p} = oim.oimInterp('{var_key}', "
                    f"{var_key}=np.array({wl_arr}), values=np.array({val_arr}))",
                ]
            # Remplacer la valeur scalaire par l'interpolateur dans param_str
            param_str += f", {p}=interp_{vname}_{p}"

        lines.append(f"{vname} = oim.{comp_type}({param_str})")

        # Bornes et libre/fixe
        for p in params:
            lo, hi = c["param_ranges"].get(p, (None, None))
            free   = p in c.get("free_params", [])
            # Nom complet du paramètre oimodeler (ex: c1_UD_d)
            lines += [
                f"{vname}.getParameters()['{c['name']}_{p}'].set(min={lo!r}, max={hi!r}, free={free})"
            ]
        lines.append("")
        comp_var_names.append(vname)

    comp_args = ", ".join(comp_var_names)
    lines += [f"model = oim.oimModel({comp_args})", ""]

    # ── Fitting ───────────────────────────────────────────────────────
    dtypes_key = "opt_dtypes" if method == "chi2" else "emcee_dtypes"
    dtypes     = result.get("dtypes", ["VIS2DATA", "T3PHI"])
    dtypes_str = repr(dtypes)

    if method == "chi2":
        lines += [
            "# ── 4. Minimisation χ² ─────────────────────────────────",
            f"fitter = oim.oimFitterMinimize(data, model, dataTypes={dtypes_str})",
            "fitter.prepare()",
            "fitter.run()",
            "",
            "fitter.printResults()",
            "",
            "# ── 5. Visualisation ────────────────────────────────────",
            'fig, ax = fitter.simulator.plot(["VIS2DATA", "T3PHI"])',
            "plt.show()",
        ]
    else:  # emcee
        nwalkers = result.get("nwalkers", 32)
        nsteps   = result.get("nsteps",   1000)
        init     = result.get("init",     "gaussian")
        lines += [
            "# ── 4. Emcee MCMC ──────────────────────────────────────",
            f"fitter = oim.oimFitterEmcee(data, model, nwalkers={nwalkers},",
            f"                            dataTypes={dtypes_str})",
            f'fitter.prepare(init="{init}", samplerFile="/tmp/sampler_emcee.txt")',
            f"fitter.run(nsteps={nsteps}, progress=True)",
            "",
            "fitter.printResults()",
            "",
            "# ── 5. Visualisation ────────────────────────────────────",
            "fig_w, _ = fitter.walkersPlot(chi2limfact=5)",
            "fig_c, _ = fitter.cornerPlot(dchi2limfact=5)",
            'fig, ax  = fitter.simulator.plot(["VIS2DATA", "T3PHI"])',
            "plt.show()",
        ]

    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Interface utilisateur
# ═══════════════════════════════════════════════════════════════════════════
st.title("🔭 OIModeler")

tab_home, tab_visu, tab_model = st.tabs(["📋 Présentation", "🔬 Visualisation des composants", "⚙️ Modélisation"])

# ─────────────────────────────────────────────────────────────────────────
# TAB 0 – Présentation
# ─────────────────────────────────────────────────────────────────────────
with tab_home:
    st.markdown("""
    Interface interactive pour la modélisation de données d'interférométrie optique
    au format **OIFITS**, reposant sur la bibliothèque Python *oimodeler*.
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🔬 Visualiseur de composantes**\n\n"
                "Explorez et paramétrez interactivement chaque composante géométrique. "
                "Visualisez l'image en temps réel et générez le code Python associé.")
    with c2:
        st.info("**📂 Chargement & Filtrage**\n\n"
                "Importez des fichiers OIFITS, appliquez des filtres spectraux "
                "et visualisez VIS², T3PHI, FLUXDATA.")
    with c3:
        st.info("**📐 Modélisation & Fitting**\n\n"
                "Configurez un modèle multi-composantes, lancez une exploration aléatoire "
                "puis affinez par minimisation du χ² ou Emcee.")
    st.markdown("#### Workflow recommandé")
    st.markdown(
        "1. **Visualiseur** → choisir les composantes\n"
        "2. **Modélisation** → I. Charger les données OIFITS\n"
        "3. **Modélisation** → II. Filtrer & visualiser\n"
        "4. **Modélisation** → IV. Configurer & enregistrer le modèle\n"
        "5. **Modélisation** → V. Lancer le fitting"
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 – Visualisation des composants
# ─────────────────────────────────────────────────────────────────────────
with tab_visu:
    col_left, col_right = st.columns(2)

    VISU_COMPONENTS: dict[str, list] = {
        k: v['params'] for k, v in COMPONENT_REGISTRY.items()
        if k not in ('oimStarHaloGaussLorentz', 'oimStarHaloIRing')  # params avancés
    }

    with col_left:
        selected_comp = st.selectbox(
            "Choisir une composante",
            list(VISU_COMPONENTS.keys()),
            format_func=lambda x: f"{x}  —  {COMPONENT_REGISTRY[x]['description']}"
        )

        required = VISU_COMPONENTS[selected_comp]
        visu_params: dict = {}

        SLIDER_CFG: dict[str, tuple] = {
            'x':     ("x (position X mas)",    -50., 50., 0., 1.),
            'y':     ("y (position Y mas)",    -50., 50., 0., 1.),
            'f':     ("f (flux)",               0., 2., 1., 0.1),
            'd':     ("d (diamètre mas)",       0., 100., 40., 1.),
            'din':   ("din (diam. int. mas)",   0., 100., 30., 1.),
            'dout':  ("dout (diam. ext. mas)",  0., 100., 50., 1.),
            'w':     ("w (largeur mas)",        0., 50., 20., 1.),
            'dx':    ("dx (largeur X mas)",     0., 100., 30., 1.),
            'dy':    ("dy (hauteur Y mas)",     0., 100., 20., 1.),
            'elong': ("elong (élongation)",     1., 3., 1.5, 0.1),
            'pa':    ("pa (angle °)",           0., 180., 45., 5.),
            'fwhm':  ("fwhm (mas)",             1., 30., 10., 1.),
            'hlr':   ("hlr (mas)",              1., 30., 10., 1.),
            'flor':  ("flor",                   0., 1., 0.5, 0.05),
            'skw':   ("skw (asymétrie)",        0., 1., 0.3, 0.05),
            'skwPa': ("skwPa (angle asym. °)",  0., 180., 30., 5.),
            'a':     ("a",                      0., 1., 0.5, 0.05),
            'a1':    ("a1",                     0., 1., 0.3, 0.05),
            'a2':    ("a2",                     0., 1., 0.2, 0.05),
        }

        cols3 = st.columns(3)
        for i, param in enumerate(required):
            with cols3[i % 3]:
                cfg = SLIDER_CFG.get(param)
                if cfg:
                    label, mn, mx, dfl, step = cfg
                    visu_params[param] = st.slider(label, mn, mx, dfl, step,
                                                   key=f"visu_{param}")
                else:
                    visu_params[param] = st.number_input(param, value=0., key=f"visu_{param}")

        # Code Python généré
        st.subheader("Code Python associé")
        params_str = ",\n    ".join(f"{k}={v}" for k, v in visu_params.items())
        code = (f"component = oim.{selected_comp}(\n    {params_str}\n)\n"
                f"model = oim.oimModel(component)\n"
                f"im = model.getImage(256, 1, fromFT=True)\n\n"
                f"plt.figure()\nplt.imshow(im**0.2, cmap='hot')\nplt.show()")
        st.code(code, language='python')

    with col_right:
        try:
            comp_cls = COMPONENT_REGISTRY[selected_comp]['class']
            comp_inst = comp_cls(**visu_params)
            mdl  = oim.oimModel(comp_inst)
            im   = mdl.getImage(256, 1, fromFT=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            im_disp = ax.imshow(im ** 0.2, cmap='hot', origin='lower')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title(f'{selected_comp}  –  γ = 0.2')
            plt.colorbar(im_disp, ax=ax, label='Intensité (γ corrigée)')
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Impossible d'afficher le composant : {e}")

    with st.expander("ℹ️ Aide sur les paramètres"):
        st.markdown("""
| Param | Description |
|-------|-------------|
| x, y | Position du centre (mas) |
| f | Flux relatif |
| d | Diamètre (mas) |
| din / dout | Diamètres intérieur / extérieur (mas) |
| w | Largeur (mas) |
| dx / dy | Dimensions boîte (mas) |
| elong | Rapport d'aspect (ellipticité) |
| pa | Angle de position (°) |
| fwhm | Largeur à mi-hauteur (mas) |
| hlr | Rayon à mi-flux (mas) |
| flor | Fraction lorentzienne |
| skw / skwPa | Asymétrie et angle associé |
| a, a1, a2 | Coefficients de limb darkening |
        """)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 – Modélisation
# ─────────────────────────────────────────────────────────────────────────
with tab_model:

    # ── I. Chargement ──────────────────────────────────────────────────
    with st.expander("I. Chargement des données OIFITS", expanded=True):
        uploaded_files = st.file_uploader(
            "Charger un ou plusieurs fichiers OIFITS",
            type=['fits', 'oifits'],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.loaded_files:
                    try:
                        tmp = f"/tmp/{f.name}"
                        with open(tmp, "wb") as fh:
                            fh.write(f.getbuffer())
                        st.session_state.loaded_files[f.name] = oim.oimData(tmp)
                        st.success(f"✓ {f.name} chargé")
                    except Exception as exc:
                        st.error(f"Erreur ({f.name}) : {exc}")

    # ── II. Filtrage & visualisation ───────────────────────────────────
    with st.expander("II. Filtrage des données et affichage", expanded=True):
        if not st.session_state.loaded_files:
            st.info("Chargez au moins un fichier OIFITS.")
            st.stop()

        selected_file = st.selectbox("Jeu de données à afficher",
                                     list(st.session_state.loaded_files.keys()))
        st.session_state.data = st.session_state.loaded_files[selected_file]

        st.markdown("##### Filtre par longueur d'onde")
        
        n_ranges = st.radio("Nombre de domaines spectraux", [1, 2], horizontal=True,
                            key="n_wl_ranges")
        
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Domaine 1**")
            rca, rcb = st.columns(2)
            with rca:
                wl1_min = st.number_input("λ min (µm)", value=3.2, step=0.1,
                                          format="%.2f", key="wl1_min")
            with rcb:
                wl1_max = st.number_input("λ max (µm)", value=3.8, step=0.1,
                                          format="%.2f", key="wl1_max")
        with rc2:
            if n_ranges == 2:
                st.markdown("**Domaine 2**")
                rcc, rcd = st.columns(2)
                with rcc:
                    wl2_min = st.number_input("λ min (µm)", value=4.5, step=0.1,
                                              format="%.2f", key="wl2_min")
                with rcd:
                    wl2_max = st.number_input("λ max (µm)", value=5.0, step=0.1,
                                              format="%.2f", key="wl2_max")
        
        st.markdown("##### Binning spectral")
        #cb1, cb2 = st.columns(2)
        #with cb1:
        #    st.markdown("**Bande L**")
        #    bin_L  = st.slider("Bin L",  1, 20, 1,  key="bin_L")
        #    norm_L = st.toggle("Normaliser σ (L)", value=False, key="norm_L")
        #with cb2:
        #    st.markdown("**Bande N**")
        #    bin_N  = st.slider("Bin N",  1, 20, 1,  key="bin_N")
        #    norm_N = st.toggle("Normaliser σ (N)", value=False, key="norm_N")
        
        
        try:
            # ── Filtre(s) longueur d'onde via expression ─────────────────
            w1_lo = wl1_min * 1e-6
            w1_hi = wl1_max * 1e-6

            if n_ranges == 1:
                # Exclut tout ce qui est hors du domaine 1
                expr = f"(EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})"
            else:
                w2_lo = wl2_min * 1e-6
                w2_hi = wl2_max * 1e-6
                # Exclut ce qui est hors domaine 1 ET hors domaine 2
                expr = (
                    f"((EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})) & "
                    f"((EFF_WAVE<{w2_lo}) | (EFF_WAVE>{w2_hi}))"
                )

            f_wl = oim.oimFlagWithExpressionFilter(expr=expr, keepOldFlag=False)
            st.session_state["_last_filter_expr"] = expr
            #f_bL = oim.oimWavelengthBinningFilter(targets=0, bin=bin_L, normalizeError=norm_L)
            #f_bN = oim.oimWavelengthBinningFilter(targets=0, bin=bin_N, normalizeError=norm_N)

            st.session_state.data.setFilter(oim.oimDataFilter([f_wl]))#, f_bL, f_bN]))

            wave_data_filtered = np.unique(st.session_state.data.vect_wl)
            
            if n_ranges == 2:
                st.info(f"Après filtrage : {len(wave_data_filtered)} points  |  "
                        f"Domaine 1 : [{wl1_min:.2f}, {wl1_max:.2f}] µm  —  "
                        f"Domaine 2 : [{wl2_min:.2f}, {wl2_max:.2f}] µm")
            else:
                st.info(f"Après filtrage : {len(wave_data_filtered)} points  |  "
                        f"λ ∈ [{wave_data_filtered.min()*1e6:.3f}, "
                        f"{wave_data_filtered.max()*1e6:.3f}] µm")


        except Exception as exc:
            st.warning(f"Impossible d'appliquer le filtre : {exc}")
            wave_data_filtered = None

        st.markdown("#### Visualisation des observables")

        def _obs_controls(prefix: str, xd: float = 3.0, xu: float = 4.0,
                          yd: float = 0., yu: float = 1.) -> tuple:
            c_l, c_r = st.columns(2)
            with c_l:
                xs = st.selectbox("Échelle X", ["linear", "log"], key=f"{prefix}_xs")
                xmn = st.number_input("Xmin (µm)", value=xd, key=f"{prefix}_xmn")
                xmx = st.number_input("Xmax (µm)", value=xu, key=f"{prefix}_xmx")
            with c_r:
                ys = st.selectbox("Échelle Y", ["linear", "log"], key=f"{prefix}_ys")
                ymn = st.number_input("Ymin", value=yd, key=f"{prefix}_ymn")
                ymx = st.number_input("Ymax", value=yu, key=f"{prefix}_ymx")
            return xmn, xmx, ymn, ymx, xs, ys

        data = st.session_state.data
        fig5, (ax1, ax2, ax3) = plt.subplots(ncols=3,figsize=(15, 4),subplot_kw={'projection': 'oimAxes'})

        ax1.oiplot(data,"SPAFREQ","VIS2DATA",xunit="cycle/mas",color="byBaseline",errorbar=True)
        #ax1.set_xlim(2.85, 5)
        ax1.set_ylim(0, 1)
        ax1.set_title("VIS2")

        ax2.oiplot(data,"EFF_WAVE","T3PHI",xunit="micron",color="byBaseline",errorbar=True)
        ax2.set_title("T3PHI")
        ax2.set_ylim(-15, 5)

        ax3.oiplot(data,"EFF_WAVE","FLUXDATA", xunit="micron",  errorbar=True)
        ax3.set_title("FLUXDATA")

        plt.tight_layout()
        st.pyplot(fig5)

 
    # ── IV. Configuration du modèle ────────────────────────────────────
    with st.expander("IV. Configuration du modèle", expanded=False):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Model", "Loading CSV model", "Interpolateurs", "Résumé des models", "Gestion des modèles"])
        with tab1 : 
            col_A, col_B, col_C = st.columns([1.5, 1, 1.5])
            with col_A:
                st.markdown("##### A. Initialiser le modèle")

                # ── Charger un modèle existant ────────────────────────
                if st.session_state.MODEL:
                    load_existing = st.checkbox("Charger un modèle existant", key="load_existing_cb")
                    if load_existing:
                        model_to_load = st.selectbox(
                            "Modèle à charger",
                            sorted(list(st.session_state.MODEL.keys())),
                            key="model_to_load_sel",
                        )
                        if st.button("📂 Charger", use_container_width=True, key="btn_load_existing"):
                            loaded = st.session_state.MODEL[model_to_load]
                            st.session_state.components = [
                                {
                                    **c,
                                    "params": COMPONENT_REGISTRY.get(c["type"], {}).get("params", c.get("params", [])),
                                    "interpolators": c.get("interpolators", {}),
                                }
                                for c in loaded["components"]
                            ]
                            st.session_state.active_comp_name = (
                                st.session_state.components[0]["name"]
                                if st.session_state.components else None
                            )
                            st.success(f"✅ Modèle **{model_to_load}** chargé pour édition !")
                            st.rerun()
                    
                
                model_name = st.text_input("Nom du modèle", value="", placeholder="ex: disque_uniforme",
                                           label_visibility="visible", key="model_name_input")
                st.write("**Ajouter un composant**")
                comp_type_sel = st.selectbox(
                    "Type", list(COMPONENT_REGISTRY.keys()), label_visibility="collapsed",
                    format_func=lambda x: f"{x} — {COMPONENT_REGISTRY[x]['description']}")
                comp_name_inp = st.text_input("Nom du composant", value=comp_type_sel,
                                            key="new_comp_name")
                if st.button("➕ Ajouter", use_container_width=False, type="primary"):
                    existing = [c['name'] for c in st.session_state.components]
                    final    = comp_name_inp
                    if final in existing:
                        suf = 2
                        while f"{comp_name_inp}_{suf}" in existing:
                            suf += 1
                        final = f"{comp_name_inp}_{suf}"
                    st.session_state.components.append(make_comp_dict(comp_type_sel, final))
                    st.session_state.active_comp_name = final
                    st.rerun()
                



            
            with col_B:
                if not st.session_state.components:
                    st.info("Ajoutez un composant.")
                else:
                    names = [c['name'] for c in st.session_state.components]
                    if st.session_state.active_comp_name not in names:
                        st.session_state.active_comp_name = names[0]
                    st.markdown("##### B. Gestion des composants")

                    st.write("**Composant actif**")
                    active_name = st.radio("", names,
                                        index=names.index(st.session_state.active_comp_name),
                                        key="active_radio",
                                        label_visibility="collapsed")
                    if active_name != st.session_state.active_comp_name:
                        st.session_state.active_comp_name = active_name
                        st.rerun()
                    comp_active = get_comp(active_name)
                    if comp_active:
                        read_widget_values_into_comp(comp_active)
                    
                    st.write("**Supprimer un composant**")
                    if st.session_state.components:
                        to_del = st.selectbox("", [c['name'] for c in st.session_state.components],
                                            label_visibility="collapsed", key="del_select")
                        if st.button("🗑️ Supprimer", use_container_width=True):
                            st.session_state.components = [
                                c for c in st.session_state.components if c['name'] != to_del]
                            names = [c['name'] for c in st.session_state.components]
                            st.session_state.active_comp_name = names[0] if names else None
                            st.rerun()

            with col_C:
                if st.session_state.components:
                    for c in st.session_state.components:
                        read_widget_values_into_comp(c)
                    with st.spinner("Rendu …", show_time=True):
                        fig_prev = generate_model_preview(st.session_state.components)
                        if fig_prev:
                            st.pyplot(fig_prev, use_container_width=False)
                            plt.close(fig_prev)
                    

                else:
                    st.info("Ajoutez un composant pour voir l'aperçu.")
                


            # ── Éditeur de paramètres du composant actif ───────────────────
            comp_edit = (get_comp(st.session_state.active_comp_name)
                        if st.session_state.active_comp_name else None)
            if comp_edit is None:
                st.info("Sélectionnez un composant pour l'éditer.")
            else:
                st.markdown(f"##### C. Configuration : **{comp_edit['name']}** ({comp_edit['type']})")
                n_cols = min(len(comp_edit['params']), 6)
                param_cols = st.columns(n_cols)
                for i, param in enumerate(comp_edit['params']):
                    lo_def, hi_def = DEFAULT_PARAM_RANGES.get(param, (0., 100.))
                    init_def       = DEFAULT_PARAM_INIT.get(param, (lo_def + hi_def) / 2)
                    cur_init       = comp_edit['initial_values'].get(param, init_def)
                    cur_lo, cur_hi = comp_edit['param_ranges'].get(param, (lo_def, hi_def))
                    cur_free       = param in comp_edit['free_params']
                    with param_cols[i % n_cols]:
                        st.markdown(f"**{param}**")
                        st.number_input("init", value=float(cur_init), key=f"{comp_edit['name']}_{param}_init", format="%.4g") #min_value=float(lo_def), max_value=float(hi_def),
                        st.number_input("min",  value=float(cur_lo),   key=f"{comp_edit['name']}_{param}_min",  format="%.4g") #min_value=float(lo_def), max_value=float(hi_def),
                        st.number_input("max",  value=float(cur_hi),   key=f"{comp_edit['name']}_{param}_max",  format="%.4g")#min_value=float(lo_def), max_value=float(hi_def),
                                        
                        st.checkbox("libre", value=cur_free,
                                    key=f"{comp_edit['name']}_{param}_free")
                read_widget_values_into_comp(comp_edit)

                st.write("**Enregistrer le modèle**")
                if st.button("✅ Enregistrer", use_container_width=False, type="primary"):
                    for c in st.session_state.components:
                        read_widget_values_into_comp(c)
                    mname = model_name.strip() or "modèle_sans_nom"
                    st.session_state.MODEL[mname] = {
                        'components': [
                            {'type': c['type'], 'name': c['name'],
                            'initial_values': c['initial_values'].copy(),
                            'param_ranges':   c['param_ranges'].copy(),
                            'free_params':    c['free_params'].copy()}
                            for c in st.session_state.components
                        ]
                    }
                    st.success(f"✅ Modèle « {mname} » enregistré !")




        with tab2 : 
            # ── Import CSV ─────────────────────────────────────────────────
            st.markdown("##### 📂 Importer un modèle depuis un fichier CSV")

            
            csv_file = st.file_uploader(
                "Charger un CSV de paramètres (format tableau de résultats)",
                type=["csv"],
                key="csv_model_uploader",
                help="Colonnes attendues : Paramètre, Valeur, Min, Max, Libre\n"
                    "Paramètre au format : c{n}_{TypeAbbr}_{param}  ex: c1_UD_d"
            )
            csv_model_name = st.text_input(
                "Nom du modèle importé",
                placeholder="ex: modele_csv",
                key="csv_model_name",
            )
            st.markdown("&nbsp;", unsafe_allow_html=True)
            do_import = st.button("📥 Importer & stocker", use_container_width=False,
                                key="btn_csv_import")

            if csv_file is not None:
                try:
                    csv_df = pd.read_csv(csv_file)
                    # Aperçu
                    with st.expander("Aperçu du CSV chargé", expanded=False):
                        st.dataframe(csv_df, use_container_width=True)

                    if do_import:
                        result, err_msg = parse_csv_to_model(csv_df)
                        if result is None:
                            st.error(f"❌ Erreur d'import CSV :\n\n{err_msg}")
                        else:
                            target_name = csv_model_name.strip() or csv_file.name.replace(".csv", "")
                            st.session_state.MODEL[target_name] = result
                            # Synchroniser également st.session_state.components pour
                            # permettre une édition immédiate dans le panneau courant
                            st.session_state.components      = [
                                dict(c) for c in result['components']
                            ]
                            st.session_state.active_comp_name = (
                                result['components'][0]['name'] if result['components'] else None
                            )
                            n_comp = len(result['components'])
                            comp_names = ', '.join(c['name'] for c in result['components'])
                            st.success(
                                f"✅ Modèle **{target_name}** importé avec succès "
                                f"({n_comp} composant{'s' if n_comp > 1 else ''} : {comp_names})"
                            )
                            st.rerun()
                except Exception as exc:
                    st.error(f"Impossible de lire le CSV : {exc}")
        with tab3:
            st.markdown("##### Configurer les interpolateurs oimodeler")
            st.caption(
                "Permet d'assigner un interpolateur `oimInterp` à un paramètre d'un composant "
                "d'un modèle existant. Deux types : **Corps noir** (`starWl`) ou "
                "**Spline custom** (valeurs par longueur d'onde)."
            )

            if not st.session_state.MODEL:
                st.info("Aucun modèle disponible. Créez ou importez un modèle d'abord.")
            else:
                # ── Sélection du modèle ───────────────────────────────────
                interp_model_name = st.selectbox(
                    "Modèle cible",
                    sorted(list(st.session_state.MODEL.keys())),
                    key="interp_model_sel",
                )
                interp_model_data = st.session_state.MODEL[interp_model_name]
                interp_comps      = interp_model_data.get("components", [])

                if not interp_comps:
                    st.warning("Ce modèle ne contient aucun composant.")
                else:
                    # ── Sélection du composant ────────────────────────────
                    comp_names_interp = [c["name"] for c in interp_comps]
                    interp_comp_name  = st.selectbox(
                        "Composant",
                        comp_names_interp,
                        key="interp_comp_sel",
                    )
                    interp_comp = next(
                        c for c in interp_comps if c["name"] == interp_comp_name
                    )

                    # Paramètres disponibles — reconstruits depuis le registre si absent
                    _comp_type = interp_comp.get("type", "")
                    _params_from_registry = COMPONENT_REGISTRY.get(_comp_type, {}).get("params", [])
                    _params_from_comp     = interp_comp.get("params", _params_from_registry)
                    interp_params_avail = [
                        p for p in _params_from_comp if p not in ("x", "y")
                    ]

                    # ── Sélection du paramètre ────────────────────────────
                    interp_param = st.selectbox(
                        "Paramètre à interpoler",
                        interp_params_avail,
                        key="interp_param_sel",
                    )

                    # État courant de l'interpolateur pour ce param
                    cur_interp = interp_comp.get("interpolators", {}).get(interp_param, {})

                    # ── Type d'interpolateur ──────────────────────────────
                    interp_type = st.radio(
                        "Type d'interpolateur",
                        ["Corps noir (starWl)", "Spline custom (wl → valeurs)"],
                        index=0 if cur_interp.get("type") != "custom" else 1,
                        horizontal=True,
                        key="interp_type_radio",
                    )

                    st.markdown("---")

                    # ════════════════════════════════════════════════════
                    # Corps noir
                    # ════════════════════════════════════════════════════
                    if interp_type == "Corps noir (starWl)":
                        st.markdown("**Paramètres du corps noir**")
                        bb1, bb2, bb3 = st.columns(3)
                        with bb1:
                            bb_temp = st.number_input(
                                "Température (K)", value=float(cur_interp.get("temp", 5000.)),
                                min_value=100., max_value=100000., step=100.,
                                key="interp_bb_temp",
                            )
                        with bb2:
                            bb_dist = st.number_input(
                                "Distance (pc)", value=float(cur_interp.get("dist", 140.)),
                                min_value=1., max_value=1e6, step=10.,
                                key="interp_bb_dist",
                            )
                        with bb3:
                            bb_lum = st.number_input(
                                "Luminosité (L☉)", value=float(cur_interp.get("lum", 1.)),
                                min_value=0.001, max_value=1e6, step=0.1,
                                key="interp_bb_lum",
                            )

                        # Prévisualisation de la courbe
                        try:
                            wl_preview = np.linspace(1e-6, 5e-6, 200)
                            interp_obj = oim.oimInterp(
                                'starWl', temp=bb_temp, dist=bb_dist,
                                lum=bb_lum, wl=wl_preview,
                            )
                            flux_preview = np.array([interp_obj(w) for w in wl_preview])
                            fig_bb, ax_bb = plt.subplots(figsize=(6, 2.5))
                            ax_bb.plot(wl_preview * 1e6, flux_preview, color='orange', lw=2)
                            ax_bb.set_xlabel("λ (µm)"); ax_bb.set_ylabel("Flux (Jy)")
                            ax_bb.set_title(f"Corps noir T={bb_temp:.0f} K"); ax_bb.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_bb, use_container_width=True)
                            plt.close(fig_bb)
                        except Exception as exc:
                            st.caption(f"Prévisualisation indisponible : {exc}")

                        if st.button("✅ Appliquer l'interpolateur corps noir",
                                     key="btn_apply_bb", use_container_width=True):
                            if "interpolators" not in interp_comp:
                                interp_comp["interpolators"] = {}
                            interp_comp["interpolators"][interp_param] = {
                                "enabled": True,
                                "type":    "blackbody",
                                "temp":    bb_temp,
                                "dist":    bb_dist,
                                "lum":     bb_lum,
                            }
                            st.success(
                                f"✅ Interpolateur corps noir appliqué à "
                                f"**{interp_comp_name}.{interp_param}** "
                                f"(T={bb_temp:.0f} K, d={bb_dist:.0f} pc, L={bb_lum:.2f} L☉)"
                            )

                    # ════════════════════════════════════════════════════
                    # Spline custom
                    # ════════════════════════════════════════════════════
                    else:
                        st.markdown("**Points de contrôle (longueur d'onde → valeur)**")
                        st.caption(
                            "Entrez les longueurs d'onde (µm) et valeurs correspondantes. "
                            "oimodeler interpolera entre ces points."
                        )

                        # Récupérer les points existants ou initialiser
                        existing_wl  = cur_interp.get("wl",     [2e-6, 3e-6, 4e-6, 5e-6])
                        existing_val = cur_interp.get("values", [0.5,  0.8,  0.6,  0.3])
                        existing_wl_um  = [w * 1e6 for w in existing_wl]

                        n_pts = st.number_input(
                            "Nombre de points", min_value=2, max_value=20,
                            value=len(existing_wl_um), step=1, key="interp_n_pts",
                        )

                        # Tableau d'entrée des points
                        wl_pts  = []
                        val_pts = []
                        pt_cols = st.columns(min(int(n_pts), 5))
                        for i in range(int(n_pts)):
                            col_i = pt_cols[i % len(pt_cols)]
                            with col_i:
                                st.markdown(f"**pt {i+1}**")
                                wl_i = st.number_input(
                                    "λ (µm)", value=float(existing_wl_um[i])
                                    if i < len(existing_wl_um) else float(1 + i),
                                    min_value=0.1, max_value=20., step=0.1, format="%.2f",
                                    key=f"interp_wl_{i}",
                                )
                                v_i = st.number_input(
                                    "valeur", value=float(existing_val[i])
                                    if i < len(existing_val) else 0.5,
                                    format="%.4g",
                                    key=f"interp_val_{i}",
                                )
                                wl_pts.append(wl_i * 1e-6)
                                val_pts.append(v_i)

                        # Variable oimodeler à interpoler
                        interp_var = st.selectbox(
                            "Variable d'interpolation",
                            ["wl", "time", "mjd"],
                            index=0,
                            key="interp_var_sel",
                            help="En général 'wl' pour une dépendance spectrale."
                        )

                        # Prévisualisation spline
                        try:
                            interp_obj = oim.oimInterp(
                                interp_var,
                                **{interp_var: np.array(wl_pts)},
                                values=np.array(val_pts),
                            )
                            wl_fine = np.linspace(min(wl_pts), max(wl_pts), 200)
                            val_fine = np.array([interp_obj(w) for w in wl_fine])
                            fig_sp, ax_sp = plt.subplots(figsize=(6, 2.5))
                            ax_sp.plot(wl_fine * 1e6, val_fine, color='steelblue', lw=2,
                                       label='Spline')
                            ax_sp.scatter(
                                [w * 1e6 for w in wl_pts], val_pts,
                                color='red', zorder=5, label='Points de contrôle'
                            )
                            ax_sp.set_xlabel("λ (µm)"); ax_sp.set_ylabel(interp_param)
                            ax_sp.set_title(f"Interpolation {interp_param}")
                            ax_sp.legend(fontsize=8); ax_sp.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_sp, use_container_width=True)
                            plt.close(fig_sp)
                        except Exception as exc:
                            st.caption(f"Prévisualisation indisponible : {exc}")

                        if st.button("✅ Appliquer l'interpolateur custom",
                                     key="btn_apply_custom", use_container_width=True):
                            if "interpolators" not in interp_comp:
                                interp_comp["interpolators"] = {}
                            interp_comp["interpolators"][interp_param] = {
                                "enabled": True,
                                "type":    "custom",
                                "var":     interp_var,
                                "wl":      wl_pts,
                                "values":  val_pts,
                            }
                            st.success(
                                f"✅ Interpolateur custom appliqué à "
                                f"**{interp_comp_name}.{interp_param}** "
                                f"({int(n_pts)} points, var={interp_var})"
                            )

                    st.markdown("---")

                    # ── Résumé des interpolateurs actifs ──────────────────
                    st.markdown("##### Interpolateurs actifs sur ce composant")
                    interps = interp_comp.get("interpolators", {})
                    if not interps:
                        st.caption("Aucun interpolateur configuré.")
                    else:
                        for p_name, cfg in interps.items():
                            if cfg.get("enabled"):
                                if cfg["type"] == "blackbody":
                                    st.success(
                                        f"🌟 **{p_name}** → Corps noir "
                                        f"T={cfg['temp']:.0f} K, "
                                        f"d={cfg['dist']:.0f} pc, "
                                        f"L={cfg['lum']:.2f} L☉"
                                    )
                                else:
                                    st.info(
                                        f"📈 **{p_name}** → Spline custom "
                                        f"({len(cfg['wl'])} points, var={cfg['var']})"
                                    )
                                # Bouton de suppression
                                if st.button(f"🗑️ Supprimer interpolateur {p_name}",
                                             key=f"del_interp_{p_name}"):
                                    del interp_comp["interpolators"][p_name]
                                    st.rerun()

                    # ── Enregistrer les modifications dans MODEL ──────────
                    st.markdown("---")
                    new_interp_name = st.text_input(
                        "Enregistrer sous le nom",
                        value=f"{interp_model_name}_interp",
                        key="interp_save_name",
                    )
                    if st.button("💾 Enregistrer le modèle avec interpolateurs",
                                 key="btn_save_interp", type="primary",
                                 use_container_width=True):
                        import copy as _copy
                        saved = _copy.deepcopy(interp_model_data)
                        # Synchroniser le composant modifié
                        for i, c in enumerate(saved["components"]):
                            if c["name"] == interp_comp_name:
                                saved["components"][i]["interpolators"] = \
                                    interp_comp.get("interpolators", {})
                        target = new_interp_name.strip() or f"{interp_model_name}_interp"
                        st.session_state.MODEL[target] = saved
                        st.success(f"✅ Modèle **{target}** enregistré avec les interpolateurs !")

        with tab4 :
            try : 
                col1, col2 = st.columns([1, 2])
                with col1 : 
                    TEST = st.selectbox("Voir un model", sorted([mname for mname in  st.session_state.MODEL.keys()]), index=0)
                    _model = build_oim_model(st.session_state.MODEL[TEST]["components"])
                    sim = oim.oimSimulator(data=st.session_state.data, model=_model)
                    st.write("$\chi²$ : " + f"{sim.chi2r:.2f}")
                    st.write(sim.model)
    
    
                    col3, col4, col5 = st.columns(3)
                    with col3 : 
                        st.write("$X$ axis")
                        X_min = st.number_input("Xmin", key="Xmin CP", value=1.)
                        X_max = st.number_input("Xmax", key="Xmax CP", value=5.)
                        
                    with col4 : 
                        st.write("$V²_Y$")
                        Vis_Y_min = st.number_input("Ymin", key="Ymin Vis", value=0.)
                        Vis_Y_max = st.number_input("Ymax", key="Ymax Vis", value=1.)
                    with col5 : 
                        st.write("$CP_Y$")
                        CP_Y_min = st.number_input("Ymin", key="Ymin CP", value=-180.)
                        CP_Y_max = st.number_input("Ymax", key="Ymax CP", value=180.)
    
    
                with col2 : 
                    fig0, ax0 = sim.plot(["VIS2DATA", "T3PHI"])
    
                    ax0[0].set_xlim([X_min*1e7, X_max*1e7])
                    ax0[0].set_ylim([Vis_Y_min, Vis_Y_max])
    
                    ax0[1].set_xlim([X_min*1e7, X_max*1e7])
                    ax0[1].set_ylim([CP_Y_min, CP_Y_max])
    
                    st.pyplot(fig0)
    
                fig1 = sim.plotWlTemplate([["VIS2DATA"],["T3PHI"]],xunit="micron",figsize=(22,3))
                fig1.set_legends(0.5,0.8,"$BASELINE$",["VIS2DATA","T3PHI"],fontsize=10,ha="center")
                ax = fig1.axes[0]
                ax.set_ylim(Vis_Y_min, Vis_Y_max)
                
                ax2 = fig1.axes[7]
                ax2.set_ylim(CP_Y_min,CP_Y_max)
                st.pyplot(fig1)
            except : 
                st.warning("veuillez definir un modele avant d'acceder à cette fenêtre")
            with tab5 : 
                liste_de_model = sorted([mname for mname in  st.session_state.MODEL.keys()])

                if [i for i in st.session_state.MODEL] : 
                    col1, col2 = st.columns(2)

                    with col1 : 
                        st.write("**Renomer un modèle**")
                        model_TBR = st.selectbox("Selectionner le modele à renomer", liste_de_model, index=1, key="model TBR")
                        
                        new_name = st.text_input("Nouveau nom", value="test K2000", placeholder="new name")
                        
                        if st.button("Renomer", type="primary"):
                            st.session_state.MODEL[new_name] = copy.deepcopy(st.session_state.MODEL.pop(model_TBR))
                            list_model_dispo = [i for i in st.session_state.MODEL]

                            if (model_TBR not in list_model_dispo) & (new_name in list_model_dispo):
                                st.success(f"Le modèle : {model_TBR} à bien été renomé par {new_name}")

                    with col2 : 
                        st.write("**Supprimer un modèle**")
                        model_TBS = st.selectbox("Selectionner le modele à supprimer", liste_de_model, index=1, key="model TBS")
                        if st.button("Supprimer", key="Supprimer un model", type="primary"):
                            st.session_state.MODEL.pop(model_TBS)
                            
                            list_model_dispo = [i for i in st.session_state.MODEL]
                            if model_TBS not in list_model_dispo : 
                                st.success(f"Le modèle : {model_TBS} à bien été supprimé")
                else :
                    st.warning("Il n'y a pas de modele définit pour le moment") 







        
    # ── V. Fitting ──────────────────────────────────────────────────────
    st.header("V. Fitting des données")

    if not st.session_state.MODEL:
        st.warning("⚠️ Aucun modèle enregistré. Configurez et enregistrez un modèle (section IV).")
        st.stop()

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        st.selectbox("Jeu de données", options=list(st.session_state.loaded_files.keys()),
                     key="fit_dataset")
    with col_sel2:
        model_options = sorted(list(st.session_state.MODEL.keys()))
        model_to_use  = st.selectbox("Modèle à utiliser", options=model_options,
                                     index=len(model_options) - 1, key="fit_model")
    with col_sel3:
        methode = st.selectbox("Méthode",
                               ["Aléatoire", "Minimisation χ²", "Emcee"],
                               key="fit_method")

    # ── V-a. Recherche aléatoire ────────────────────────────────────────
    if methode == "Aléatoire":
        st.markdown("##### Configuration de la recherche aléatoire")
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            n_runs    = st.number_input("Nombre d'itérations", 10, 1000, 100, 10)
            use_seed  = st.checkbox("Graine fixe", value=True)
            seed_val  = st.number_input("Graine", 0, 99999, 42) if use_seed else None
        with ca2:
            rand_dtypes = st.multiselect("Données à utiliser",
                                         ["VIS2DATA", "T3PHI", "VISPHI", "T3AMP", "FLUXDATA"],
                                         default=["VIS2DATA", "T3PHI"])
        with ca3:
            use_custom_wave = st.checkbox("Longueurs d'onde personnalisées")
            wave_override   = None
            if use_custom_wave:
                wco1, wco2, wco3 = st.columns(3)
                with wco1: wl_cmin = st.number_input("λ min (m)", value=1e-6, format="%.2e", key="wco_min")
                with wco2: wl_cmax = st.number_input("λ max (m)", value=5e-6, format="%.2e", key="wco_max")
                with wco3: n_wl    = st.number_input("Nb points", 10, 200, 50, key="wco_n")
                wave_override = np.linspace(wl_cmin, wl_cmax, n_wl)
                st.success(f"✓ {n_wl} points entre {wl_cmin:.2e} et {wl_cmax:.2e} m")

        model_comps = st.session_state.MODEL[model_to_use]["components"]
        if not model_comps:
            st.warning("Le modèle sélectionné est vide.")
        elif st.session_state.data is None:
            st.warning("Chargez des données OIFITS.")
        else:
            with st.expander(f"Résumé du modèle « {model_to_use} »"):
                for c in model_comps:
                    st.write(f"**{c['name']}** ({c['type']}) — libres : {', '.join(c['free_params'])}")

            if st.button("🚀 Lancer la recherche aléatoire", type="primary",
                         use_container_width=True):
                configs = [
                    ComponentConfig(component_type=c['type'], name=c['name'],
                                    initial_values=c['initial_values'],
                                    param_ranges=c['param_ranges'],
                                    free_params=c['free_params'],
                                    interpolators=c.get('interpolators', {}))
                    for c in model_comps
                ]
                try:
                    st.session_state.data.useFilter = True
                    bp, bc, hist = random_search(st.session_state.data, configs,
                                                 n_runs=n_runs, seed=seed_val,
                                                 wave_data=wave_override)
                    # Reconstruire le meilleur modèle
                    best_comps = [cfg.create_instance(bp.get(cfg.name, {}))
                                  for cfg in configs]
                    st.session_state.best_model      = oim.oimModel(*best_comps)
                    st.session_state.best_params     = bp
                    st.session_state.best_chi2       = bc
                    st.session_state.history         = hist
                    st.session_state.optimization_done = True
                    st.success("✅ Optimisation terminée !")
                    st.balloons()
                except Exception as exc:
                    st.error(f"Erreur : {exc}")

            if st.session_state.optimization_done:
                st.markdown("### Résultats de la recherche aléatoire")
                st.success(f"Meilleur χ²ᵣ : **{st.session_state.best_chi2:.4f}**")
                _, tbl = get_result_df(st.session_state.best_model, is_fit=False)
                st.dataframe(tbl, use_container_width=True)

                if st.button("💾 Enregistrer ce meilleur modèle", use_container_width=True,
                             key="save_random"):
                    update_model_from_fit(
                        f"Best_Random_{model_to_use}",
                        model_to_use,
                        st.session_state.best_model,
                        chi2r=st.session_state.best_chi2,   # ✅
                    )
                    st.success(f"Modèle **Best_Random_{model_to_use}** enregistré !")

                # Historique graphique
                st.markdown("##### Historique")
                runs   = [r['run']   for r in st.session_state.history]
                chi2s  = [r['chi2r'] for r in st.session_state.history]
                cummin = []
                cur    = float('inf')
                for v in chi2s:
                    cur = min(cur, v)
                    cummin.append(cur)

                gh1, gh2 = st.columns(2)
                with gh1:
                    fig1, ax1 = plt.subplots(figsize=(7, 5))
                    ax1.scatter(runs, chi2s, alpha=0.4, s=15, label='Tous les essais')
                    ax1.plot(runs, cummin, 'r-', lw=2, label='Meilleur χ²ᵣ')
                    ax1.set_xlabel('Run'); ax1.set_ylabel('χ²ᵣ')
                    ax1.set_title('Évolution'); ax1.legend(); ax1.grid(alpha=.3)
                    st.pyplot(fig1); plt.close(fig1)
                with gh2:
                    fig2, ax2 = plt.subplots(figsize=(7, 5))
                    ax2.hist(chi2s, bins=30, alpha=.7, edgecolor='black')
                    ax2.axvline(min(chi2s), color='r', ls='--', lw=2,
                                label=f'Min = {min(chi2s):.4f}')
                    ax2.set_xlabel('χ²ᵣ'); ax2.set_ylabel('Fréquence')
                    ax2.set_title('Distribution'); ax2.legend(); ax2.grid(alpha=.3)
                    st.pyplot(fig2); plt.close(fig2)

    # ── V-b. Minimisation χ² ────────────────────────────────────────────
    elif methode == "Minimisation χ²":
        st.markdown("### Minimisation du χ²")
        opt_dtypes = st.multiselect("Données à fitter",
                                    ["VIS2DATA", "T3PHI", "FLUXDATA"],
                                    default=["VIS2DATA", "T3PHI"],
                                    key="chi2_dtypes")

        if st.session_state.data is None:
            st.warning("Chargez des données OIFITS.")
        else:
            model_chi2 = build_oim_model(st.session_state.MODEL[model_to_use]["components"])
            if model_chi2 is None:
                st.error("Impossible de construire le modèle.")
            elif st.button("▶️ Lancer la minimisation χ²", type="primary"):
                st.session_state.data.useFilter = True
                try:
                    model_init = copy.deepcopy(model_chi2)
                    sim_init   = oim.oimSimulator(data=st.session_state.data, model=model_init)
                    sim_init.compute(computeChi2=True, computeSimulatedData=False)
                    chi2_init  = sim_init.chi2r

                    lmfit = oim.oimFitterMinimize(st.session_state.data, model_chi2,
                                                   dataTypes=opt_dtypes)
                    lmfit.prepare()
                    lmfit.run()
                    st.balloons()

                    st.session_state.chi2_result = {
                        'model_initial':    model_init,
                        'best_chi2_model':  lmfit.simulator.model,
                        'chi2_init':        chi2_init,
                        'chi2_final':       lmfit.simulator.chi2r,
                        'lmfit':            lmfit,
                        'model_to_use':     model_to_use,
                        'state':            0,
                    }
                except Exception as exc:
                    st.error(f"Erreur minimisation : {exc}")

            if st.session_state.chi2_result is not None:
                r = st.session_state.chi2_result
                if r['chi2_final'] > r['chi2_init']:
                    st.warning(f"⚠️ Divergence : {r['chi2_init']:.2f} → {r['chi2_final']:.2f}")
                else:
                    st.success(f"✅ χ²ᵣ : {r['chi2_init']:.2f} → {r['chi2_final']:.2f}")

                cr1, cr2 = st.columns(2)
                with cr1:
                    st.markdown(f"**Avant** (χ²ᵣ = {r['chi2_init']:.2f})")
                    _, tbl1 = get_result_df(r['model_initial'], is_fit=False)
                    st.dataframe(tbl1, use_container_width=True)
                with cr2:
                    st.markdown(f"**Après** (χ²ᵣ = {r['chi2_final']:.2f})")
                    _, tbl2 = get_result_df(r['best_chi2_model'], is_fit=False)
                    st.dataframe(tbl2, use_container_width=True)

                # Figure comparaison
                try:
                    st.session_state.data.useFilter = True
                    decomp = decompose_model_flux(r['best_chi2_model'], st.session_state.data)

                    ax_v2 = r['lmfit'].simulator.plotWithResiduals(
                        ["VIS2DATA"], xunit="cycle/mas",
                        kwargsData=dict(color="byBaseline"))[1]
                    ax_t3 = r['lmfit'].simulator.plotWithResiduals(
                        ["T3PHI"], xunit="cycle/mas",
                        kwargsData=dict(color="byBaseline"))[1]
                    d_img = extract_model_image(r['best_chi2_model'])

                    # Figure FLUX avec oimAxes (seul axe qui supporte oiplot)
                    fig_flux = plot_flux_decomposition(decomp, st.session_state.data)
                    ax_flux_src = fig_flux.axes[0]

                    plt.close('all')

                    # Figure finale 4 panneaux
                    fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5))

                    # Panneau 0 : recopie du graphe FLUX
                    copy_axes_lines(ax_flux_src, axes_cmp[0])
                    axes_cmp[0].set_title("FLUXDATA / composantes")
                    # Légende dédupliquée
                    handles, labels = axes_cmp[0].get_legend_handles_labels()
                    seen = {}
                    for h, l in zip(handles, labels):
                        if l not in seen:
                            seen[l] = h
                    axes_cmp[0].legend(seen.values(), seen.keys(), fontsize=7)

                    # Panneau 1 : VIS²
                    copy_axes_lines(ax_v2[0], axes_cmp[1])
                    axes_cmp[1].set_title("VIS²")

                    # Panneau 2 : T3PHI
                    copy_axes_lines(ax_t3[0], axes_cmp[2])
                    axes_cmp[2].set_title("T3PHI")

                    # Panneau 3 : Image modèle
                    axes_cmp[3].imshow(d_img[0, 0] ** 0.2, cmap='hot', origin='lower')
                    axes_cmp[3].set_title("Modèle (γ=0.2)")

                    plt.tight_layout()
                    st.pyplot(fig_cmp, use_container_width=True)
                    plt.close(fig_cmp)
                    plt.close(fig_flux)

                except Exception as exc:
                    st.warning(f"Impossible d'afficher la comparaison : {exc}") 


                # ── Code Python reproductible ─────────────────────────
                with st.expander("Code Python reproductible", expanded=False):
                    _filter_params = {
                        "expr":   st.session_state.get("_last_filter_expr", ""),
                        "bin_L":  st.session_state.get("bin_L", 1),
                        "bin_N":  st.session_state.get("bin_N", 1),
                        "norm_L": st.session_state.get("norm_L", False),
                        "norm_N": st.session_state.get("norm_N", False),
                    }
                    _code_chi2 = generate_fitting_code(
                        method        = "chi2",
                        result        = {"dtypes": opt_dtypes},
                        data_filename = st.session_state.get("fit_dataset", "data.fits"),
                        model_comps   = st.session_state.MODEL[r["model_to_use"]]["components"],
                        filter_params = _filter_params,
                    )
                    st.code(_code_chi2, language="python")
                               
                if st.button("💾 Enregistrer le meilleur modèle χ²", use_container_width=True,key="save_chi2", type="primary"):
                    update_model_from_fit(
                        f"Best_Chi2r_{r['model_to_use']}",
                        r['model_to_use'],
                        r['best_chi2_model'],
                        chi2r=r['chi2_final'],           
                    )
                    st.success(f"Modèle **Best_Chi2r_{r['model_to_use']}** enregistré !")
        # ── V-c. Emcee ──────────────────────────────────────────────────────
    elif methode == "Emcee":
        st.markdown("### Emcee MCMC")
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            emcee_dtypes = st.multiselect("Données à fitter",
                                          ["VIS2DATA", "T3PHI", "FLUXDATA"],
                                          default=["VIS2DATA", "T3PHI"],
                                          key="emcee_dtypes")
        with ec2:
            nb_walkers  = st.number_input("Walkers",  1, 64, 32,  key="emcee_walkers")
        with ec3:
            nb_steps    = st.number_input("Steps",    0, 40000, 1000, key="emcee_steps")
        with ec4:
            init_mode   = st.selectbox("Init", ['random', 'gaussian', ""], index=0,
                                       key="emcee_init")

        if st.session_state.data is None:
            st.warning("Chargez des données OIFITS.")
        else:
            model_emcee = build_oim_model(st.session_state.MODEL[model_to_use]["components"])
            if model_emcee is None:
                st.error("Impossible de construire le modèle.")
            elif st.button("▶️ Lancer Emcee", type="primary"):
                st.session_state.data.useFilter = True
                try:
                    model_init = copy.deepcopy(model_emcee)
                    sim_init   = oim.oimSimulator(data=st.session_state.data, model=model_init)
                    sim_init.compute(computeChi2=True, computeSimulatedData=False)
                    chi2_init  = sim_init.chi2r

                    emfit = oim.oimFitterEmcee(st.session_state.data, model_emcee,nwalkers=nb_walkers,dataTypes=emcee_dtypes)
                    sampler_path = Path("/tmp/sampler_emcee.txt")
                    sampler_path.unlink(missing_ok=True)   # ✅ supprime le fichier s'il existe

                    emfit.prepare(init=init_mode,samplerFile=str(sampler_path))
                    with st.spinner("MCMC en cours …", show_time=True):
                        emfit.run(nsteps=nb_steps, progress=True)

                    st.session_state.Emcee_Result = {
                        'model_initial':    model_init,
                        'best_Emcee_model': emfit.simulator.model,
                        'chi2_init':        chi2_init,
                        'chi2_final':       emfit.simulator.chi2r,
                        'lmfit':            emfit,
                        'model_to_use':     model_to_use,
                    }
                    st.success("✅ Emcee terminé !")
                    st.balloons()
                except ValueError as e:
                    st.error(f"ValueError : {e}")
                except Exception as e:
                    st.error(f"Erreur Emcee : {e}")

            if st.session_state.Emcee_Result is not None:
                er = st.session_state.Emcee_Result

                # ── LIGNE 1 : χ²ᵣ + tableau des paramètres ──────────────
                st.markdown(f"χ²ᵣ : **{er['chi2_init']:.2f}** → **{er['chi2_final']:.2f}**")
                st.markdown("##### Paramètres ajustés")
                _, tbl_em = get_result_df(er['best_Emcee_model'], is_fit=False)
                st.dataframe(tbl_em, use_container_width=True, height=350)

                if st.button("💾 Enregistrer le meilleur modèle Emcee",
                             use_container_width=True, key="save_emcee"):
                    update_model_from_fit(
                        f"Best_Emcee_{er['model_to_use']}",
                        er['model_to_use'],
                        er['best_Emcee_model'],
                        chi2r=er['chi2_final'],
                    )
                    st.success(f"Modèle **Best_Emcee_{er['model_to_use']}** enregistré !")

                # ── Code Python reproductible ─────────────────────────
                with st.expander("Code Python reproductible", expanded=False):
                    _filter_params = {
                        "expr":   st.session_state.get("_last_filter_expr", ""),
                        "bin_L":  st.session_state.get("bin_L", 1),
                        "bin_N":  st.session_state.get("bin_N", 1),
                        "norm_L": st.session_state.get("norm_L", False),
                        "norm_N": st.session_state.get("norm_N", False),
                    }
                    _code_emcee = generate_fitting_code(
                        method        = "emcee",
                        result        = {
                            "dtypes":   emcee_dtypes,
                            "nwalkers": nb_walkers,
                            "nsteps":   nb_steps,
                            "init":     init_mode,
                        },
                        data_filename = st.session_state.get("fit_dataset", "data.fits"),
                        model_comps   = st.session_state.MODEL[er["model_to_use"]]["components"],
                        filter_params = _filter_params,
                    )
                    st.code(_code_emcee, language="python")

                st.markdown("---")

                # ── LIGNE 2 : multiselect ────────────────────────────────
                st.markdown("#### Affichage des résultats")
                # PAR :
                GRAPH_OPTIONS = {
                    "Walkers":               "walkers",
                    "Corner plot":           "corner",
                    "VIS² / T3PHI":          "vis_t3",
                    "FLUXDATA / composantes": "flux",
                    "Image modèle":          "image",
                }
                selected_graphs = st.multiselect(
                    "Graphes à afficher",
                    list(GRAPH_OPTIONS.keys()),
                    default=["VIS² / T3PHI", "FLUXDATA / composantes"],
                    key="emcee_graphs",
                )
                selected_keys = [GRAPH_OPTIONS[g] for g in selected_graphs]

                # ── LIGNE 3 : col_params (gauche) | col_graph (droite) ───
                if selected_keys:
                    col_params, col_graph = st.columns([1, 2])

                    with col_params:
                        st.markdown("##### Paramètres des graphes")

                        # ── Walkers : pas de paramètre ───────────────────
                        if "walkers" in selected_keys:
                            st.markdown("**Walkers**")
                            st.caption("Aucun paramètre ajustable.")
                            st.markdown("---")

                        # ── Corner : pas de paramètre ────────────────────
                        if "corner" in selected_keys:
                            st.markdown("**Corner plot**")
                            st.caption("Aucun paramètre ajustable.")
                            st.markdown("---")

                        # ── VIS² / T3PHI ─────────────────────────────────
                        if "vis_t3" in selected_keys:
                            st.markdown("**VIS² / T3PHI**")
                            vt_xs    = st.selectbox("Échelle X", ["linear", "log"],
                                                    key="em_vt_xs")
                            vt_xmn   = st.number_input("Xmin (cycle/mas)", value=0.,
                                                        key="em_vt_xmn")
                            vt_xmx   = st.number_input("Xmax (cycle/mas)", value=5.,
                                                        key="em_vt_xmx")
                            vt_v2_ys = st.selectbox("Échelle Y  VIS²", ["linear", "log"],
                                                     key="em_vt_v2_ys")
                            vt_v2_ymn = st.number_input("Ymin VIS²", value=0.,
                                                         key="em_vt_v2_ymn")
                            vt_v2_ymx = st.number_input("Ymax VIS²", value=1.,
                                                         key="em_vt_v2_ymx")
                            vt_t3_ymn = st.number_input("Ymin T3PHI (°)", value=-15.,
                                                         key="em_vt_t3_ymn")
                            vt_t3_ymx = st.number_input("Ymax T3PHI (°)", value=15.,
                                                         key="em_vt_t3_ymx")
                            st.markdown("---")
                        # ── FLUX ─────────────────────────────────────────
                        if "flux" in selected_keys:
                            st.markdown("**FLUXDATA / composantes**")
                            st.caption("Aucun paramètre ajustable.")
                            st.markdown("---")

                        # ── Image ─────────────────────────────────────────
                        if "image" in selected_keys:
                            st.markdown("**Image modèle**")
                            img_gamma = st.slider("Gamma γ", 0.05, 1.0, 0.2, 0.05,
                                                  key="em_img_gamma")
                            img_cmap  = st.selectbox("Colormap",
                                                     ["hot", "inferno", "viridis",
                                                      "plasma", "gray", "afmhot"],
                                                     key="em_img_cmap")
                            img_size  = st.number_input("Taille image (px)", 64, 512, 128,
                                                         step=64, key="em_img_size")
                            img_scale = st.number_input("Échelle (mas/px)", 0.1, 10., 1.,
                                                         step=0.1, key="em_img_scale")
                            use_wl    = st.checkbox("Filtrer sur λ", value=False,
                                                    key="em_img_use_wl")
                            wl_val    = None
                            if use_wl:
                                wl_val = st.number_input("λ (µm)", value=3.5, step=0.1,
                                                          key="em_img_wl") * 1e-6
                            st.markdown("---")





                    with col_graph:
                        st.markdown("##### Graphes")

                        # ── Walkers ───────────────────────────────────────
                        if "walkers" in selected_keys:
                            st.markdown("**Walkers**")
                            try:
                                fw, _ = er['lmfit'].walkersPlot(chi2limfact=5)
                                st.pyplot(fw, use_container_width=True)
                                plt.close(fw)
                            except Exception as exc:
                                st.warning(f"Walkers : {exc}")

                        # ── Corner ────────────────────────────────────────
                        if "corner" in selected_keys:
                            st.markdown("**Corner plot**")
                            try:
                                fc, _ = er['lmfit'].cornerPlot(dchi2limfact=5)
                                st.pyplot(fc, use_container_width=True)
                                plt.close(fc)
                            except Exception as exc:
                                st.warning(f"Corner : {exc}")

                        # ── VIS² / T3PHI ──────────────────────────────────
                        if "vis_t3" in selected_keys:
                            st.markdown("**VIS² / T3PHI**")
                            try:
                                sim_plot = oim.oimSimulator(
                                    data=st.session_state.data,
                                    model=er['best_Emcee_model']
                                )
                                sim_plot.compute(computeChi2=False, computeSimulatedData=True)

                                fig_0, ax_0 = sim.plot(["VIS2DATA", "T3PHI"])
                                
                                ax_0[0].set_xscale(vt_xs)
                                ax_0[0].set_yscale(vt_v2_ys)                                
                                ax_0[0].set_xlim(vt_xmn*1e7, vt_xmx*1e7)
                                ax_0[0].set_ylim(vt_v2_ymn, vt_v2_ymx)
                                ax_0[0].set_title("VIS²")
                                ax_0[0].grid(True, alpha=0.3)

                                ax_0[1].set_xscale(vt_xs)
                                ax_0[1].set_yscale(vt_v2_ys)                              
                                ax_0[1].set_xlim(vt_xmn*1e7, vt_xmx*1e7)
                                ax_0[1].set_ylim(vt_t3_ymn, vt_t3_ymx)
                                ax_0[1].set_title("T3PHI")
                                ax_0[1].grid(True, alpha=0.3)


                                st.pyplot(fig_0)

                            except Exception as exc:
                                st.warning(f"VIS²/T3PHI : {exc}")

                        # ── FLUX ─────────────────────────────────────────
                        if "flux" in selected_keys:
                            st.markdown("**FLUXDATA / composantes**")
                            try:
                                st.session_state.data.useFilter = True
                                decomp_em = decompose_model_flux(
                                    er['best_Emcee_model'], st.session_state.data)
                                fig_flux_em = plot_flux_decomposition(
                                    decomp_em, st.session_state.data)
                                st.pyplot(fig_flux_em, use_container_width=True)
                                plt.close(fig_flux_em)
                            except Exception as exc:
                                st.warning(f"FLUXDATA : {exc}")
                        # ── Image ─────────────────────────────────────────
                        if "image" in selected_keys:
                            try:
                                img_data = extract_model_image(
                                    er['best_Emcee_model'],
                                    img_size=int(img_size),
                                    img_scale=float(img_scale),
                                    wl_value=wl_val,
                                )
                                display_img = img_data[0, 0] ** img_gamma
                                extent_half = img_size * img_scale / 2
                                fig_img, ax_img = plt.subplots(figsize=(5, 5))
                                im_plot = ax_img.imshow(
                                    display_img, cmap=img_cmap, origin='lower',
                                    extent=[-extent_half, extent_half,
                                            -extent_half, extent_half]
                                )
                                plt.colorbar(im_plot, ax=ax_img,
                                             label=f'Intensité (γ={img_gamma})')
                                ax_img.set_xlabel("ΔRA (mas)")
                                ax_img.set_ylabel("ΔDec (mas)")
                                wl_label = f" @ {wl_val*1e6:.2f} µm" if wl_val else ""
                                ax_img.set_title(f"Modèle{wl_label}")
                                st.pyplot(fig_img, use_container_width=True)
                                plt.close(fig_img)
                            except Exception as exc:
                                st.warning(f"Image : {exc}")

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray;'>"
        "OIModeler · Modélisation d'interférométrie optique · Basé sur <em>oimodeler</em>"
        "</div>",
        unsafe_allow_html=True,
    )



















