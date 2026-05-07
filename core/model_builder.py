# core/model_builder.py
"""
Construction des modèles oimodeler à partir de listes de dicts de composants.
Logique pure – aucune dépendance Streamlit.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core.component import ComponentConfig


def build_oim_model(oim, registry: dict, comp_list: list):
    """
    Instancie un oimModel à partir d'une liste de dicts de composants.

    Retourne None si la liste est vide ou si une erreur survient.
    Les erreurs sont propagées à l'appelant (UI) pour affichage adapté.
    """
    if not comp_list:
        return None

    instances = [
        ComponentConfig(
            component_type=c['type'],
            registry=registry,
            name=c['name'],
            initial_values=c['initial_values'],
            param_ranges=c['param_ranges'],
            free_params=c['free_params'],
            interpolators=c.get('interpolators', {}),
        ).create_instance(oim)
        for c in comp_list
    ]
    return oim.oimModel(*instances)


def generate_model_image_preview(oim, registry: dict, comp_list: list, fov:int=128, px_size:float=0.15, gamma:float=0.2, wl:float=3.5e-6) -> plt.Figure | None:
    """
    Génère une figure matplotlib d'aperçu du modèle (image FT).
    Retourne None si le modèle ne peut pas être construit.
    """
    model = build_oim_model(oim, registry, comp_list)
    if model is None:
        return None

    
    tot_size = int(fov * 0.5 * px_size)
    im = model.getImage(fov, px_size, wl=wl, fromFT=True)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(im ** gamma, cmap='hot', origin='lower',
              extent=[-tot_size, tot_size, -tot_size, tot_size])
    ax.set_xlabel('X mas', fontsize=6)
    ax.set_ylabel('Y mas', fontsize=6)
    ax.tick_params(axis='both', labelsize=6)
    return fig


def generate_model_v2_t3phi_preview(oim, registry: dict, comp_list: list, data, V2_Y_min: float=0., V2_Y_max: float=1., T3PHI_Y_min: float=-180., T3PHI_Y_max: float=180.,  ) -> plt.Figure | None:
    """Génère un aperçu VIS²/T3PHI du modèle sur les données observées."""
    model = build_oim_model(oim, registry, comp_list)
    if model is None:
        return None

    sim = oim.oimSimulator(data=data, model=model)
    fig, _ = sim.plot(["VIS2DATA", "T3PHI"])
    _[0].set_ylim(V2_Y_min, V2_Y_max)
    #_[0].set_xlim(0, 5e7)
    _[1].set_ylim(T3PHI_Y_min, T3PHI_Y_max)
    return fig


def extract_model_image(oim, model, img_size: int = 256,
                        img_scale: float = 0.05,
                        wl_value: float | None = 3.5e-6) -> np.ndarray:
    """
    Extrait et somme les images de chaque composant individuellement.
    Retourne un ndarray de shape (1, 1, img_size, img_size).
    """
    data_sum = None

    for comp in model.components:
        try:
            m_single = oim.oimModel(comp)
            kwargs = {'fromFT': True}
            if wl_value is not None:
                kwargs['wl'] = [wl_value]

            fig_c, _, data_c = m_single.showModel(img_size, img_scale, **kwargs)
            plt.close(fig_c)

            data_sum = data_c.copy() if data_sum is None else data_sum + data_c
        except Exception:
            pass

    if data_sum is None:
        data_sum = np.zeros((1, 1, img_size, img_size))

    return data_sum


def decompose_model_flux(oim, model, data) -> dict:
    """
    Pour chaque composant, crée un oimModel individuel et calcule les données simulées.

    Retourne un dict :
        {
            'full':  {'model': oimModel, 'sim': oimSimulator},
            'c1_UD': {'model': oimModel, 'sim': oimSimulator},
            ...
        }
    """
    result = {}

    sim_full = oim.oimSimulator(data=data, model=model)
    sim_full.compute(computeChi2=False, computeSimulatedData=True)
    result['full'] = {'model': model, 'sim': sim_full}

    for comp in model.components:
        comp_name  = comp.name if hasattr(comp, 'name') else type(comp).__name__
        m_single   = oim.oimModel(comp)
        sim_single = oim.oimSimulator(data=data, model=m_single)
        sim_single.compute(computeChi2=False, computeSimulatedData=True)
        result[comp_name] = {'model': m_single, 'sim': sim_single}

    return result
