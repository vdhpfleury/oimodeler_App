# components/plots.py
"""
Composants UI réutilisables pour les graphes matplotlib.

Pattern systématique :
    fig = make_plot(...)
    st.pyplot(fig)
    plt.close(fig)   ← TOUJOURS fermer pour éviter les fuites mémoire

Les fonctions de ce module retournent des figures matplotlib.
La fermeture est à la charge de l'appelant (page) pour permettre
des usages composites (sous-figures).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config.constants import COMP_COLORS, COMP_STYLES


def plot_flux_decomposition(decomp: dict, data) -> plt.Figure:
    """Affiche FLUXDATA pour les observations, le modèle complet, et chaque composant."""
    fig = plt.figure(figsize=(9, 5))
    ax  = plt.subplot(projection='oimAxes')

    try:
        data.useFilter = True
        ax.oiplot(data, "EFF_WAVE", "FLUXDATA",
                  errorbar=True, xunit='micron',
                  kwargs_error={"alpha": 0.3},
                  color='grey', label='Data')
    except Exception:
        pass

    try:
        ax.oiplot(decomp['full']['sim'].simulatedData,
                  "EFF_WAVE", "FLUXDATA",
                  xunit='micron', color='red', lw=2, label='Full model')
    except Exception:
        pass

    for i, key in enumerate(k for k in decomp if k != 'full'):
        color = COMP_COLORS[i % len(COMP_COLORS)]
        ls    = COMP_STYLES[i % len(COMP_STYLES)]
        try:
            ax.oiplot(decomp[key]['sim'].simulatedData,
                      "EFF_WAVE", "FLUXDATA",
                      xunit='micron', color=color, ls=ls, label=key)
        except Exception:
            pass

    ax.set_ylabel('Flux density (Jy)', fontsize=10)
    ax.set_xlabel(r'Wavelength (µm)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def copy_axes_lines(src_ax, dst_ax) -> None:
    """Copie les lignes d'un axe matplotlib vers un autre."""
    for line in src_ax.get_lines():
        dst_ax.plot(
            line.get_xdata(), line.get_ydata(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            color=line.get_color(),
            label=line.get_label(),
        )
    dst_ax.set_xlabel(src_ax.get_xlabel())
    dst_ax.set_ylabel(src_ax.get_ylabel())
    dst_ax.grid(True)


def safe_pyplot(st_module, fig: plt.Figure, **kwargs) -> None:
    """
    Affiche une figure Streamlit et la ferme immédiatement.
    Wrapper pratique pour garantir plt.close() systématique.
    """
    try:
        st_module.pyplot(fig, **kwargs)
    finally:
        plt.close(fig)
