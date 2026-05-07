# pages/explorer.py
"""
Page "Component Explorer" – Exploration interactive des composants oimodeler.

Dépendances :
- services/data_service.py  → get_oim(), get_registry()
- components/plots.py       → safe_pyplot()
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from services.data_service import get_oim, get_registry
from components.plots import safe_pyplot


# ── Configuration des sliders par paramètre ───────────────────────────────
_SLIDER_CFG: dict[str, tuple] = {
    'x':     ("x (X position mas)",    -50., 50.,  0.,   1.),
    'y':     ("y (Y position mas)",    -50., 50.,  0.,   1.),
    'f':     ("f (flux)",               0.,  2.,   1.,   0.1),
    'd':     ("d (diameter mas)",       0.,  100., 40.,  1.),
    'din':   ("din (inner diam. mas)",  0.,  100., 30.,  1.),
    'dout':  ("dout (outer diam. mas)", 0.,  100., 50.,  1.),
    'w':     ("w (width mas)",          0.,  50.,  20.,  1.),
    'dx':    ("dx (X width mas)",       0.,  100., 30.,  1.),
    'dy':    ("dy (Y height mas)",      0.,  100., 20.,  1.),
    'elong': ("elong (elongation)",     1.,  3.,   1.5,  0.1),
    'pa':    ("pa (angle °)",           0.,  180., 45.,  5.),
    'fwhm':  ("fwhm (mas)",             1.,  30.,  10.,  1.),
    'hlr':   ("hlr (mas)",              1.,  30.,  10.,  1.),
    'flor':  ("flor",                   0.,  1.,   0.5,  0.05),
    'skw':   ("skw (skewness)",         0.,  1.,   0.3,  0.05),
    'skwPa': ("skwPa (skew angle °)",   0.,  180., 30.,  5.),
    'a':     ("a",                      0.,  1.,   0.5,  0.05),
    'a1':    ("a1",                     0.,  1.,   0.3,  0.05),
    'a2':    ("a2",                     0.,  1.,   0.2,  0.05),
}


def render() -> None:
    registry = get_registry()
    oim      = get_oim()

    # Exclut les composants avancés (trop de paramètres pour l'explorateur)
    visu_components = {
        k: v['params'] for k, v in registry.items()
        if k not in ('oimStarHaloGaussLorentz', 'oimStarHaloIRing')
    }

    col_left, col_right = st.columns(2)

    with col_left:
        selected_comp = st.selectbox(
            "Choose a component",
            list(visu_components.keys()),
            format_func=lambda x: f"{x}  —  {registry[x]['description']}",
        )

        required    = visu_components[selected_comp]
        visu_params: dict = {}

        cols3 = st.columns(3)
        for i, param in enumerate(required):
            with cols3[i % 3]:
                cfg = _SLIDER_CFG.get(param)
                if cfg:
                    label, mn, mx, dfl, step = cfg
                    visu_params[param] = st.slider(
                        label, mn, mx, dfl, step, key=f"visu_{param}"
                    )
                else:
                    visu_params[param] = st.number_input(
                        param, value=0., key=f"visu_{param}"
                    )

        # ── Code Python généré ────────────────────────────────────────
        st.subheader("Associated Python code")
        params_str = ",\n    ".join(f"{k}={v}" for k, v in visu_params.items())
        code = (
            f"import oimodeler as oim\n"
            f"import matplotlib.pyplot as plt\n\n\n"
            f"component = oim.{selected_comp}(\n    {params_str}\n)\n"
            f"model = oim.oimModel(component)\n"
            f"im = model.getImage(256, 1, fromFT=True)\n\n"
            f"plt.figure()\nplt.imshow(im**0.2, cmap='hot')\nplt.show()"
        )
        st.code(code, language='python')

    with col_right:
        try:
            comp_cls  = registry[selected_comp]['class']
            comp_inst = comp_cls(**visu_params)
            mdl       = oim.oimModel(comp_inst)
            im        = mdl.getImage(256, 1, fromFT=True)

            fig, ax = plt.subplots(figsize=(6, 6))
            im_disp = ax.imshow(im ** 0.2, cmap='hot', origin='lower')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title(f'{selected_comp}  –  γ = 0.2')
            plt.colorbar(im_disp, ax=ax, label='Intensity (γ corrected)')
            safe_pyplot(st, fig)


            
            #fig_uv, ax_uv = plt.subplots(figsize=(6, 6))
            #u = np.linspace(-100, 100, 200)
            #v = np.zeros_like(u)
            #vis = mdl.getComplexCoherentFlux(u, v)
            #vis_amp = np.abs(vis)
            #ax_uv.plot(u, vis_amp)
            #safe_pyplot(st, fig_uv)


        except Exception as e:
            st.error(f"Cannot display component: {e}")

    # ── Aide sur les paramètres ───────────────────────────────────────
    with st.expander("ℹ️ Parameter help"):
        st.markdown("""
        | Param | Description |
        |-------|-------------|
        | x, y | Center position (mas) |
        | f | Relative flux |
        | d | Diameter (mas) |
        | din / dout | Inner / outer diameters (mas) |
        | w | Width (mas) |
        | dx / dy | Box dimensions (mas) |
        | elong | Aspect ratio (ellipticity) |
        | pa | Position angle (°) |
        | fwhm | Full width at half maximum (mas) |
        | hlr | Half-light radius (mas) |
        | flor | Lorentzian fraction |
        | skw / skwPa | Skewness and associated angle |
        | a, a1, a2 | Limb darkening coefficients |
                """)
