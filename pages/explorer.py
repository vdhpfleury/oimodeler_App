# pages/explorer.py
"""
Page "Component Explorer" – Exploration interactive des composants oimodeler.

Dépendances :
- services/data_service.py  → get_oim(), get_registry()
- components/plots.py       → safe_pyplot()
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from services.data_service import get_oim, get_registry
from core.validation import num, choice, InvalidInput
from components.plots import safe_pyplot

logger = logging.getLogger(__name__)


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
    'P':     ("P",                      0.,  100.,   0.1,  0.05),
    'width': ("width",                  0.,  100.,   0.1,  0.05),
    'dim':   ("dim (image dimension px)", 16, 512, 128, 16),
    'rin':   ("rin (inner radius au)",  0.,  50.,  0.5,  0.1),
    'rout':  ("rout (outer radius au)", 0.,  200., 5.,   0.5),
    'r0':    ("r0 (ref. radius au)",    0.1, 50.,  1.,   0.1),
    'T0':    ("T0 (ref. temperature K)", 10., 3000., 300., 10.),
    'Mdust': ("Mdust (dust mass Msun)", 0.,  5.,   0.2,  0.05),
    'q':     ("q (temperature exponent)", -1., 0.,  -0.5, 0.05),
    'p':     ("p (density exponent)",   -3.,  3.,   0.,   0.1),
    'kappa_abs': ("kappa_abs (opacity cm2/g)", 0., 10.,  1.,   0.1),
    'dist':  ("dist (distance pc)",     1.,  10000., 100., 10.),
    'a3':    ("a3",                     0.,  1.,   0.1,  0.05),
    'a4':    ("a4",                     0.,  1.,   0.1,  0.05),
    'h':     ("h (rim height mas)",     0.,  10.,  1.,   0.1),
    'incl':  ("incl (inclination °)",   0.,  90.,  30.,  1.),
}


def render() -> None:
    registry = get_registry()
    oim      = get_oim()

    # Exclut les composants avancés (trop de paramètres pour l'explorateur)
    visu_components = {
        k: v['params'] for k, v in registry.items()
        if k not in ('oimStarHaloGaussLorentz', 'oimStarHaloIRing')
    }

    # ── Block 1 : component selection + parameters ─────────────────────
    st.markdown("##### Component & parameters")
    comp_options = list(visu_components.keys())
    selected_comp_raw = st.selectbox(
        "Choose a component",
        comp_options,
        format_func=lambda x: f"{x}  —  {registry[x]['description']}",
    )
    try:
        # selectbox returns an unrecognized client value as-is — a
        # forged component name would otherwise reach registry[...]
        # and raise an unhandled KeyError (V4).
        selected_comp = choice(selected_comp_raw, comp_options, "Component")
    except InvalidInput as exc:
        st.error(str(exc))
        selected_comp = comp_options[0]

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

    st.markdown("---")

    # ── Block 2 : image parameters/plot  |  visibility vs baseline ─────
    col_img, col_vis = st.columns(2)

    with col_img:
        st.markdown("##### Image")
        with st.expander("image parameters", expanded=True):
            ic1, ic2 = st.columns(2)
            with ic1:
                img_dim_raw = st.number_input("dimension in px", value=128, min_value=16, max_value=1024, key="img param dim")
                px_size_raw = st.number_input("px size  in mas", value=0.5, min_value=0.01, max_value=10., key="img param px")
                wl_um_raw   = st.number_input("wavelength in µm", value=3.5, min_value=0.1,
                                          max_value=20., key="img param wl")
            with ic2:
                gamma_raw = st.number_input("gamma", value=0.2, key="img param gamma")
                clip_lo_raw = st.number_input("colormap percentile min", value=0.5,
                                          min_value=0., max_value=100., key="img param clip lo")
                clip_hi_raw = st.number_input("colormap percentile max", value=99.5,
                                          min_value=0., max_value=100., key="img param clip hi")

    with col_vis:
        st.markdown("##### Visibility vs baseline")
        with st.expander("visibility vs baseline parameters", expanded=True):
            vb1, vb2 = st.columns(2)
            with vb1:
                b_max_raw = st.number_input(
                    "Max baseline (m)", value=200., min_value=1., max_value=1000.,
                    key="vb param bmax",
                )
                b_n_raw = st.number_input(
                    "Number of points", value=200, min_value=10, max_value=1000,
                    key="vb param n",
                )
            with vb2:
                vb_wl_um_raw = st.number_input(
                    "wavelength in µm", value=3.5, min_value=0.1, max_value=20.,
                    key="vb param wl",
                )

    try:
        # Widget bounds are cosmetic only — img_dim/b_n feed array
        # allocations directly, the concrete OOM DoS vector from the
        # audit (V6).
        img_dim   = num(img_dim_raw, 16, 1024, "Dimension", integer=True)
        px_size   = num(px_size_raw, 0.01, 10.0, "Pixel size")
        gamma     = num(gamma_raw, 0.01, 5.0, "Gamma")
        wl_val    = num(wl_um_raw, 0.1, 20.0, "Wavelength") * 1e-6
        b_max     = num(b_max_raw, 1., 1000., "Max baseline")
        b_n       = num(b_n_raw, 10, 1000, "Number of points", integer=True)
        vb_wl_val = num(vb_wl_um_raw, 0.1, 20.0, "Visibility wavelength") * 1e-6

        # Widget bounds aren't server-enforced — re-validate before use.
        clip_lo, clip_hi = sorted((
            min(max(float(clip_lo_raw), 0.), 100.),
            min(max(float(clip_hi_raw), 0.), 100.),
        ))

        comp_cls  = registry[selected_comp]['class']
        comp_inst = comp_cls(**visu_params)
        mdl       = oim.oimModel(comp_inst)
    except InvalidInput as exc:
        st.warning(str(exc))
    except Exception as e:
        st.error(f"Cannot build component: {e}")
    else:
        # ── Image ────────────────────────────────────────────────────
        with col_img:
            try:
                # Astronomical convention: RA increases to the left (East left).
                extent_half = img_dim * px_size / 2
                extent = [extent_half, -extent_half, -extent_half, extent_half]

                try:
                    im = mdl.getImage(img_dim, px_size, wl=wl_val, fromFT=False)
                    if not np.any(im) or not np.all(np.isfinite(im)):
                        # Some component classes (e.g. radial-profile-based
                        # ones like oimTempGrad) don't implement a direct
                        # image and silently fall back to an all-zero stub
                        # instead of raising — treat that the same as an
                        # error so the fromFT=True fallback actually runs.
                        raise ValueError("direct image unavailable or degenerate")
                except Exception:
                    # fromFT=False fails for some components (no analytic
                    # image) — fall back to the Fourier-transform path.
                    im = mdl.getImage(img_dim, px_size, wl=wl_val, fromFT=True)

                display_im = im ** gamma
                vmin, vmax = np.percentile(display_im, [clip_lo, clip_hi])
                fig, ax = plt.subplots(figsize=(6, 6))
                im_disp = ax.imshow(display_im, cmap='hot', origin='lower', extent=extent,
                                    vmin=vmin, vmax=vmax)
                ax.set_xlabel('ΔRA (mas)')
                ax.set_ylabel('ΔDec (mas)')
                ax.set_title(f'{selected_comp}  –  γ = {gamma}')
                plt.colorbar(im_disp, ax=ax, label='Intensity (γ corrected)')
                safe_pyplot(st, fig)
            except Exception as e:
                st.error(f"Cannot display component image: {e}")

        # ── Visibility vs baseline (East-West / North-South) ───────────
        # u (ucoord) is the Fourier conjugate of the component's x /
        # RA axis, v (vcoord) of y / Dec — same convention as the
        # image above, not an independent assumption (see
        # oimComponentFourier.getComplexCoherentFlux: ucoord/vcoord
        # feed fxp/fyp exactly like x_arr/y_arr do in getImage()).
        with col_vis:
            try:
                baselines = np.linspace(0., b_max, num=b_n)
                spf   = baselines / vb_wl_val
                zeros = np.zeros_like(spf)

                # At the zero-baseline point, oimodeler's Bessel-based
                # visibility formulas (e.g. 2*J1(x)/x for a uniform disk)
                # hit a 0/0 — already handled correctly internally via
                # np.nan_to_num(..., nan=1), the true V(0)=1 limit, but
                # numpy still warns on the underlying division. Silenced
                # here since the result is right; not silenced globally.
                with np.errstate(invalid='ignore', divide='ignore'):
                    ccf_ew = mdl.getComplexCoherentFlux(spf, zeros, wl=vb_wl_val)
                    ccf_ns = mdl.getComplexCoherentFlux(zeros, spf, wl=vb_wl_val)

                v_ew = np.abs(ccf_ew)
                v_ns = np.abs(ccf_ns)
                # Normalize by the zero-baseline response (= total flux),
                # the true maximum for any physically valid (non-negative)
                # image — same result as /max() but explicit about why.
                norm = v_ew[0] if v_ew[0] > 0 else 1.0
                v_ew = v_ew / norm
                v_ns = v_ns / norm

                fig_vis, ax_vis = plt.subplots(figsize=(6, 6))
                ax_vis.plot(baselines, v_ew, label='East–West', color='tab:blue')
                ax_vis.plot(baselines, v_ns, label='North–South', color='tab:orange', ls='--')
                ax_vis.set_xlabel('Baseline length (m)')
                ax_vis.set_ylabel('Normalized visibility')
                ax_vis.set_ylim(-0.02, 1.05)
                ax_vis.set_title(f'{selected_comp}  –  λ = {vb_wl_val * 1e6:.2f} µm')
                ax_vis.legend()
                ax_vis.grid(alpha=0.3)
                safe_pyplot(st, fig_vis)
            except Exception:
                logger.exception("Visibility-vs-baseline rendering failed")
                st.warning(
                    "Could not render the visibility-vs-baseline plot for "
                    "the current settings."
                )

    st.markdown("---")

    # ── Block 3 : reproducible code  |  parameter help ──────────────────
    col_code, col_help = st.columns(2)
    with col_code:
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

    with col_help:
        st.subheader("Parameter help")
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
