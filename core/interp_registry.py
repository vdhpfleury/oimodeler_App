# core/interp_registry.py
"""
Declarative registry of every oimodeler `oimInterp` parameter
interpolator, mirroring core/filter_registry.py's pattern for the Data
page's filter workbench.

Every class/parameter here was verified against the actually-installed
oimodeler package's source (`inspect.getsource` on `oimodeler.oimParam`),
not copied blindly from the online docs — several real constructor kwargs
differ from what the docs table alone would suggest, and the app's own
previous hardcoded "Blackbody" interpolator was silently broken because
of exactly this: it called `oim.oimInterp('starWl', temp=..., lum=...,
wl=...)`, but the real `oimParamLinearStarWl._init` takes `T=`/`L=`
(not `temp=`/`lum=`) and has no `wl` parameter at all — every value the
UI collected was discarded via `**kwargs`, and the interpolator was
always built with T=0, R=None, L=None regardless of the widget inputs.

Only the macros oimodeler's own `oimodeler.oimParam._interpolators`
dict actually recognizes are listed here (verified directly, not
assumed) — the plain `oimParamInterpolator`/`oimParamInterpolatorKeyframes`
/`oimParamGaussian`/`oimParamMultipleGaussian`/`oimParamPolynomial`/
`oimParamPowerLaw` base classes have no registered macro of their own;
`oim.oimInterp(...)` can only be called with one of their -Wl/-Time
subclasses' macro names, which is what this registry exposes.

`userFunc` (`oimParamUserFunc`) is intentionally NOT included: its
`userfunc` parameter is an arbitrary Python callable, the same class of
code-execution risk as `oimFlagWithExpressionFilter`'s `eval()` sink
(see filter_registry.py's docstring and docs/security_audit_2026-09.md
V5) — there is no way to expose "pick any Python function" as a public,
unauthenticated web form input without it being RCE.

Every numeric parameter declares (min, max) here so the page can
re-validate it server-side before it ever reaches an interpolator
constructor — widget bounds are never trusted on their own (V4). Cross-
parameter constraints that a single widget's (min, max) can't express
(e.g. mGaussWl's values/x0/fwhm needing equal length, starWl needing at
least one of R or L) are enforced by each entry's optional `validate`
callable, raising `core.validation.InvalidInput`.

Pure declarative data (+ small pure validator functions), no Streamlit/
oimodeler import — safe to import from core/ and to unit test in
isolation.
"""
from __future__ import annotations

from core.validation import InvalidInput

# Wavelength bounds shared by every wavelength-in-µm widget here — matches
# the convention already used by core/filter_registry.py and the rest of
# the app (Explorer, Fitting's model image, Modelling's preview).
_WL_MIN_UM, _WL_MAX_UM = 0.1, 30.0

# scipy.interpolate.interp1d's `kind` values that are meaningful for the
# handful of oimodeler interpolators built on it (oimParamInterpolatorWl/
# Time via oimParamInterpolatorKeyframes, oimParamLinearRangeWl,
# oimParamLinearTemplateWl) — verified against their _interpFunction
# source, which passes `kind` straight through to interp1d(kind=...).
_INTERP1D_KINDS = ["linear", "nearest", "previous", "next", "quadratic", "cubic"]


def _require_same_length(names_and_values: list[tuple[str, list]]) -> None:
    lengths = {name: len(v) for name, v in names_and_values}
    if len(set(lengths.values())) > 1:
        raise InvalidInput(
            "These lists must have the same length: "
            + ", ".join(f"{n} ({n_len})" for n, n_len in lengths.items())
        )


def _validate_mgauss(kwargs: dict) -> None:
    _require_same_length([
        ("values", kwargs["values"]), ("x0", kwargs["x0"]), ("fwhm", kwargs["fwhm"]),
    ])
    if any(f <= 0 for f in kwargs["fwhm"]):
        raise InvalidInput("fwhm: every value must be > 0.")


def _validate_gauss(kwargs: dict) -> None:
    if kwargs["fwhm"] <= 0:
        raise InvalidInput("fwhm must be > 0.")


def _validate_cos_time(kwargs: dict) -> None:
    if len(kwargs["values"]) != 2:
        raise InvalidInput("values: exactly 2 numbers required (min, max).")
    if kwargs["P"] <= 0:
        raise InvalidInput("P (period) must be > 0.")
    x0 = kwargs.get("x0")
    if x0 is not None and not (0.0 < x0 < 1.0):
        raise InvalidInput("x0 (inflection point) must be strictly between 0 and 1.")


def _validate_poly(kwargs: dict) -> None:
    if len(kwargs["coeffs"]) != kwargs["order"] + 1:
        raise InvalidInput(
            f"coeffs: expected {kwargs['order'] + 1} values for order "
            f"{kwargs['order']} (order + 1), got {len(kwargs['coeffs'])}."
        )


def _validate_powerlaw(kwargs: dict) -> None:
    if kwargs["x0"] == 0:
        raise InvalidInput("x0 must not be 0 (used as a divisor).")


def _validate_range_wl(kwargs: dict) -> None:
    if kwargs["wlmax"] <= kwargs["wlmin"]:
        raise InvalidInput("wlmax must be greater than wlmin.")
    if len(kwargs["values"]) < 2:
        raise InvalidInput("values: at least 2 points required.")


def _validate_template_wl(kwargs: dict) -> None:
    if kwargs["dwl"] <= 0:
        raise InvalidInput("dwl (wavelength step) must be > 0.")
    if len(kwargs["values"]) < 2:
        raise InvalidInput("values: at least 2 points required.")


def _validate_star_wl(kwargs: dict) -> None:
    if kwargs.get("R") is None and kwargs.get("L") is None:
        raise InvalidInput(
            "Provide at least one of Radius (R) or Luminosity (L) — "
            "without either, oimodeler cannot compute the stellar flux "
            "(it would raise when the model is actually evaluated)."
        )
    if kwargs["dist"] <= 0:
        raise InvalidInput("dist must be > 0.")


def _validate_keyframes(kwargs: dict, keyframe_key: str) -> None:
    if len(kwargs[keyframe_key]) != len(kwargs["values"]):
        raise InvalidInput(f"{keyframe_key} and values must have the same length.")
    if len(kwargs["values"]) < 2:
        raise InvalidInput("At least 2 control points are required.")


INTERP_REGISTRY: dict[str, dict] = {

    # ── Keyframe interpolators ──────────────────────────────────────────
    "wl": {
        "class_name": "oimParamInterpolatorWl",
        "description": "Interpolation between keyframes in wavelength.",
        "dependence": "wl",
        "parameters": {
            "wl":          {"type": "array_float_um", "label": "Wavelengths (µm)",
                             "min": _WL_MIN_UM, "max": _WL_MAX_UM, "max_len": 50},
            "values":      {"type": "array_float", "label": "Values",
                             "min": -1e9, "max": 1e9, "max_len": 50},
            "kind":        {"type": "select", "label": "Interpolation kind",
                             "options": _INTERP1D_KINDS, "default": "linear"},
            "fixedRef":    {"type": "bool", "label": "Fixed reference", "default": True},
            "extrapolate": {"type": "bool", "label": "Extrapolate outside range", "default": False},
        },
        "validate": lambda kw: _validate_keyframes(kw, "wl"),
    },
    "time": {
        "class_name": "oimParamInterpolatorTime",
        "description": "Interpolation between keyframes in MJD (time).",
        "dependence": "mjd",
        "parameters": {
            "mjd":         {"type": "array_float", "label": "MJD control points",
                             "min": -100000.0, "max": 100000.0, "max_len": 50},
            "values":      {"type": "array_float", "label": "Values",
                             "min": -1e9, "max": 1e9, "max_len": 50},
            "kind":        {"type": "select", "label": "Interpolation kind",
                             "options": _INTERP1D_KINDS, "default": "linear"},
            "fixedRef":    {"type": "bool", "label": "Fixed reference", "default": True},
            "extrapolate": {"type": "bool", "label": "Extrapolate outside range", "default": False},
        },
        "validate": lambda kw: _validate_keyframes(kw, "mjd"),
    },

    # ── Gaussian ──────────────────────────────────────────────────────
    "GaussWl": {
        "class_name": "oimParamGaussianWl",
        "description": "Gaussian interpolator in wavelength.",
        "dependence": "wl",
        "parameters": {
            "val0":  {"type": "float", "label": "Baseline value (val0)",
                      "default": 0.0, "min": -1e9, "max": 1e9},
            "value": {"type": "float", "label": "Peak value (value)",
                      "default": 1.0, "min": -1e9, "max": 1e9},
            "x0":    {"type": "wl_um", "label": "Center wavelength x0 (µm)",
                      "default": 2.2, "min": _WL_MIN_UM, "max": _WL_MAX_UM},
            "fwhm":  {"type": "wl_um", "label": "FWHM (µm)",
                      "default": 0.5, "min": 1e-4, "max": _WL_MAX_UM},
        },
        "validate": _validate_gauss,
    },
    "GaussTime": {
        "class_name": "oimParamGaussianTime",
        "description": "Gaussian interpolator in MJD (time).",
        "dependence": "mjd",
        "parameters": {
            "val0":  {"type": "float", "label": "Baseline value (val0)",
                      "default": 0.0, "min": -1e9, "max": 1e9},
            "value": {"type": "float", "label": "Peak value (value)",
                      "default": 1.0, "min": -1e9, "max": 1e9},
            "x0":    {"type": "float", "label": "Center MJD (x0)",
                      "default": 0.0, "min": -100000.0, "max": 100000.0},
            "fwhm":  {"type": "float", "label": "FWHM (days)",
                      "default": 1.0, "min": 1e-4, "max": 100000.0},
        },
        "validate": _validate_gauss,
    },

    # ── Multiple Gaussian ─────────────────────────────────────────────
    "mGaussWl": {
        "class_name": "oimParamMultipleGaussianWl",
        "description": "Multiple Gaussian interpolator in wavelength.",
        "dependence": "wl",
        "parameters": {
            "val0":   {"type": "float", "label": "Baseline value (val0)",
                       "default": 0.0, "min": -1e9, "max": 1e9},
            "values": {"type": "array_float", "label": "Peak values (one per Gaussian)",
                       "min": -1e9, "max": 1e9, "max_len": 20},
            "x0":     {"type": "array_float_um", "label": "Centers x0 (µm, one per Gaussian)",
                       "min": _WL_MIN_UM, "max": _WL_MAX_UM, "max_len": 20},
            "fwhm":   {"type": "array_float_um", "label": "FWHM (µm, one per Gaussian)",
                       "min": 1e-4, "max": _WL_MAX_UM, "max_len": 20},
        },
        "validate": _validate_mgauss,
    },
    "mGaussTime": {
        "class_name": "oimParamMultipleGaussianTime",
        "description": "Multiple Gaussian interpolator in MJD (time).",
        "dependence": "mjd",
        "parameters": {
            "val0":   {"type": "float", "label": "Baseline value (val0)",
                       "default": 0.0, "min": -1e9, "max": 1e9},
            "values": {"type": "array_float", "label": "Peak values (one per Gaussian)",
                       "min": -1e9, "max": 1e9, "max_len": 20},
            "x0":     {"type": "array_float", "label": "Centers x0 (MJD, one per Gaussian)",
                       "min": -100000.0, "max": 100000.0, "max_len": 20},
            "fwhm":   {"type": "array_float", "label": "FWHM (days, one per Gaussian)",
                       "min": 1e-4, "max": 100000.0, "max_len": 20},
        },
        "validate": _validate_mgauss,
    },

    # ── Cosine (time only) ───────────────────────────────────────────
    "cosTime": {
        "class_name": "oimParamCosineTime",
        "description": "Asymmetrical cosine time interpolator.",
        "dependence": "mjd",
        "parameters": {
            "T0":     {"type": "float", "label": "T0 — start (MJD)",
                       "default": 0.0, "min": -100000.0, "max": 100000.0},
            "P":      {"type": "float", "label": "P — period (days)",
                       "default": 1.0, "min": 1e-6, "max": 100000.0},
            "values": {"type": "array_float", "label": "Values [min, max]",
                       "min": -1e9, "max": 1e9, "max_len": 2},
            "x0":     {"type": "optional_float", "label": "x0 — asymmetry inflection (0-1)",
                       "min": 0.0001, "max": 0.9999},
        },
        "validate": _validate_cos_time,
    },

    # ── Polynomial ────────────────────────────────────────────────────
    "polyWl": {
        "class_name": "oimParamPolynomialWl",
        "description": "Polynomial interpolation in wavelength.",
        "dependence": "wl",
        "parameters": {
            "order":  {"type": "int", "label": "Order", "default": 2, "min": 0, "max": 10},
            "coeffs": {"type": "array_float", "label": "Coefficients (order + 1 values, "
                       "highest degree first)", "min": -1e9, "max": 1e9, "max_len": 11},
            "x0":     {"type": "wl_um", "label": "Reference wavelength x0 (µm)",
                       "default": 0.0, "min": 0.0, "max": _WL_MAX_UM},
        },
        "validate": _validate_poly,
    },
    "polyTime": {
        "class_name": "oimParamPolynomialTime",
        "description": "Polynomial interpolation in MJD (time).",
        "dependence": "mjd",
        "parameters": {
            "order":  {"type": "int", "label": "Order", "default": 2, "min": 0, "max": 10},
            "coeffs": {"type": "array_float", "label": "Coefficients (order + 1 values, "
                       "highest degree first)", "min": -1e9, "max": 1e9, "max_len": 11},
            "x0":     {"type": "float", "label": "Reference MJD (x0)",
                       "default": 0.0, "min": -100000.0, "max": 100000.0},
        },
        "validate": _validate_poly,
    },

    # ── Power law ─────────────────────────────────────────────────────
    "powerlawWl": {
        "class_name": "oimParamPowerLawWl",
        "description": "Power-law interpolation in wavelength.",
        "dependence": "wl",
        "parameters": {
            "x0": {"type": "wl_um", "label": "Reference wavelength x0 (µm)",
                   "default": 2.2, "min": _WL_MIN_UM, "max": _WL_MAX_UM},
            "A":  {"type": "float", "label": "Scale factor (A)",
                   "default": 1.0, "min": -1e9, "max": 1e9},
            "p":  {"type": "float", "label": "Index (p)",
                   "default": 1.0, "min": -100.0, "max": 100.0},
        },
        "validate": _validate_powerlaw,
    },
    "powerlawTime": {
        "class_name": "oimParamPowerLawTime",
        "description": "Power-law interpolation in MJD (time).",
        "dependence": "mjd",
        "parameters": {
            "x0": {"type": "float", "label": "Reference MJD (x0)",
                   "default": 1.0, "min": -100000.0, "max": 100000.0},
            "A":  {"type": "float", "label": "Scale factor (A)",
                   "default": 1.0, "min": -1e9, "max": 1e9},
            "p":  {"type": "float", "label": "Index (p)",
                   "default": 1.0, "min": -100.0, "max": 100.0},
        },
        "validate": _validate_powerlaw,
    },

    # ── Linear range / template / blackbody (wavelength only) ─────────
    "rangeWl": {
        "class_name": "oimParamLinearRangeWl",
        "description": "Linear range interpolation in wavelength — `values` evenly "
                        "spaced between wlmin and wlmax.",
        "dependence": "wl",
        "parameters": {
            "wlmin":  {"type": "wl_um", "label": "wlmin (µm)",
                       "default": 2.0, "min": _WL_MIN_UM, "max": _WL_MAX_UM},
            "wlmax":  {"type": "wl_um", "label": "wlmax (µm)",
                       "default": 3.0, "min": _WL_MIN_UM, "max": _WL_MAX_UM},
            "values": {"type": "array_float", "label": "Values",
                       "min": -1e9, "max": 1e9, "max_len": 50},
            "kind":   {"type": "select", "label": "Interpolation kind",
                       "options": _INTERP1D_KINDS, "default": "linear"},
        },
        "validate": _validate_range_wl,
    },
    "templateWl": {
        "class_name": "oimParamLinearTemplateWl",
        "description": "Interpolation in wavelength using an external regular-grid "
                        "template (e.g. an emission-line profile), normalized to its "
                        "own maximum and scaled by f_contrib.",
        "dependence": "wl",
        "parameters": {
            "wl0":       {"type": "wl_um", "label": "wl0 — grid start (µm)",
                          "default": 2.0, "min": _WL_MIN_UM, "max": _WL_MAX_UM},
            "dwl":       {"type": "wl_um", "label": "dwl — grid step (µm)",
                          "default": 0.01, "min": 1e-5, "max": 5.0},
            "f_contrib": {"type": "float", "label": "Flux contribution (f_contrib)",
                          "default": 1.0, "min": 0.0, "max": 1e6},
            "values":    {"type": "array_float", "label": "Template values (regular grid)",
                          "min": -1e9, "max": 1e9, "max_len": 200},
            "kind":      {"type": "select", "label": "Interpolation kind",
                          "options": _INTERP1D_KINDS, "default": "linear"},
        },
        "validate": _validate_template_wl,
    },
    "tempWl": {
        "class_name": "oimParamLinearTemperatureWl",
        "description": "Blackbody flux in wavelength for a given temperature "
                        "(Planck's law) — meant for a flux-like parameter.",
        "dependence": "wl",
        "parameters": {
            "T":           {"type": "float", "label": "Temperature T (K)",
                            "default": 1000.0, "min": 0.0, "max": 3000.0},
            "solid_angle": {"type": "float", "label": "Solid angle (steradian)",
                            "default": 1e-10, "min": 0.0, "max": 1.0, "format": "%.3e"},
        },
    },
    "starWl": {
        "class_name": "oimParamLinearStarWl",
        "description": "Blackbody stellar flux in wavelength for a given effective "
                        "temperature, distance, and radius or luminosity (provide "
                        "one of the two — radius takes priority if both are set).",
        "dependence": "wl",
        "parameters": {
            "T":    {"type": "float", "label": "Effective temperature T (K)",
                     "default": 5000.0, "min": 100.0, "max": 100000.0},
            "dist": {"type": "float", "label": "Distance (pc)",
                     "default": 140.0, "min": 0.001, "max": 1e6},
            "R":    {"type": "optional_float", "label": "Radius R (solar radii)",
                     "min": 1e-6, "max": 1e6},
            "L":    {"type": "optional_float", "label": "Luminosity L (solar luminosities)",
                     "min": 1e-6, "max": 1e9},
        },
        "validate": _validate_star_wl,
    },
}


# ── Which of an applied interpolator's OWN kwargs become oimodeler
# sub-parameters, in what order, and which of those are actually
# free-fittable ───────────────────────────────────────────────────────
# oimodeler exposes every interpolator's sub-parameters through a single
# flat `.params` list (see oimParamInterpolator.params, built from each
# subclass's `_getParams()`) — but that list silently MIXES parameters
# of different physical kinds (e.g. GaussWl's `.params` is
# `[x0, fwhm, val0, value]`: x0/fwhm are wavelengths, val0/value are the
# interpolated parameter's own unit), and some entries in it are
# hardcoded `free=False` by oimodeler itself (verified by building each
# interpolator and reading back `.params[i].free`) regardless of the
# base parameter's own free status: oimParamPowerLaw's x0,
# oimParamLinearRangeWl's wlmin/wlmax, oimParamLinearTemplateWl's
# f_contrib, oimParamLinearStarWl's T/R/L/dist. Applying one shared
# (free, min, max) to the whole list — what an earlier version of this
# app did — is wrong on both counts: it could bound a wavelength
# (metres) by a flux parameter's [0, 1] range, and it could try to make
# a permanently-fixed sub-parameter "free" for nothing.
#
# Each entry below is the FULL, ordered `.params` composition (verified
# with inspect.getsource + live construction against the installed
# oimodeler, not assumed) as (kwarg_name, count, controllable) triples —
# `controllable=False` marks a slot oimodeler never lets vary, which
# core/component.py must skip over (not zero out) when walking
# `.params`, and the UI never renders a free/bounds control for.
# templateWl and starWl end up with no controllable entries at all —
# nothing either exposes is ever fittable in the installed oimodeler.
def _keyframe_layout(kwargs: dict, keyframe_key: str) -> list[tuple[str, int, bool]]:
    layout = []
    if not kwargs.get("fixedRef", True):
        layout.append((keyframe_key, len(kwargs[keyframe_key]), True))
    layout.append(("values", len(kwargs["values"]), True))
    return layout


_FULL_LAYOUT_FUNCS = {
    "wl":           lambda kw: _keyframe_layout(kw, "wl"),
    "time":         lambda kw: _keyframe_layout(kw, "mjd"),
    "GaussWl":      lambda kw: [("x0", 1, True), ("fwhm", 1, True), ("val0", 1, True), ("value", 1, True)],
    "GaussTime":    lambda kw: [("x0", 1, True), ("fwhm", 1, True), ("val0", 1, True), ("value", 1, True)],
    "mGaussWl":     lambda kw: [("val0", 1, True), ("x0", len(kw["x0"]), True),
                                 ("fwhm", len(kw["fwhm"]), True), ("values", len(kw["values"]), True)],
    "mGaussTime":   lambda kw: [("val0", 1, True), ("x0", len(kw["x0"]), True),
                                 ("fwhm", len(kw["fwhm"]), True), ("values", len(kw["values"]), True)],
    "cosTime":      lambda kw: (
        [("T0", 1, True), ("P", 1, True)]
        + ([("x0", 1, True)] if kw.get("x0") is not None else [])
        + [("values", len(kw["values"]), True)]
    ),
    "polyWl":       lambda kw: [("coeffs", len(kw["coeffs"]), True)],
    "polyTime":     lambda kw: [("coeffs", len(kw["coeffs"]), True)],
    "powerlawWl":   lambda kw: [("x0", 1, False), ("A", 1, True), ("p", 1, True)],
    "powerlawTime": lambda kw: [("x0", 1, False), ("A", 1, True), ("p", 1, True)],
    "rangeWl":      lambda kw: [("values", len(kw["values"]), True),
                                 ("wlmin", 1, False), ("wlmax", 1, False)],
    "templateWl":   lambda kw: [("f_contrib", 1, False)],
    "tempWl":       lambda kw: [("T", 1, True)],
    "starWl":       lambda kw: [("T", 1, False)],
}


def get_full_layout(macro: str, kwargs: dict) -> list[tuple[str, int, bool]]:
    """Ordered (kwarg_name, count, controllable) triples spanning oimodeler's
    ENTIRE `.params` list for this macro+kwargs — including the slots it
    hardcodes free=False for. Use this to walk `.params` positionally
    (core/component.py); use get_fittable_layout() for the subset the UI
    should offer free/bounds controls for."""
    return _FULL_LAYOUT_FUNCS[macro](kwargs)


def get_fittable_layout(macro: str, kwargs: dict) -> list[tuple[str, int]]:
    """Ordered (kwarg_name, count) pairs for only the sub-parameters
    oimodeler actually lets vary — what the UI should render a
    free/bounds control for, one row per element (count > 1 for a
    list-valued kwarg)."""
    return [(name, count) for name, count, controllable in get_full_layout(macro, kwargs)
            if controllable]
