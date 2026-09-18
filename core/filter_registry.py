# core/filter_registry.py
"""
Declarative registry of the oimodeler data filters exposed by the Data
page's filter workbench.

Every parameter list here was verified against the actually-installed
oimodeler package's source (`inspect.getsource`), not copied blindly from
any external reference — several real defaults/behaviours differ from
what a stale or version-mismatched reference might suggest (e.g.
oimWavelengthRangeFilter's `method` is really "cut" vs. anything else,
not "flag"/"remove"; oimDiffErrFilter's `rangeType` really defaults to
"index", not "inside"/"outside"; oimSetMinErrFilter has a `relThreshold`
param the params list must not silently drop; oimSetMinErrFilter's
`values`/`relThreshold` must have the same length as `dataType` or
oimUtils.setMinimumError raises IndexError applying the filter — see
that entry's own comment).

`oimFlagWithExpressionFilter` is intentionally NOT included: its `expr`
parameter reaches oimodeler's `oifitsFlagWithExpression`, which does a
literal `eval()` — exposing it as free text on a public, unauthenticated
deployment is a code-execution risk (see docs/security_audit_2026-09.md
V5). It can be re-added later behind the same strict allowlist
core.validation.filter_expression() already provides, if a real need
comes up.

Every numeric parameter declares (min, max) here so the page can
re-validate it server-side before it ever reaches a filter constructor —
widget bounds are never trusted on their own (V4).

Pure declarative data, no Streamlit/oimodeler import — safe to import
from core/ and to unit test in isolation.
"""
from __future__ import annotations

# Wavelength bounds shared by every wavelength-in-µm widget in this
# registry — matches the convention used elsewhere in the app (Explorer,
# Fitting's model image, Modelling's preview).
_WL_MIN_UM, _WL_MAX_UM = 0.1, 30.0

FILTER_REGISTRY: dict[str, dict] = {

    # ── Array / table filters ───────────────────────────────────────
    "oimRemoveArrayFilter": {
        "description": "Remove array/table(s) by name (OI_VIS2, OI_T3, ...).",
        "parameters": {
            "targets": {"type": "target", "label": "Targets"},
            "arr":     {"type": "array",  "label": "Arrays / Tables"},
        },
    },
    "oimRemoveInsnameFilter": {
        "description": "Remove arrays/tables by INSNAME(s).",
        "parameters": {
            "targets": {"type": "target",  "label": "Targets"},
            "arr":     {"type": "array",   "label": "Arrays / Tables"},
            "insname": {"type": "insname", "label": "INSNAME"},
        },
    },

    # ── Data type filters ────────────────────────────────────────────
    "oimDataTypeFilter": {
        "description": "Set column values to 0 by column name(s) (VIS2DATA, VISAMP, ...).",
        "parameters": {
            "targets":  {"type": "target",   "label": "Targets"},
            "arr":      {"type": "array",    "label": "Arrays / Tables"},
            "dataType": {"type": "dataType", "label": "Data type"},
        },
    },
    "oimKeepDataTypeFilter": {
        "description": "Keep only the specified columns (VIS2DATA, VISAMP, ...).",
        "parameters": {
            "targets":  {"type": "target",   "label": "Targets"},
            "arr":      {"type": "array",    "label": "Arrays / Tables"},
            "dataType": {"type": "dataType", "label": "Data type"},
        },
    },

    # ── Flags ──────────────────────────────────────────────────────
    "oimResetFlagsFilter": {
        "description": "Unflag data (set all flags to False).",
        "parameters": {
            "targets": {"type": "target", "label": "Targets"},
            "arr":     {"type": "array",  "label": "Arrays / Tables"},
        },
    },

    # ── Wavelength filters ───────────────────────────────────────────
    "oimWavelengthRangeFilter": {
        "description": "Cut the wavelength range(s) — keeps points inside the given range(s).",
        "parameters": {
            "targets": {"type": "target", "label": "Targets"},
            "arr":     {"type": "array",  "label": "Arrays / Tables"},
            "wlRange": {"type": "wavelength_range", "label": "Wavelength range to keep"},
            "method":  {
                "type": "select", "label": "Method",
                "options": ["cut", "flag"],
                "default": "cut",
                "help": "cut: physically removes points outside the range. "
                        "flag: keeps the array shape, flags points outside instead.",
            },
        },
    },
    "oimWavelengthShiftFilter": {
        "description": "Shift the wavelength table.",
        "parameters": {
            "targets": {"type": "target", "label": "Targets"},
            "arr":     {"type": "array",  "label": "Arrays / Tables"},
            "wlShift": {
                "type": "float_um", "label": "Wavelength shift",
                "default": 0.0, "min": -100.0, "max": 100.0,
            },
        },
    },
    "oimWavelengthSmoothingFilter": {
        "description": "Apply spectral smoothing.",
        "parameters": {
            "targets":        {"type": "target", "label": "Targets"},
            "arr":            {"type": "array",  "label": "Arrays / Tables"},
            "smoothPix":      {"type": "int",  "label": "Smoothing kernel (pixels)",
                                "default": 2, "min": 1, "max": 50},
            "normalizeError": {"type": "bool", "label": "Normalize error", "default": True},
        },
    },
    "oimWavelengthBinningFilter": {
        "description": "Apply spectral binning.",
        "parameters": {
            "targets":        {"type": "target", "label": "Targets"},
            "arr":            {"type": "array",  "label": "Arrays / Tables"},
            "bin":            {"type": "int",  "label": "Binning factor",
                                "default": 1, "min": 1, "max": 50},
            "normalizeError": {"type": "bool", "label": "Normalize error", "default": True},
        },
    },
    "oimWavelengthIntpBinFilter": {
        "description": "Bin the wavelength to a fixed grid, interpolating at bin edges.",
        "parameters": {
            "targets":       {"type": "target", "label": "Targets"},
            "arr":           {"type": "array",  "label": "Arrays / Tables"},
            "binGrid":       {"type": "array_float_um", "label": "Bin grid (µm, comma-separated)",
                               "min": _WL_MIN_UM, "max": _WL_MAX_UM, "max_len": 200},
            "resetFlags":    {"type": "bool",  "label": "Reset flags", "default": True},
            "averageError":  {"type": "bool",  "label": "Average error", "default": False},
            "nSpecChannels": {"type": "float", "label": "Number of spectral channels",
                               "default": 1.0, "min": 0.1, "max": 1000.0},
        },
    },

    # ── Baselines ─────────────────────────────────────────────────
    "oimKeepBaselinesFilter": {
        "description": "Keep only the specified baseline(s).",
        "parameters": {
            "targets":     {"type": "target",   "label": "Targets"},
            "arr":         {"type": "array",    "label": "Arrays / Tables"},
            "baselines":   {"type": "baseline", "label": "Baselines"},
            "keepOldFlag": {"type": "bool", "label": "Keep old flags", "default": True},
        },
    },
    "oimRemoveBaselinesFilter": {
        "description": "Remove the specified baseline(s).",
        "parameters": {
            "targets":     {"type": "target",   "label": "Targets"},
            "arr":         {"type": "array",    "label": "Arrays / Tables"},
            "baselines":   {"type": "baseline", "label": "Baselines"},
            "keepOldFlag": {"type": "bool", "label": "Keep old flags", "default": True},
        },
    },

    # ── Telescopes ────────────────────────────────────────────────
    "oimKeepTelescopesFilter": {
        "description": "Keep only the specified telescope(s).",
        "parameters": {
            "targets":     {"type": "target",    "label": "Targets"},
            "arr":         {"type": "array",     "label": "Arrays / Tables"},
            "telescopes":  {"type": "telescope", "label": "Telescopes"},
            "keepOldFlag": {"type": "bool", "label": "Keep old flags", "default": True},
        },
    },
    "oimRemoveTelescopesFilter": {
        "description": "Remove the specified telescope(s).",
        "parameters": {
            "targets":     {"type": "target",    "label": "Targets"},
            "arr":         {"type": "array",     "label": "Arrays / Tables"},
            "telescopes":  {"type": "telescope", "label": "Telescopes"},
            "keepOldFlag": {"type": "bool", "label": "Keep old flags", "default": True},
        },
    },

    # ── Error filters ─────────────────────────────────────────────
    "oimDiffErrFilter": {
        "description": "Compute a differential error from the std of the signal inside "
                        "or outside a range (channel index, or wavelength in µm).",
        "parameters": {
            "targets":      {"type": "target", "label": "Targets"},
            "arr":          {"type": "array",  "label": "Arrays / Tables"},
            "rangeType":    {
                "type": "select", "label": "Range type",
                "options": ["index", "wavelength"], "default": "index",
            },
            "ranges":       {"type": "diff_err_range", "label": "Range"},
            "excludeRange": {"type": "bool", "label": "Exclude range", "default": False},
            "dataType":     {"type": "dataType", "label": "Data type"},
        },
    },
    "oimSetMinErrFilter": {
        "description": "Set a minimum error on data — % for visibilities, degrees for phases.",
        "parameters": {
            "targets":      {"type": "target",   "label": "Targets"},
            "arr":          {"type": "array",    "label": "Arrays / Tables"},
            # dataType MUST be declared (and rendered) before values/
            # relThreshold: oimodeler's setMinimumError() requires both to
            # be either a single scalar (broadcast to every data type) or
            # a list of EXACTLY the same length as dataType, in the same
            # order — passing a shorter list raises IndexError deep
            # inside oimUtils.setMinimumError as soon as the filter is
            # applied (reproduced: selecting 2 data types with only 1
            # minimum-error value crashes the whole page). The per-
            # datatype widget types below read this dataType selection
            # from its own session_state key (see _render_param) to
            # render exactly one value per selected data type.
            "dataType":     {"type": "dataType", "label": "Data type"},
            "values":       {"type": "per_datatype_float", "label": "Minimum error",
                              "default": 5.0, "min": 0.0, "max": 1000.0},
            "relThreshold": {
                "type": "per_datatype_optional_float",
                "label": "Relative threshold (VISAMP/VIS2DATA only)",
                "min": 0.0, "max": 1000.0,
            },
        },
    },
}
