# tests/test_interp_registry.py
"""
Regression tests for core/interp_registry.py — every oimInterp macro the
app exposes must actually build and evaluate against the real, installed
oimodeler (it isn't pinned — see the project skill), not just look
plausible on paper. The previous hardcoded "Blackbody" interpolator
looked fine but silently passed the wrong kwarg names (temp=/lum=
instead of T=/L=) straight through **kwargs and was discarded — these
tests exist so a future oimodeler upgrade that renames/reshapes a
constructor breaks a test here instead of silently producing wrong
science in production.

These tests require the real installed `oimodeler` package and are
skipped if it isn't available.
"""
from __future__ import annotations

import numpy as np
import pytest

oim = pytest.importorskip("oimodeler")

from core.component import ComponentConfig
from core.interp_registry import INTERP_REGISTRY
from core.validation import InvalidInput
from core.registry import build_registry


@pytest.fixture(scope="module")
def registry():
    return build_registry(oim)


# One valid, representative kwargs dict per macro — deliberately exercises
# every parameter type the registry declares (arrays, optional_float,
# select, bool, ...), already unit-converted the way the UI form would
# (wl in metres, not µm) so this matches exactly what
# core/component.py's create_instance() receives.
VALID_KWARGS: dict[str, dict] = {
    "wl": {"wl": [2e-6, 3e-6, 4e-6, 5e-6], "values": [0.2, 0.5, 0.8, 0.3],
           "kind": "linear", "fixedRef": True, "extrapolate": False},
    "time": {"mjd": [58000.0, 58100.0, 58200.0], "values": [0.1, 0.5, 0.2],
              "kind": "linear", "fixedRef": True, "extrapolate": False},
    "GaussWl": {"val0": 0.0, "value": 1.0, "x0": 2.2e-6, "fwhm": 0.5e-6},
    "GaussTime": {"val0": 0.0, "value": 1.0, "x0": 0.0, "fwhm": 1.0},
    "mGaussWl": {"val0": 0.0, "values": [1.0, 2.0], "x0": [2e-6, 4e-6], "fwhm": [0.2e-6, 0.3e-6]},
    "mGaussTime": {"val0": 0.0, "values": [1.0, 2.0], "x0": [100.0, 200.0], "fwhm": [10.0, 20.0]},
    "cosTime": {"T0": 0.0, "P": 1.0, "values": [0.0, 1.0], "x0": 0.5},
    "polyWl": {"order": 2, "coeffs": [0.1, 0.2, 0.3], "x0": 0.0},
    "polyTime": {"order": 2, "coeffs": [0.1, 0.2, 0.3], "x0": 0.0},
    "powerlawWl": {"x0": 2.2e-6, "A": 1.0, "p": 1.0},
    "powerlawTime": {"x0": 1.0, "A": 1.0, "p": 1.0},
    "rangeWl": {"wlmin": 2e-6, "wlmax": 3e-6, "values": [0.1, 0.3, 0.5], "kind": "linear"},
    "templateWl": {"wl0": 2e-6, "dwl": 0.01e-6, "f_contrib": 1.0,
                    "values": [0.1, 0.5, 1.0, 0.4, 0.1], "kind": "linear"},
    "tempWl": {"T": 1000.0, "solid_angle": 1e-10},
    "starWl": {"T": 5000.0, "dist": 140.0, "R": 1.0, "L": None},
}


def test_every_registry_macro_has_valid_kwargs_fixture():
    """Guards against a registry entry being added without a matching
    test fixture above (and vice versa)."""
    assert set(VALID_KWARGS) == set(INTERP_REGISTRY)


@pytest.mark.parametrize("macro", sorted(INTERP_REGISTRY))
def test_interpolator_builds_and_evaluates(macro):
    """Every registered interpolator must build via oim.oimInterp(macro,
    **kwargs) and, once attached to a real component parameter, evaluate
    to finite values — exactly the path core/component.py's
    create_instance() exercises in the running app."""
    spec   = INTERP_REGISTRY[macro]
    kwargs = VALID_KWARGS[macro]

    validate = spec.get("validate")
    if validate:
        validate(kwargs)  # must not raise on valid input

    comp = oim.oimUD(d=oim.oimInterp(macro, **kwargs))
    live_param = comp.params["d"]
    assert type(live_param).__name__ == spec["class_name"]

    if spec["dependence"] == "wl":
        result = live_param(np.linspace(1e-6, 15e-6, 20))
    else:
        result = live_param(t=np.linspace(58000.0, 58500.0, 20))

    arr = np.asarray(result, dtype=float)
    assert arr.shape == (20,)
    assert np.all(np.isfinite(arr))


def test_component_config_applies_interpolator_end_to_end(registry):
    """The actual app code path: a component dict with an 'enabled'
    interpolator in the new {macro, kwargs} format, built through
    ComponentConfig.create_instance() (core/component.py), produces a
    component whose parameter is a live interpolator rather than a
    plain float — and the resulting oimModel can actually render."""
    cfg = ComponentConfig(
        component_type="oimUD", registry=registry, name="c1",
        initial_values={"x": 0.0, "y": 0.0, "f": 1.0, "d": 2.0},
        param_ranges={"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.1, 5.0)},
        free_params=["d"],
        interpolators={
            "f": {"enabled": True, "macro": "wl",
                  "kwargs": {"wl": [3e-6, 5e-6], "values": [0.2, 0.8]}},
        },
    )
    instance = cfg.create_instance(oim)
    f_param = instance.params["f"]
    assert type(f_param).__name__ == "oimParamInterpolatorWl"

    model = oim.oimModel(instance)
    image = model.getImage(32, 0.5, wl=4e-6, fromFT=True)
    assert np.all(np.isfinite(image))


def test_interpolator_subparams_respect_per_element_bounds(registry):
    """Regression test: an interpolated parameter isn't a plain oimParam
    anymore (it's swapped for e.g. oimParamInterpolatorWl) — oimodeler
    enumerates its OWN sub-parameters (one oimParam per keyframe) as the
    actual free dimensions for fitting, and each used to silently inherit
    the *pre-interpolation* component class's hardcoded default free
    status (e.g. oimUD.f defaults free=True), completely ignoring
    whatever the UI's free/fixed checkbox and range said. This made
    every interpolated parameter always free in MCMC/grid/random-search
    no matter what the user configured — the concrete cause of a
    reported Emcee "Initial state has a large condition number" failure.
    The fix stores explicit per-sub-parameter bounds in the interpolator
    config's 'bounds' key (see core/interp_registry.py's
    get_full_layout()) rather than reusing the base scalar parameter's
    own free/range for every sub-parameter."""
    def make_cfg(bounds):
        return ComponentConfig(
            component_type="oimUD", registry=registry, name="c1",
            initial_values={"x": 0.0, "y": 0.0, "f": 1.0, "d": 2.0},
            param_ranges={"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.1, 5.0)},
            free_params=["d"],  # 'f' itself not free — irrelevant now that it's interpolated
            interpolators={
                "f": {"enabled": True, "macro": "wl",
                      "kwargs": {"wl": [3e-6, 5e-6], "values": [0.2, 0.8]},
                      "bounds": bounds},
            },
        )

    # Both control points explicitly fixed.
    fixed_model = oim.oimModel(make_cfg({
        "values": [{"free": False, "min": 0.0, "max": 1.0},
                   {"free": False, "min": 0.0, "max": 1.0}],
    }).create_instance(oim))
    assert not any("f_interp" in name for name in fixed_model.getFreeParameters())
    for name, p in fixed_model.getParameters().items():
        if "f_interp" in name:
            assert p.free is False

    # Per-element bounds DIFFER between the two control points — proves
    # they're addressed individually, not one shared range for the pair.
    mixed_model = oim.oimModel(make_cfg({
        "values": [{"free": True, "min": 0.0, "max": 0.5},
                   {"free": False, "min": 0.7, "max": 0.9}],
    }).create_instance(oim))
    all_params = mixed_model.getParameters()
    interp_names = sorted(n for n in all_params if "f_interp" in n)
    assert len(interp_names) == 2
    p0, p1 = (all_params[n] for n in interp_names)
    assert p0.free is True and (p0.min, p0.max) == (0.0, 0.5)
    assert p1.free is False and (p1.min, p1.max) == (0.7, 0.9)
    assert set(mixed_model.getFreeParameters()) & set(interp_names) == {interp_names[0]}


def test_interpolator_mixed_unit_subparams_get_independent_bounds(registry):
    """GaussWl's own .params is [x0, fwhm, val0, value] — x0/fwhm are
    wavelengths (metres), val0/value share the interpolated parameter's
    unit. A single shared bound for the whole list (the pre-per-element
    design) would apply a flux-like [0, 1] range to a wavelength. Confirms
    each of the 4 gets its own independent (free, min, max) and that a
    wavelength-scale bound is never confused with a flux-scale one."""
    cfg = ComponentConfig(
        component_type="oimUD", registry=registry, name="c1",
        initial_values={"x": 0.0, "y": 0.0, "f": 1.0, "d": 2.0},
        param_ranges={"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.1, 5.0)},
        free_params=["d"],
        interpolators={
            "f": {
                "enabled": True, "macro": "GaussWl",
                "kwargs": {"val0": 0.0, "value": 1.0, "x0": 2.2e-6, "fwhm": 0.5e-6},
                "bounds": {
                    "x0":    [{"free": True, "min": 1e-6, "max": 5e-6}],
                    "fwhm":  [{"free": False, "min": 1e-7, "max": 1e-5}],
                    "val0":  [{"free": False, "min": 0.0, "max": 1.0}],
                    "value": [{"free": True, "min": 0.0, "max": 1.0}],
                },
            },
        },
    )
    instance = cfg.create_instance(oim)
    all_params = oim.oimModel(instance).getParameters()
    x0, fwhm, val0, value = instance.params["f"].params

    assert (x0.free, x0.min, x0.max) == (True, 1e-6, 5e-6)
    assert (fwhm.free, fwhm.min, fwhm.max) == (False, 1e-7, 1e-5)
    assert (val0.free, val0.min, val0.max) == (False, 0.0, 1.0)
    assert (value.free, value.min, value.max) == (True, 0.0, 1.0)


def test_interpolator_noncontrollable_subparams_left_to_oimodeler(registry):
    """powerlawWl's .params is [x0, A, p] but x0 is hardcoded free=False
    by oimodeler itself (oimParamPowerLaw._init sets self.x0.free=False
    directly) — no 'bounds' entry should exist for it (get_fittable_layout
    excludes it), and applying the fix must not choke walking past it
    positionally to reach A/p."""
    cfg = ComponentConfig(
        component_type="oimUD", registry=registry, name="c1",
        initial_values={"x": 0.0, "y": 0.0, "f": 1.0, "d": 2.0},
        param_ranges={"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.1, 5.0)},
        free_params=["d"],
        interpolators={
            "f": {
                "enabled": True, "macro": "powerlawWl",
                "kwargs": {"x0": 2e-6, "A": 1.0, "p": 1.0},
                "bounds": {
                    "A": [{"free": True, "min": -10.0, "max": 10.0}],
                    "p": [{"free": False, "min": -5.0, "max": 5.0}],
                },
            },
        },
    )
    instance = cfg.create_instance(oim)
    x0, A, p = instance.params["f"].params
    assert x0.free is False  # oimodeler's own hardcoded default, untouched
    assert (A.free, A.min, A.max) == (True, -10.0, 10.0)
    assert (p.free, p.min, p.max) == (False, -5.0, 5.0)


def test_disabled_interpolator_keeps_plain_float(registry):
    """enabled=False must NOT be swapped for an oimInterp (matches every
    other 'enabled' flag in this codebase's filter/interpolator specs)."""
    cfg = ComponentConfig(
        component_type="oimUD", registry=registry, name="c1",
        initial_values={"x": 0.0, "y": 0.0, "f": 1.0, "d": 2.0},
        param_ranges={"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.1, 5.0)},
        free_params=["d"],
        interpolators={
            "f": {"enabled": False, "macro": "wl",
                  "kwargs": {"wl": [3e-6, 5e-6], "values": [0.2, 0.8]}},
        },
    )
    instance = cfg.create_instance(oim)
    assert type(instance.params["f"]).__name__ != "oimParamInterpolatorWl"
    assert instance.params["f"].value == 1.0


def test_user_func_excluded_for_security():
    """oimParamUserFunc takes an arbitrary Python callable — same class of
    risk as oimFlagWithExpressionFilter's eval() sink (see
    filter_registry.py). Must never be exposed on a public, unauthenticated
    deployment."""
    assert "userFunc" not in INTERP_REGISTRY


# ── Validators actually reject bad input (not dead code) ──────────────

@pytest.mark.parametrize("macro,broken_kwargs,bad_key", [
    ("mGaussWl", {**VALID_KWARGS["mGaussWl"], "fwhm": [0.2e-6]}, "values/x0/fwhm length"),
    ("mGaussWl", {**VALID_KWARGS["mGaussWl"], "fwhm": [0.0, 0.0]}, "fwhm > 0"),
    ("GaussWl", {**VALID_KWARGS["GaussWl"], "fwhm": 0.0}, "fwhm > 0"),
    ("cosTime", {**VALID_KWARGS["cosTime"], "values": [0.0, 1.0, 2.0]}, "values length 2"),
    ("cosTime", {**VALID_KWARGS["cosTime"], "P": 0.0}, "P > 0"),
    ("polyWl", {**VALID_KWARGS["polyWl"], "coeffs": [0.1, 0.2]}, "coeffs length"),
    ("powerlawWl", {**VALID_KWARGS["powerlawWl"], "x0": 0.0}, "x0 != 0"),
    ("rangeWl", {**VALID_KWARGS["rangeWl"], "wlmax": 1e-6}, "wlmax > wlmin"),
    ("templateWl", {**VALID_KWARGS["templateWl"], "dwl": 0.0}, "dwl > 0"),
    ("starWl", {**VALID_KWARGS["starWl"], "R": None, "L": None}, "R or L required"),
    ("starWl", {**VALID_KWARGS["starWl"], "dist": 0.0}, "dist > 0"),
    ("wl", {"wl": [2e-6, 3e-6], "values": [0.1]}, "wl/values length"),
])
def test_validator_rejects_bad_input(macro, broken_kwargs, bad_key):
    validate = INTERP_REGISTRY[macro]["validate"]
    with pytest.raises(InvalidInput):
        validate(broken_kwargs)
