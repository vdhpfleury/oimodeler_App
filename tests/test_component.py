# tests/test_component.py
"""
Regression tests for core/component.py bugs found during beta testing
around oimTempGrad ("Florentin's" reported errors):

- 'dim' (image/grid-resolution) must not be free by default: leaving it
  free lets scipy/Emcee write a numpy.float64 into it every iteration,
  crashing oimodeler's own np.linspace(..., dim) calls with
  "'numpy.float64' object cannot be interpreted as an integer".
- 'kappa_abs' (the only param name containing an underscore) must not be
  silently skipped when applying free/min/max — create_instance()'s naive
  `param_name.split('_')[-1]` used to yield "abs", which isn't in
  param_names, so the check was always skipped for this parameter.
- 'rin' must not default to a range starting at 0: oimodeler's default
  (logarithmic) radial grid raises ValueError("Logarithmic grid requires
  rin > 0.") for rin<=0 on every model evaluation.

These tests require the real installed `oimodeler` package (it isn't
pinned - see the project skill) and are skipped if it isn't available.
"""
from __future__ import annotations

import pytest

oim = pytest.importorskip("oimodeler")

from config.constants import DEFAULT_PARAM_RANGES
from core.registry import build_registry
from core.component import ComponentConfig, make_comp_dict


@pytest.fixture(scope="module")
def registry():
    return build_registry(oim)


def test_dim_excluded_from_default_free_params(registry):
    cfg = ComponentConfig(component_type="oimTempGrad", registry=registry, name="tg")
    assert "dim" not in cfg.free_params
    assert "elong" in cfg.free_params  # sanity: other params still default-free


def test_make_comp_dict_excludes_dim_from_free_params(registry):
    d = make_comp_dict("oimTempGrad", "tg", registry)
    assert "dim" not in d["free_params"]


def test_kappa_abs_free_min_max_applied(registry):
    cfg = ComponentConfig(
        component_type="oimTempGrad", registry=registry, name="tg",
        free_params=["kappa_abs"],
        param_ranges={"kappa_abs": (2.0, 5.0)},
    )
    instance = cfg.create_instance(oim)
    kappa = instance.params["kappa_abs"]
    assert kappa.free is True
    assert kappa.min == 2.0
    assert kappa.max == 5.0


def test_rin_default_range_is_strictly_positive():
    lo, _hi = DEFAULT_PARAM_RANGES["rin"]
    assert lo > 0


def test_tempgrad_evaluates_with_default_rin(registry):
    """rin=0 used to crash getComplexCoherentFlux with oimodeler's default
    logarithmic radial grid ("Logarithmic grid requires rin > 0."); the
    component's own default init value (DEFAULT_PARAM_INIT) must stay
    clear of that."""
    cfg = ComponentConfig(component_type="oimTempGrad", registry=registry, name="tg")
    instance = cfg.create_instance(oim)
    model = oim.oimModel(instance)
    # A single point evaluation is enough to exercise the radial-grid
    # construction that raised for rin<=0.
    model.getComplexCoherentFlux([1e6], [1e6], [2.2e-6])
