# tests/test_normalization.py
"""
Regression tests for oim.oimParamNorm support (core/normalization.py),
integrated from the oimodeler example:

    pt2 = oim.oimPt()
    g2  = oim.oimGauss(f=oim.oimInterp("wl", wl=[3e-6, 4e-6], values=[1, 2]))
    m2  = oim.oimModel(g2, pt2)
    pt2.params["f"] = oim.oimParamNorm(g2.params["f"])

Covers: validation, build_oim_model()/random_search() applying the
normalization at the model level (it references OTHER components'
already-built parameter objects, unlike interpolators), and the fact that
oim.oimParamNorm has no .value/.min/.max/.error of its own — confirmed
live to crash core/results.py's get_result_df() before that function was
hardened against it (only .free, always False, and __call__(wl, t)).

These tests require the real installed `oimodeler` package (it isn't
pinned - see the project skill) and are skipped if it isn't available.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

oim = pytest.importorskip("oimodeler")

from core.registry import build_registry
from core.normalization import validate_normalization_refs
from core.validation import InvalidInput
from core.model_builder import build_oim_model
from core.fitting import random_search
from core.component import ComponentConfig
from core.results import get_result_df
from core.code_generator import generate_fitting_code
from core.model_export import model_to_txt, NORMALIZATION_SECTION_MARKER
from core.csv_import import (
    parse_csv_to_model, split_normalization_section, parse_normalization_section,
)


@pytest.fixture(scope="module")
def registry():
    return build_registry(oim)


def _two_comp_model(norm=1.0, refs=None):
    """comp1 = oimUD (plain, f=0.3 fixed); comp2 = oimGauss with 'f'
    normalized against comp1's 'f' (norm - comp1.f)."""
    return [
        {
            "type": "oimUD", "name": "c1_UD",
            "initial_values": {"x": 0., "y": 0., "f": 0.3, "d": 2.0},
            "param_ranges": {"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.5, 5.0)},
            "free_params": [], "interpolators": {}, "normalizations": {},
        },
        {
            "type": "oimGauss", "name": "c2_Gauss",
            "initial_values": {"x": 0., "y": 0., "f": 0.5, "fwhm": 2.0},
            "param_ranges": {"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "fwhm": (0.1, 10.)},
            "free_params": ["fwhm"], "interpolators": {},
            "normalizations": {
                "f": {
                    "enabled": True, "norm": norm,
                    "refs": refs if refs is not None else [{"component": "c1_UD", "param": "f"}],
                },
            },
        },
    ]


# ── Validation ──────────────────────────────────────────────────────────

def test_validate_rejects_empty_refs():
    with pytest.raises(InvalidInput):
        validate_normalization_refs("c2", [], [{"name": "c1"}, {"name": "c2"}], {})


def test_validate_rejects_self_reference():
    comp_list = [{"name": "c1", "type": "oimUD"}, {"name": "c2", "type": "oimGauss"}]
    registry_stub = {"oimUD": {"params": ["f"]}, "oimGauss": {"params": ["f"]}}
    with pytest.raises(InvalidInput):
        validate_normalization_refs(
            "c2", [{"component": "c2", "param": "f"}], comp_list, registry_stub,
        )


def test_validate_rejects_unknown_component():
    comp_list = [{"name": "c1", "type": "oimUD"}, {"name": "c2", "type": "oimGauss"}]
    registry_stub = {"oimUD": {"params": ["f"]}, "oimGauss": {"params": ["f"]}}
    with pytest.raises(InvalidInput):
        validate_normalization_refs(
            "c2", [{"component": "ghost", "param": "f"}], comp_list, registry_stub,
        )


def test_validate_rejects_unknown_param():
    comp_list = [{"name": "c1", "type": "oimUD"}, {"name": "c2", "type": "oimGauss"}]
    registry_stub = {"oimUD": {"params": ["f", "d"]}, "oimGauss": {"params": ["f"]}}
    with pytest.raises(InvalidInput):
        validate_normalization_refs(
            "c2", [{"component": "c1", "param": "nonexistent"}], comp_list, registry_stub,
        )


def test_validate_accepts_valid_ref():
    comp_list = [{"name": "c1", "type": "oimUD"}, {"name": "c2", "type": "oimGauss"}]
    registry_stub = {"oimUD": {"params": ["f", "d"]}, "oimGauss": {"params": ["f"]}}
    validate_normalization_refs(  # must not raise
        "c2", [{"component": "c1", "param": "f"}], comp_list, registry_stub,
    )


# ── build_oim_model() applies the normalization ────────────────────────

def test_build_oim_model_applies_normalization(registry):
    model = build_oim_model(oim, registry, _two_comp_model(norm=1.0))
    assert model is not None

    g2 = model.components[1]
    f_param = g2.params["f"]
    assert isinstance(f_param, oim.oimParamNorm)
    # norm=1.0 - c1_UD's f (0.3) == 0.7
    assert math.isclose(float(f_param()), 0.7, abs_tol=1e-9)
    assert f_param.free is False


def test_build_oim_model_stale_ref_raises(registry):
    comps = _two_comp_model(refs=[{"component": "does_not_exist", "param": "f"}])
    with pytest.raises(ValueError):
        build_oim_model(oim, registry, comps)


def test_random_search_applies_normalization(registry):
    """random_search() builds instances independently of build_oim_model()
    (core/fitting.py) — it must apply normalizations too, or a model with
    one would silently fit with the raw, un-normalized parameter."""
    comps = _two_comp_model(norm=1.0)
    configs = [
        ComponentConfig(
            component_type=c["type"], registry=registry, name=c["name"],
            initial_values=c["initial_values"], param_ranges=c["param_ranges"],
            free_params=c["free_params"], interpolators=c.get("interpolators", {}),
            normalizations=c.get("normalizations", {}),
        )
        for c in comps
    ]
    captured = {}
    orig_oimModel = oim.oimModel

    def _spy(*args, **kwargs):
        m = orig_oimModel(*args, **kwargs)
        if "model" not in captured:
            captured["model"] = m
        return m

    oim.oimModel = _spy
    try:
        random_search(oim, data=None, component_configs=configs, n_runs=1, seed=1)
    except Exception:
        pass  # sim.compute(data=None) is expected to fail — only the model matters
    finally:
        oim.oimModel = orig_oimModel

    assert "model" in captured
    f_param = captured["model"].components[1].params["f"]
    assert isinstance(f_param, oim.oimParamNorm)


# ── core/results.py hardening ───────────────────────────────────────────

def test_get_result_df_does_not_crash_on_normalized_param(registry):
    model = build_oim_model(oim, registry, _two_comp_model(norm=1.0))
    chi2r, df = get_result_df(model, is_fit=False)
    assert chi2r is None  # get_result_df() only fills chi2r for is_fit=True

    row = df[df["Parameter"] == "c2_GD_f"]
    assert len(row) == 1
    assert math.isclose(row.iloc[0]["Value"], 0.7, abs_tol=1e-9)
    assert row.iloc[0]["Free"] == False  # noqa: E712 — pandas bool, not `is False`
    assert pd.isna(row.iloc[0]["Min"])
    assert pd.isna(row.iloc[0]["Max"])


# ── core/code_generator.py ──────────────────────────────────────────────

def test_generated_code_emits_oimparamnorm_assignment(registry):
    code = generate_fitting_code(
        method="chi2",
        result={"dtypes": ["VIS2DATA"]},
        data_filenames=["dummy.fits"],
        model_comps=_two_comp_model(norm=1.0),
        applied_filters=[],
        registry=registry,
    )
    assert "oim.oimParamNorm" in code
    assert 'comp2.params[\'f\'] = oim.oimParamNorm([comp1.params[\'f\']], norm=1.0)' in code
    # oimParamNorm has no .set() — the normalized key must never appear in
    # param_settings (it would crash the generated script).
    assert "'c2_GD_f'" not in code.split("# ── 5.")[0].split("param_settings = {")[1].split("}")[0]


def test_generated_script_with_normalization_executes(tmp_path, registry):
    """Full acceptance test: build the reproducible script for a model
    with a normalization and actually run it with `python`."""
    import subprocess
    import sys
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = os.path.join(repo_root, "tutorial", "Data", "RealData", "MATISSE", "HD179218")
    data_file = "OiXP_HD179218_MATISSE_A0-B2-C1-D0_2019-03-24.fits"
    if not os.path.isfile(os.path.join(data_dir, data_file)):
        pytest.skip("sample MATISSE dataset not available")

    code = generate_fitting_code(
        method="chi2",
        result={"dtypes": ["VIS2DATA"]},
        data_filenames=[data_file],
        model_comps=_two_comp_model(norm=1.0),
        applied_filters=[],
        registry=registry,
    )
    code = code.replace(
        "path = '[ABSOLUTE PATH TO OIFITS FOLDER - TO BE FILLED BY USER]'",
        f"path = {data_dir!r}",
    )
    code = code.replace("plt.show()", "plt.close('all')")
    script_path = tmp_path / "generated_norm.py"
    script_path.write_text(code)

    proc = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"generated script failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "not found on model" not in proc.stdout


# ── TXT export/import round-trip ────────────────────────────────────────

def test_model_to_txt_round_trips_normalization(registry):
    model_dict = {"components": _two_comp_model(norm=1.0)}
    txt = model_to_txt(model_dict, registry)
    assert NORMALIZATION_SECTION_MARKER in txt
    flat_table = txt.split(NORMALIZATION_SECTION_MARKER)[0]
    assert not any(
        line.startswith("c2_GD_f\t") for line in flat_table.splitlines()
    ), "normalized param must not land in the flat table"

    rest_text, norm_text = split_normalization_section(txt)
    norm_rows = parse_normalization_section(norm_text)
    assert (2, "f") in norm_rows
    assert norm_rows[(2, "f")]["refs"] == [{"component": "c1_UD", "param": "f"}]

    import io
    import pandas as pd
    df = pd.read_csv(io.StringIO(rest_text), sep="\t")
    result, err, warns = parse_csv_to_model(df, registry, norm_rows=norm_rows)
    assert result is not None, err
    assert warns == []

    # parse_csv_to_model() always reassigns component names to
    # "c{idx}_{TypeAbbr}" (the shortname parsed from the Parameter
    # column), regardless of what they were called in model_dict above —
    # oimGauss's real shortname is "GD", not the free-form "Gauss" name
    # _two_comp_model() gave it.
    comps = {c["name"]: c for c in result["components"]}
    assert comps["c2_GD"]["normalizations"]["f"]["enabled"] is True
    assert comps["c2_GD"]["normalizations"]["f"]["refs"] == [
        {"component": "c1_UD", "param": "f"}
    ]

    # And the reconstructed dict must actually build a working model.
    model = build_oim_model(oim, registry, result["components"])
    f_param = model.components[1].params["f"]
    assert isinstance(f_param, oim.oimParamNorm)
