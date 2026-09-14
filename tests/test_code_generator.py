# tests/test_code_generator.py
"""
Regression tests for core/code_generator.py's generate_fitting_code().

Root cause covered: the generator used to build min/max/free bounds as three
flat, positionally-ordered lists (one entry per registry-declared param name)
and zip() them against the REAL oimodeler model's model.getParameters().keys().
That silently breaks (wrong bounds assigned to the wrong parameter, or a
configured-as-free parameter that's silently never set free at all) as soon
as a component has an interpolated parameter (oimInterp), because oimodeler
expands one interpolated parameter into several real keys
("..._interp1", "..._interp2", ...) that were never counted when the flat
lists were built. See docs of the fix for the concrete failure mode.

These tests require the real installed `oimodeler` package (it isn't pinned -
see the project skill) and are skipped if it isn't available.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

oim = pytest.importorskip("oimodeler")

from core.code_generator import generate_fitting_code
from core.model_builder import build_oim_model
from core.registry import build_registry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "tutorial", "Data", "RealData", "MATISSE", "HD179218")
DATA_FILE = "OiXP_HD179218_MATISSE_A0-B2-C1-D0_2019-03-24.fits"

FILTER_PARAMS = {"expr": "", "bin_L": 1, "bin_N": 1, "norm_L": False, "norm_N": False}

# A model with one plain component and one component that has an
# interpolated parameter ('f' on comp2, custom wl spline) — this is exactly
# the combination that triggers the historical bug.
MODEL_COMPS = [
    {
        "type": "oimUD",
        "name": "comp1_UD",
        "initial_values": {"x": 0.0, "y": 0.0, "f": 1.0, "d": 2.0},
        "param_ranges": {"x": (-5., 5.), "y": (-5., 5.), "f": (0.1, 0.9), "d": (0.5, 5.0)},
        "free_params": ["f", "d"],
        "interpolators": {},
    },
    {
        "type": "oimUD",
        "name": "comp2_UD_interp",
        "initial_values": {"x": 1.0, "y": 1.0, "f": 0.5, "d": 1.0},
        "param_ranges": {"x": (-5., 5.), "y": (-5., 5.), "f": (0., 1.), "d": (0.1, 3.0)},
        "free_params": ["d"],
        "interpolators": {
            "f": {
                "enabled": True, "type": "custom", "var": "wl",
                "wl": [3e-6, 4e-6, 5e-6], "values": [0.2, 0.5, 0.8],
            },
        },
    },
]


@pytest.fixture(scope="module")
def registry():
    return build_registry(oim)


def _extract_param_settings(code: str) -> dict:
    match = re.search(r"param_settings = \{.*?\n\}", code, re.S)
    assert match, "generated code must define a param_settings dict"
    return ast.literal_eval(match.group(0).split("=", 1)[1].strip())


def test_core_module_has_no_streamlit_dependency():
    """core/ must stay importable and callable without a Streamlit session."""
    import core.code_generator as cg
    source = ast.parse(open(cg.__file__).read())
    imported_names = {
        alias.name.split(".")[0]
        for node in ast.walk(source)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "streamlit" not in imported_names
    assert not hasattr(cg, "st")


def test_generated_code_does_not_use_positional_zip(registry):
    code = generate_fitting_code(
        method="chi2",
        result={"dtypes": ["VIS2DATA"]},
        data_filenames=[DATA_FILE],
        model_comps=MODEL_COMPS,
        filter_params=FILTER_PARAMS,
        registry=registry,
    )
    # The historical bug: zip(model.getParameters().keys(), min_value, max_value, free_status)
    assert "zip(model.getParameters()" not in code
    assert "\t" not in code, "generated code must use spaces, never a literal tab"
    assert 'os.path.join(path,' in code
    assert '= path + "' not in code


def test_param_settings_are_name_keyed_and_skip_interpolated_params(registry):
    """
    The generated bounds must be looked up by the exact key oimodeler's real
    oimModel.getParameters() uses ("c{i+1}_{shortname}_{param}"), must cover
    every genuinely scalar parameter, and must never target the individual
    sub-parameters of an interpolated parameter (there's no single UI-
    configured (min,max,free) triple that applies to those positionally).
    """
    code = generate_fitting_code(
        method="chi2",
        result={"dtypes": ["VIS2DATA"]},
        data_filenames=[DATA_FILE],
        model_comps=MODEL_COMPS,
        filter_params=FILTER_PARAMS,
        registry=registry,
    )
    settings = _extract_param_settings(code)

    # Ground truth: instantiate the real oimodeler model the same way the
    # live app does (core/model_builder.py) and inspect its real parameter keys.
    model = build_oim_model(oim, registry, MODEL_COMPS)
    real_keys = set(model.getParameters().keys())

    non_interp_keys = {k for k in real_keys if "_interp" not in k}
    interp_keys = real_keys - non_interp_keys

    assert interp_keys, "test setup sanity check: model should have interpolated sub-params"
    assert set(settings.keys()) == non_interp_keys, (
        f"generated param_settings keys {set(settings) ^ non_interp_keys} "
        "don't match the real model's scalar parameter keys"
    )
    assert not (set(settings.keys()) & interp_keys), (
        "generated param_settings must never target an interpolator's sub-parameters"
    )

    # The free/bounds values themselves must be the ones configured in MODEL_COMPS,
    # correctly attributed by name (not shifted by position).
    assert settings["c1_UD_d"] == (0.5, 5.0, True)
    assert settings["c1_UD_f"] == (0.1, 0.9, True)
    assert settings["c2_UD_d"] == (0.1, 3.0, True)
    assert settings["c2_UD_x"][2] is False  # x/y never marked free here


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_DIR, DATA_FILE)),
    reason="sample MATISSE dataset not available",
)
@pytest.mark.parametrize(
    "method, result",
    [
        ("chi2", {"dtypes": ["VIS2DATA"]}),
        ("emcee", {"dtypes": ["VIS2DATA"], "nwalkers": 16, "nsteps": 3, "init": "random"}),
    ],
)
def test_generated_script_actually_runs(tmp_path, registry, method, result):
    """
    The concrete acceptance test: generate a standalone script (chi2 and
    emcee paths, with a component that has an interpolated parameter) and
    execute it with `python`. It must exit 0 and must not print the
    "parameter ... not found on model" guard the generator emits if a name
    lookup ever fails again.
    """
    code = generate_fitting_code(
        method=method,
        result=result,
        data_filenames=[DATA_FILE],
        model_comps=MODEL_COMPS,
        filter_params=FILTER_PARAMS,
        registry=registry,
    )
    code = code.replace(
        "path = '[ABSOLUTE PATH TO OIFITS FOLDER - TO BE FILLED BY USER]'",
        f"path = {DATA_DIR!r}",
    )
    # Don't block the test runner on an interactive plot window.
    code = code.replace("plt.show()", "plt.close('all')")

    script_path = tmp_path / f"generated_{method}.py"
    script_path.write_text(code)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=300,
    )

    assert proc.returncode == 0, (
        f"generated {method} script failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "not found on model" not in proc.stdout
