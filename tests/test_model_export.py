# tests/test_model_export.py
"""
Regression tests for core/model_export.py.

Covers two independent export paths that must both round-trip through
core.csv_import.parse_csv_to_model():
- model_to_txt(): the app's own export of a saved session model.
- EXTERNAL_WRITER_SNIPPET: the standalone function shown to users to copy
  into their own oimodeler scripts. Root cause covered: oimParam.min/max
  are numpy scalars whose repr() (numpy>=2.0) prints "np.float16(-inf)"
  instead of a plain "-inf" — silently unparseable and undetected unless
  actually executed and read back.

These tests require the real installed `oimodeler` package (it isn't
pinned - see the project skill) and are skipped if it isn't available.
"""
from __future__ import annotations

import ast
import io
import math

import pandas as pd
import pytest

oim = pytest.importorskip("oimodeler")

from core.registry import build_registry
from core.csv_import import parse_csv_to_model
from core.model_export import model_to_txt, EXTERNAL_WRITER_SNIPPET


@pytest.fixture(scope="module")
def registry():
    return build_registry(oim)


def test_external_writer_snippet_is_valid_standalone_python():
    """Must parse on its own — it's copy-pasted into a user's script, not
    imported, so a syntax error here would only surface for them."""
    ast.parse(EXTERNAL_WRITER_SNIPPET)


def test_model_to_txt_round_trips(registry):
    model_dict = {
        "components": [
            {
                "type": "oimUD", "name": "c1_UD",
                "initial_values": {"x": 0., "y": 0., "f": 1., "d": 2.0},
                "param_ranges": {"x": (-5., 5.), "y": (-5., 5.),
                                  "f": (0., 1.), "d": (0.5, 5.0)},
                "free_params": ["f", "d"], "interpolators": {},
            },
            {
                "type": "oimBackground", "name": "c2_Bg",
                "initial_values": {"f": 0.1},
                "param_ranges": {"f": (0., 1.)},
                "free_params": [], "interpolators": {},
            },
        ],
    }
    txt = model_to_txt(model_dict, registry)
    assert "\t" in txt, "must be tab-separated, not comma"

    df = pd.read_csv(io.StringIO(txt), sep="\t")
    result, err = parse_csv_to_model(df, registry)
    assert result is not None, err

    comps = {c["type"]: c for c in result["components"]}
    assert set(comps) == {"oimUD", "oimBackground"}
    assert set(comps["oimUD"]["free_params"]) == {"f", "d"}
    assert math.isclose(comps["oimUD"]["param_ranges"]["d"][0], 0.5)
    assert math.isclose(comps["oimUD"]["param_ranges"]["d"][1], 5.0)


def test_external_writer_snippet_round_trips(tmp_path, registry):
    """Executes the exact code shown to users, against a real oimodeler
    model, and confirms the file it writes is readable by the app."""
    c1 = oim.oimUD(x=0., y=0., f=1., d=2.0)
    c2 = oim.oimBackground(f=0.1)
    model = oim.oimModel(c1, c2)
    model.getParameters()["c1_UD_d"].set(min=0.5, max=5.0, free=True)
    model.getParameters()["c1_UD_f"].set(min=0., max=1., free=True)

    ns = {"model": model}
    exec(EXTERNAL_WRITER_SNIPPET, ns)
    write_model_to_txt = ns["write_model_to_txt"]

    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        write_model_to_txt("external_test_model")
        content = (tmp_path / "external_test_model.txt").read_text()
    finally:
        os.chdir(cwd)

    assert "np.float16" not in content, "numpy scalar repr leaked into the file"

    df = pd.read_csv(tmp_path / "external_test_model.txt", sep=None, engine="python")
    result, err = parse_csv_to_model(df, registry)
    assert result is not None, err

    comps = {c["type"]: c for c in result["components"]}
    ud = comps["oimUD"]
    assert set(ud["free_params"]) == {"f", "d"}
    assert ud["param_ranges"]["x"] == (float("-inf"), float("inf"))
    assert math.isclose(ud["param_ranges"]["d"][0], 0.5)
    assert math.isclose(ud["param_ranges"]["d"][1], 5.0)
    assert math.isclose(ud["initial_values"]["d"], 2.0)
