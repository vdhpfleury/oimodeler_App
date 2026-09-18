# tests/test_filter_registry.py
"""
Regression test for a crash reported in production:

    IndexError: list index out of range
    at oimodeler/oimUtils.py: vali = values[dataTypes.index(dataTypei)]

oimSetMinErrFilter's `values`/`relThreshold` must be either a scalar
(broadcast to every data type) or a list of EXACTLY the same length as
`dataType` — a shorter list raises this IndexError as soon as the filter
is actually applied to data, not at filter-construction time. The Data
page's filter form used to always collect a single scalar `values` while
letting `dataType` be a multiselect (multiple types), so selecting more
than one data type crashed the whole page the moment the filter list was
evaluated.

Fixed in core/filter_registry.py (dataType now declared before values/
relThreshold) and pages/data.py's _render_param() ("per_datatype_float"/
"per_datatype_optional_float": one widget per selected data type,
emitting a list of exactly that length).

This test guards the oimodeler-side contract that fix depends on — it
doesn't exercise the Streamlit form itself (no Streamlit session in
pytest), but proves a mismatched-length list crashes and a
matching-length list doesn't, against the real installed oimodeler.
"""
from __future__ import annotations

from pathlib import Path

import pytest

oim = pytest.importorskip("oimodeler")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "tutorial" / "Data" / "RealData" / "MATISSE" / "HD179218" / \
    "OiXP_HD179218_MATISSE_A0-B2-C1-D0_2019-03-24.fits"


def test_mismatched_length_values_crashes_documenting_the_hazard():
    """Documents the real oimodeler hazard core/filter_registry.py and
    pages/data.py's per-datatype widgets must avoid ever reproducing."""
    data = oim.oimData(str(DATA_FILE))
    f = oim.oimSetMinErrFilter(
        targets="all", arr="all",
        dataType=["VIS2DATA", "T3PHI"], values=5.0,
    )
    with pytest.raises(IndexError):
        data.setFilter(oim.oimDataFilter([f]))


def test_one_value_per_datatype_applies_cleanly():
    """The fix: always emit one values/relThreshold entry per selected
    data type, matching length exactly — the same shape
    pages/data.py's per_datatype_float widgets now produce."""
    data = oim.oimData(str(DATA_FILE))
    f = oim.oimSetMinErrFilter(
        targets="all", arr="all",
        dataType=["VIS2DATA", "T3PHI"], values=[5.0, 2.0],
        relThreshold=[None, None],
    )
    data.setFilter(oim.oimDataFilter([f]))  # must not raise


def test_single_datatype_still_works():
    """A single selected data type (the common case) still produces a
    length-1 list, not a bare scalar — also valid per oimodeler's own
    contract ('scalar or list of the same length as dataType')."""
    data = oim.oimData(str(DATA_FILE))
    f = oim.oimSetMinErrFilter(
        targets="all", arr="all",
        dataType=["VIS2DATA"], values=[5.0], relThreshold=[None],
    )
    data.setFilter(oim.oimDataFilter([f]))  # must not raise
