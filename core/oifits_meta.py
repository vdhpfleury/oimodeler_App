# core/oifits_meta.py
"""
Lightweight, read-only OIFITS header/table introspection for the Data
page's filter workbench.

Deliberately independent from oimodeler's oimData (which is heavier and
mutated by filters) — this only ever reads FITS headers/tables to build:
- a per-row metadata table (extract_filter_metadata) used to populate the
  filter form's cascading widgets (which arrays/instruments/data types/
  baselines/telescopes/wavelength range actually exist in the current
  selection), and
- get_available_values(), a pure pandas summary of that table restricted
  to whichever files/arrays are currently targeted by the filter form.

Pure logic, no Streamlit import — testable in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

_OBSERVABLE_TABLES = ("OI_VIS", "OI_VIS2", "OI_T3", "OI_FLUX")
_TABLE_DATA_TYPES = {
    "OI_VIS":  ["VISAMP", "VISPHI"],
    "OI_VIS2": ["VIS2DATA"],
    "OI_T3":   ["T3AMP", "T3PHI"],
    "OI_FLUX": ["FLUXDATA"],
}


def clean_string(value) -> str:
    """Convert a FITS string/bytes value to a clean Python str."""
    if isinstance(value, bytes):
        return value.decode(errors="ignore").strip()
    return str(value).strip()


def _baseline_name(sta_indices, telescopes: dict[int, str]) -> tuple[str, str]:
    """Convert STA_INDEX values into a baseline (2 stations) or closure
    triangle (3 stations) name, plus a comma-joined telescope list."""
    sta_indices = np.asarray(sta_indices).flatten()
    if len(sta_indices) not in (2, 3):
        return "N/A", "N/A"
    names = [telescopes.get(int(i), str(int(i))) for i in sta_indices]
    return "-".join(names), ", ".join(names)


def read_file_summary(filepath: str) -> dict:
    """Read a short summary of an OIFITS file.

    Never raises: any missing/unreadable field is returned as None so a
    partial summary can still be displayed rather than losing the whole
    file's block over one missing header keyword.

    Returns
    -------
    dict with keys: instrument, target, date_obs, vlti_config,
    wl_min_um, wl_max_um, n_wl (all None if unavailable/unreadable).
    """
    summary = {
        "instrument":  None,
        "target":      None,
        "date_obs":    None,
        "vlti_config": None,
        "wl_min_um":   None,
        "wl_max_um":   None,
        "n_wl":        None,
    }
    try:
        with fits.open(filepath, memmap=False) as hdul:
            header = hdul[0].header
            summary["instrument"] = header.get("INSTRUME")
            summary["target"]     = header.get("OBJECT")
            summary["date_obs"]   = header.get("DATE-OBS")

            stations = []
            wl_um = None
            for hdu in hdul:
                if hdu.name == "OI_ARRAY" and "STA_NAME" in getattr(hdu, "columns", []).names:
                    stations = [str(s).strip() for s in hdu.data["STA_NAME"]]
                elif hdu.name == "OI_WAVELENGTH" and "EFF_WAVE" in getattr(hdu, "columns", []).names:
                    wl = np.asarray(hdu.data["EFF_WAVE"], dtype=float)
                    wl_um = wl if wl_um is None else np.concatenate([wl_um, wl])

            if stations:
                summary["vlti_config"] = "-".join(stations)
            if wl_um is not None and wl_um.size:
                summary["wl_min_um"] = float(wl_um.min()) * 1e6
                summary["wl_max_um"] = float(wl_um.max()) * 1e6
                summary["n_wl"]      = int(np.unique(wl_um).size)
    except Exception:
        # A malformed/partial header must never break the Data page — the
        # caller shows whatever fields came back (possibly all None).
        pass

    return summary


def extract_filter_metadata(filepath: str, display_name: str) -> pd.DataFrame:
    """Read one OIFITS file and return a metadata DataFrame with one row
    per (target, array, datatype, baseline) combination it contains.

    `display_name` is the name shown to / selected by the user (the
    original uploaded filename, i.e. the key already used elsewhere as
    st.session_state.loaded_files's key) — independent of `filepath`,
    which is the sanitized on-disk path (services/storage.py).

    Used only to populate the filter form's cascading widgets — never to
    build a filter itself, so a partially-unreadable file degrades to an
    empty/partial table rather than raising.
    """
    rows: list[dict] = []

    with fits.open(filepath, memmap=False) as hdul:
        targets: dict[int, str] = {}
        if "OI_TARGET" in hdul and hdul["OI_TARGET"].data is not None:
            for row in hdul["OI_TARGET"].data:
                targets[int(row["TARGET_ID"])] = clean_string(row["TARGET"])

        telescopes: dict[int, str] = {}
        if "OI_ARRAY" in hdul and hdul["OI_ARRAY"].data is not None:
            for row in hdul["OI_ARRAY"].data:
                telescopes[int(row["STA_INDEX"])] = clean_string(row["STA_NAME"])

        wavelengths: dict[str, dict] = {}
        for hdu in hdul:
            if not hdu.name.startswith("OI_WAVELENGTH") or hdu.data is None:
                continue
            if "EFF_WAVE" not in hdu.data.names:
                continue
            insname = clean_string(hdu.header.get("INSNAME", "UNKNOWN"))
            wl = np.asarray(hdu.data["EFF_WAVE"], dtype=float)
            wl = wl[np.isfinite(wl)]
            if wl.size == 0:
                continue
            wavelengths[insname] = {
                "wl_min_um": float(wl.min()) * 1e6,
                "wl_max_um": float(wl.max()) * 1e6,
                "n_channels": int(wl.size),
            }

        for hdu in hdul:
            if hdu.name not in _OBSERVABLE_TABLES or hdu.data is None:
                continue

            table = hdu.data
            array_name = hdu.name
            insname = clean_string(hdu.header.get("INSNAME", "UNKNOWN"))
            wl_info = wavelengths.get(
                insname, {"wl_min_um": np.nan, "wl_max_um": np.nan, "n_channels": np.nan},
            )
            # Best-effort guess at which recognized columns (VIS2DATA,
            # FLUXDATA, ...) this table carries — real files sometimes
            # don't match (e.g. a GRAVITY reduction naming its flux
            # column differently). That must never hide the ARRAY itself
            # from the filter form: oimRemoveArrayFilter/oimRemoveInsnameFilter
            # and the "Arrays / Tables" target widget operate on the table
            # name, not on a recognized column inside it — see the `else`
            # branch below, which still emits one row per baseline/target
            # (with datatype=None, dropped by get_available_values()'s
            # dataTypes list but NOT by its arrays list).
            data_types = [dt for dt in _TABLE_DATA_TYPES.get(array_name, [])
                          if dt in table.names]

            for row_number, data_row in enumerate(table):
                if "TARGET_ID" in table.names:
                    target_id = int(data_row["TARGET_ID"])
                    target_name = targets.get(target_id, f"TARGET_ID={target_id}")
                else:
                    target_id, target_name = None, "UNKNOWN"

                if "STA_INDEX" in table.names:
                    baseline, telescope_list = _baseline_name(
                        data_row["STA_INDEX"], telescopes,
                    )
                else:
                    baseline, telescope_list = "N/A", "N/A"

                base_row = {
                    "file":        display_name,
                    "target_id":   target_id,
                    "target":      target_name,
                    "insname":     insname,
                    "array":       array_name,
                    "baseline":    baseline,
                    "telescopes":  telescope_list,
                    "wl_min_um":   wl_info["wl_min_um"],
                    "wl_max_um":   wl_info["wl_max_um"],
                    "n_channels":  wl_info["n_channels"],
                    "row":         row_number,
                }
                if data_types:
                    for datatype in data_types:
                        rows.append({**base_row, "datatype": datatype})
                else:
                    rows.append({**base_row, "datatype": None})

    return pd.DataFrame(rows)


def restrict_to_targets(metadata: pd.DataFrame, file_names: list[str],
                        target_indices: list[int] | None) -> pd.DataFrame:
    """Restrict the metadata table to the files selected via a filter's
    "targets" parameter. `target_indices=None` means targets="all"."""
    if target_indices is None or metadata is None or metadata.empty:
        return metadata
    selected = {file_names[i] for i in target_indices if 0 <= i < len(file_names)}
    return metadata[metadata["file"].isin(selected)]


def get_available_values(
    metadata: pd.DataFrame, file_names: list[str],
    target_indices: list[int] | None, selected_arrays: list[str] | None = None,
) -> dict:
    """Summarize the (target-restricted) metadata table into the lists of
    values the filter form's widgets should offer. `selected_arrays`
    further restricts dataType options to the chosen array(s) — e.g.
    selecting OI_VIS2 only offers VIS2DATA, not T3AMP/T3PHI.
    """
    empty = {
        "arrays": [], "insnames": [], "dataTypes": [],
        "baselines": [], "telescopes": [], "wl_range": (None, None),
    }
    subset = restrict_to_targets(metadata, file_names, target_indices)
    if subset is None or subset.empty:
        return empty

    datatype_subset = (
        subset[subset["array"].isin(selected_arrays)] if selected_arrays else subset
    )

    baseline_values = subset["baseline"].dropna().unique()
    # Baselines have one "-" (two telescopes); closure triangles (OI_T3)
    # have two "-" and are excluded here — they aren't valid "baselines".
    baselines = sorted(b for b in baseline_values if b.count("-") == 1)

    telescope_values: set[str] = set()
    for entry in subset["telescopes"].dropna():
        for name in entry.split(","):
            name = name.strip()
            if name and name != "N/A":
                telescope_values.add(name)

    wl_min = subset["wl_min_um"].min()
    wl_max = subset["wl_max_um"].max()

    return {
        "arrays":     sorted(subset["array"].dropna().unique()),
        "insnames":   sorted(subset["insname"].dropna().unique()),
        "dataTypes":  sorted(datatype_subset["datatype"].dropna().unique()),
        "baselines":  baselines,
        "telescopes": sorted(telescope_values),
        "wl_range": (
            float(wl_min) if pd.notna(wl_min) else None,
            float(wl_max) if pd.notna(wl_max) else None,
        ),
    }


def summarize_oimdata(data, file_names: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """Build one small table per file: one row per observable array
    (OI_VIS2, OI_T3, OI_VIS, OI_FLUX) present in the CURRENT `data` (i.e.
    already filtered, if a filter is active — oimodeler applies its
    filter in place once `data.useFilter` is True), with baseline/
    triangle count, wavelength-channel count, instrument name, and the
    data types actually present and non-zeroed.

    A data type entirely zeroed by a filter (e.g. oimDataTypeFilter) is
    treated as removed — that matches what oimodeler itself considers
    "emptied" by a filter, even though the column technically still exists.

    `data` is a live oimodeler oimData object — this function only reads
    it, never mutates it. Accepting it directly (rather than a path) is
    consistent with the rest of core/ (model_builder.py, fitting.py,
    results.py all take live oimodeler objects as arguments); it is `oim`
    itself — not this function — that must never be imported at module
    level here, and it isn't.
    """
    summaries = []

    for i, hdul in enumerate(data.data):
        file_label = file_names[i] if i < len(file_names) else f"file {i}"

        wavelengths: dict[str, int] = {}
        for hdu in hdul:
            if not hdu.name.startswith("OI_WAVELENGTH") or hdu.data is None:
                continue
            if "EFF_WAVE" not in hdu.data.names:
                continue
            insname = clean_string(hdu.header.get("INSNAME", "UNKNOWN"))
            wavelengths[insname] = len(hdu.data["EFF_WAVE"])

        rows = []
        for hdu in hdul:
            if hdu.name not in _OBSERVABLE_TABLES or hdu.data is None:
                continue

            table = hdu.data
            insname = clean_string(hdu.header.get("INSNAME", "UNKNOWN"))
            present_types = []
            for dt in _TABLE_DATA_TYPES.get(hdu.name, []):
                if dt not in table.names:
                    continue
                values = np.asarray(table[dt])
                if values.size and np.all(values == 0):
                    continue  # zeroed out by a filter — treated as removed
                present_types.append(dt)

            rows.append({
                "Array":      hdu.name,
                "nB":         len(table),
                "nλ":         wavelengths.get(insname, "?"),
                "INSNAME":    insname,
                "Data types": ", ".join(present_types) if present_types else "—",
            })

        summaries.append((file_label, pd.DataFrame(rows)))

    return summaries
