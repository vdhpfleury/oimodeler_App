# core/oifits_meta.py
"""
Lightweight, read-only OIFITS header/table summary for the Data page.

Deliberately independent from oimodeler's oimData (which is heavier and
mutated by filters) — this only ever reads a few primary-header keywords
and the OI_ARRAY/OI_WAVELENGTH tables to build a short, human-readable
summary of a file (instrument, target, date, VLTI configuration, native
spectral coverage), plus the raw wavelength bounds used to prefill a
per-file filter's default range.

Pure logic, no Streamlit import — testable in isolation.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits


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
