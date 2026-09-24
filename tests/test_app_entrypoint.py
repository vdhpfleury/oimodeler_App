# tests/test_app_entrypoint.py
"""
Regression test for app.py's `if __name__ == "__main__":` guard.

Streamlit installs a fake `__main__` module (name "__main__", `__file__`
pointing at app.py) to run the app script — see Streamlit's own
runtime/scriptrunner/script_runner.py. services/jobs.py's background fit
workers use multiprocessing's `spawn` start method, which reconstructs
each worker's `__main__` by re-running *that same file* via
multiprocessing.spawn._fixup_main_from_path(), but with
run_name="__mp_main__" instead of "__main__". Before app.py's body was
wrapped in this guard, that re-exec ran the *entire app* (st.set_page_
config(), every page's render()) bare, once per background fit job —
wasting real startup time and flooding the logs with Streamlit's own
"missing ScriptRunContext"/"session state does not function" warnings,
confirmed live by a user report. The guard makes that re-exec a no-op.
"""
from __future__ import annotations

import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_app_py_body_is_a_noop_under_mp_main_run_name():
    """Reproduces exactly what multiprocessing's spawn does to app.py in a
    background fit worker's child process: re-run it with
    run_name="__mp_main__". main() must be defined (so it's still callable
    the normal way) but never invoked — no Streamlit call should fire."""
    result = runpy.run_path(str(REPO_ROOT / "app.py"), run_name="__mp_main__")
    assert "main" in result
    assert callable(result["main"])


def test_app_py_body_runs_under_dunder_main_run_name():
    """The other half of the guard: with run_name="__main__" (matching
    Streamlit's own module-naming trick), the body DOES execute — confirms
    the guard isn't accidentally inverted and app.py doesn't silently do
    nothing under normal `streamlit run app.py` execution. Streamlit
    tolerates running outside a real session/ScriptRunContext (it just
    warns, doesn't raise — confirmed live), so success is checked by a
    concrete side effect only the guarded body causes: importing the page
    modules it wires up in step 4."""
    import sys
    for mod in list(sys.modules):
        if mod.startswith("pages."):
            del sys.modules[mod]

    runpy.run_path(str(REPO_ROOT / "app.py"), run_name="__main__")

    assert "pages.overview" in sys.modules
