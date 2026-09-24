#!/usr/bin/env python3
"""Standalone installation-health check for OIModeler App.

Run this after ``pip install -r requirements.txt`` and before
``streamlit run app.py`` to confirm the environment is set up correctly:

    python doctor.py

It checks the Python version, pip, whether a virtual environment is
active, that every package the app needs at runtime is importable, and
that the expected project files are present. It exits with status 0 when
everything looks fine and 1 otherwise, so it can also be used in scripts.
"""
import importlib
import sys
from pathlib import Path

MIN_PYTHON = (3, 9)

# name shown to the user -> module name to import (only differs for oimodeler's
# case since the PyPI/GitHub project and the import name happen to match here)
REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "astropy",
    "pandas",
    "matplotlib",
    "streamlit",
    "oimodeler",
]

REQUIRED_PATHS = [
    "app.py",
    "requirements.txt",
    "core",
    "pages",
    "services",
    "config",
]

OK = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"


def check_python_version():
    version = sys.version_info
    label = f"{version.major}.{version.minor}.{version.micro}"
    if version[:2] >= MIN_PYTHON:
        print(f"{OK} Python {label}")
        return True
    print(f"{FAIL} Python {label} (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or higher)")
    return False


def check_pip():
    try:
        import pip
    except ImportError:
        print(f"{FAIL} pip is not available for this Python interpreter")
        print("       See: https://pip.pypa.io/en/stable/installation/")
        return False
    print(f"{OK} pip {pip.__version__}")
    return True


def check_virtualenv():
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        print(f"{OK} Virtual environment active ({sys.prefix})")
    else:
        print(f"{WARN} No virtual environment detected")
        print("       Recommended: create one with 'python3 -m venv env_oim' to avoid")
        print("       dependency conflicts with other Python projects on this machine.")
    return True


def check_packages():
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
        except ImportError as exc:
            print(f"{FAIL} {package} — not installed ({exc})")
            all_ok = False
            continue
        version = getattr(module, "__version__", "unknown version")
        print(f"{OK} {package} {version}")
    return all_ok


def check_project_files(root: Path):
    all_ok = True
    for name in REQUIRED_PATHS:
        if (root / name).exists():
            print(f"{OK} {name}")
        else:
            print(f"{FAIL} {name} — missing (run doctor.py from the project's root folder)")
            all_ok = False
    return all_ok


def main():
    root = Path(__file__).resolve().parent

    print("OIModeler App - Installation check")
    print("===================================\n")

    print("-- Python & pip --")
    python_ok = check_python_version()
    pip_ok = check_pip()
    check_virtualenv()

    print("\n-- Project files --")
    files_ok = check_project_files(root)

    print("\n-- Dependencies --")
    packages_ok = check_packages()

    print()
    if python_ok and pip_ok and files_ok and packages_ok:
        print("All checks passed. You can now run:\n\n    streamlit run app.py\n")
        return 0

    print("Some checks failed — see the [FAIL] lines above.")
    print("For help fixing them, see the Troubleshooting section in README.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
