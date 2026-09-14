# tests/conftest.py
"""Make the repository root importable regardless of the CWD pytest is run from."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
