#!/usr/bin/env bash
# Relaunch OIModeler App without re-running the full installer (no Python/Git
# check, no download, no dependency install). This is what the desktop
# shortcut created by install.sh points to; safe to run directly too.
set -e
cd "$(dirname "$0")/.."

if [ ! -d "env_oim" ]; then
  echo "No environment found here. Run installer/install.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source env_oim/bin/activate
exec streamlit run app.py
