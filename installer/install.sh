#!/usr/bin/env bash
# One-command installer for OIModeler App on Linux and macOS.
#
# Usage (from an empty directory — downloads the app for you):
#   curl -fsSL https://raw.githubusercontent.com/vdhpfleury/oimodeler_App/main/installer/install.sh | bash
#
# Usage (from an existing checkout — reuses it, no download):
#   bash installer/install.sh
#
# What it does: locates a supported Python (3.11-3.13), downloads the app
# if it isn't already present, creates an isolated virtual environment,
# installs every dependency, runs the doctor.py health check, then launches
# the app. See README.md's Compatibility/Troubleshooting sections for why
# each step exists.
set -euo pipefail

REPO_ARCHIVE_URL="https://github.com/vdhpfleury/oimodeler_App/archive/refs/heads/main.tar.gz"
SUPPORTED_VERSIONS=(3.11 3.12 3.13)
TOTAL_STEPS=6

step() {
  printf "\n[%s/%s] %s\n" "$1" "$TOTAL_STEPS" "$2"
}

ok() {
  printf "      \xe2\x9c\x93 %s\n" "$1"
}

fail() {
  printf "\n      \xc3\x97 %s\n" "$1" >&2
}

download_to_stdout() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$1"
  else
    fail "Neither curl nor wget is available to download the app."
    fail "Install one of them (e.g. 'sudo apt install curl') and try again."
    exit 1
  fi
}

find_python() {
  local v c ver
  for v in "${SUPPORTED_VERSIONS[@]}"; do
    if command -v "python$v" >/dev/null 2>&1; then
      echo "python$v"
      return 0
    fi
  done
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      ver=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
      for v in "${SUPPORTED_VERSIONS[@]}"; do
        if [ "$ver" = "$v" ]; then
          echo "$c"
          return 0
        fi
      done
    fi
  done
  return 1
}

echo "========================================"
echo "        OIModeler App Installer"
echo "========================================"

step 1 "Checking prerequisites (Python, Git)..."
if PYTHON_BIN=$(find_python); then
  PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  ok "Python $PY_VERSION ($PYTHON_BIN)"
else
  fail "No Python 3.11, 3.12 or 3.13 was found on your PATH."
  echo
  echo "      OIModeler App needs one of these versions specifically — see"
  echo "      the Compatibility section in README.md for why. Install one:"
  case "$(uname -s)" in
    Darwin) echo "        macOS:  brew install python@3.11" ;;
    Linux)  echo "        Linux:  sudo apt install python3.11 python3.11-venv" ;;
    *)      echo "        Windows: use installer/install.ps1 instead" ;;
  esac
  echo "        Any OS: https://www.python.org/downloads/"
  echo
  echo "      Then run this installer again."
  exit 1
fi

if command -v git >/dev/null 2>&1; then
  ok "Git $(git --version | grep -o '[0-9][0-9.]*' | head -1)"
else
  fail "Git was not found on your PATH."
  echo
  echo "      pip needs Git to install the oimodeler library from its GitHub"
  echo "      repository (a dependency of this app). Install it:"
  case "$(uname -s)" in
    Darwin) echo "        macOS:  brew install git  (or install Xcode Command Line Tools)" ;;
    Linux)  echo "        Linux:  sudo apt install git" ;;
    *)      echo "        Any OS: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git" ;;
  esac
  echo
  echo "      Then run this installer again."
  exit 1
fi

step 2 "Getting the application..."
if [ -f "app.py" ] && [ -f "requirements.txt" ]; then
  APP_DIR="$PWD"
  ok "Already in an OIModeler App checkout ($APP_DIR)"
else
  APP_DIR="$PWD/oimodeler_App"
  if [ -f "$APP_DIR/app.py" ]; then
    ok "Found an existing download at $APP_DIR"
  else
    mkdir -p "$APP_DIR"
    if ! download_to_stdout "$REPO_ARCHIVE_URL" | tar -xz -C "$APP_DIR" --strip-components=1; then
      fail "Could not download the application from GitHub."
      fail "Check your internet connection and try again."
      exit 1
    fi
    ok "Downloaded to $APP_DIR"
  fi
fi
cd "$APP_DIR"

step 3 "Creating an isolated environment..."
if [ ! -d "env_oim" ]; then
  "$PYTHON_BIN" -m venv env_oim
fi
# shellcheck disable=SC1091
source env_oim/bin/activate
ok "Environment ready ($APP_DIR/env_oim)"

step 4 "Installing dependencies (this can take a few minutes)..."
if pip install --upgrade pip -q && pip install -r requirements.txt -q; then
  ok "Dependencies installed"
else
  fail "Dependency installation failed — see the error above."
  fail "For common causes (wrong Python version, missing Git), see the"
  fail "Troubleshooting section in README.md."
  exit 1
fi

step 5 "Verifying the installation..."
if python doctor.py; then
  ok "Installation verified"
else
  fail "The health check found a problem — see the report above, and the"
  fail "Troubleshooting section in README.md."
  exit 1
fi

step 6 "Starting OIModeler App..."
echo
echo "========================================"
echo "Installation successful! Launching now."
echo "Press Ctrl+C to stop the app."
echo "========================================"
echo
exec streamlit run app.py
