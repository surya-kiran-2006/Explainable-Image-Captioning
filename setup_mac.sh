#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is intended for macOS."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Warning: this does not look like a native Apple Silicon shell (uname -m: $(uname -m))."
  echo "Open Terminal natively, not through Rosetta, for MPS acceleration."
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 was not found. Install it first (recommended: brew install python@3.11)."
  exit 1
fi

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.11.0 is used as a known-compatible MPS baseline for this project.
# Apple documents MPS support for Apple Silicon and macOS 14+ for this release.
python -m pip install "torch==2.11.0"
python -m pip install -r requirements-mac.txt

if command -v brew >/dev/null 2>&1; then
  if ! command -v tesseract >/dev/null 2>&1; then
    echo "Installing Tesseract OCR with Homebrew..."
    brew install tesseract
  fi
else
  echo "Homebrew not found. Install Homebrew, then run: brew install tesseract"
fi

python check_mac.py

echo
echo "Setup complete. Next: place your checkpoint under models/blip-flickr8k-final and run ./run_mac.sh"
