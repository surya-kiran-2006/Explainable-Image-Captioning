#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Run ./setup_mac.sh first."
  exit 1
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python app.py
