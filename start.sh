#!/usr/bin/env bash
# Streamlit demo. The API is `python -m fundmatch serve`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m streamlit run app.py --server.port 8765 --server.address 0.0.0.0 --server.headless true
