#!/bin/bash
# Launch the AdaptiveKV web UI (plain HTML, no Streamlit)
# Uses .venv/bin/python directly, so it works even though the venv was created at another path.
cd "$(dirname "$0")"
.venv/bin/python web/server.py --port "${1:-8600}"
