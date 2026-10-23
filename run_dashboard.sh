#!/bin/bash
# Launch AdaptiveKV dashboard
# Reads ONLY from results/ — no numbers in code
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting dashboard at http://localhost:8501"
streamlit run dashboard/app.py --server.port 8501
