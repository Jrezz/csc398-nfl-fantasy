#!/bin/bash
# Start the Streamlit dashboard.
# If pipeline hasn't been run yet, run it first.

cd "$(dirname "$0")"

if [ ! -f "results/metrics.json" ]; then
    echo "Running pipeline first..."
    .venv/bin/python run_pipeline.py
fi

.venv/bin/streamlit run dashboard/app.py
