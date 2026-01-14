#!/bin/bash
# Stop hook: Check if the Bayesian modeling workflow is complete
# Blocks stopping if required deliverables are missing

cd "$CLAUDE_PROJECT_DIR"

MISSING=""

# Phase 1: Check EDA
if [[ ! -f "analysis/eda/eda_report.md" ]]; then
  MISSING="EDA report (analysis/eda/eda_report.md) not found. "
fi

# Phase 2: Check experiment plan
if [[ ! -f "analysis/experiments/experiment_plan.md" ]]; then
  MISSING="${MISSING}Experiment plan (analysis/experiments/experiment_plan.md) not found. "
fi

# Phase 3: Check for at least one model fit
# Look for fit results (CSV files in fit/ directories)
FIT_RESULTS=$(find analysis/experiments -path "*/fit/*.csv" -type f 2>/dev/null | head -1)
if [[ -z "$FIT_RESULTS" ]]; then
  MISSING="${MISSING}No model fits found (expected CSV files in analysis/experiments/*/fit/). "
fi

# Phase 4: Check final report
if [[ ! -f "analysis/final_report.md" ]]; then
  MISSING="${MISSING}Final report (analysis/final_report.md) not found."
fi

# If anything is missing, block stopping
if [[ -n "$MISSING" ]]; then
  echo "Workflow incomplete: $MISSING" >&2
  exit 2  # Exit code 2 blocks stopping and sends stderr to Claude
fi

# All checks passed
exit 0
