#!/bin/bash
# Stop hook: Programmatic check for Bayesian workflow deliverables
# Only runs if workflow_mode is enabled in agent-config.json
# Exit 0 = allow stop, Exit 2 = block and send message to Claude

cd "$CLAUDE_PROJECT_DIR"

# Check if workflow mode is enabled in config
CONFIG_FILE=".claude/agent-config.json"
WORKFLOW_MODE=$(jq -r '.workflow_mode // false' "$CONFIG_FILE" 2>/dev/null)
if [[ "$WORKFLOW_MODE" != "true" ]]; then
    exit 0
fi

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
    exit 2
fi

# All checks passed
exit 0
