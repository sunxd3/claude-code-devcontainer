#!/bin/bash
# SessionStart hook: Install dependencies and set up environment
# Runs on startup and resume

set -e

cd "$CLAUDE_PROJECT_DIR"

# Install Python dependencies
echo "Installing dependencies..." >&2

# Install shared_utils first (it's a dependency of analysis)
if [[ -d "shared_utils" ]]; then
  cd shared_utils
  uv sync --quiet 2>/dev/null || uv sync
  cd "$CLAUDE_PROJECT_DIR"
fi

# Install analysis package
if [[ -d "analysis" ]]; then
  cd analysis
  uv sync --quiet 2>/dev/null || uv sync
  cd "$CLAUDE_PROJECT_DIR"
fi

# Set up environment variables for CmdStan if available
if [[ -n "$CLAUDE_ENV_FILE" ]]; then
  # Add common CmdStan paths
  if [[ -d "$HOME/.cmdstan" ]]; then
    CMDSTAN_DIR=$(ls -d "$HOME/.cmdstan/cmdstan-"* 2>/dev/null | head -1)
    if [[ -n "$CMDSTAN_DIR" ]]; then
      echo "export CMDSTAN=$CMDSTAN_DIR" >> "$CLAUDE_ENV_FILE"
      echo "export PATH=\"\$PATH:$CMDSTAN_DIR/bin\"" >> "$CLAUDE_ENV_FILE"
    fi
  fi
fi

# Build context message based on what data is available
CONTEXT=""

# Check for data files in analysis/data/
if [[ -d "analysis/data" ]]; then
  DATA_FILES=$(ls analysis/data/*.{json,csv} 2>/dev/null | head -5 || true)
  if [[ -n "$DATA_FILES" ]]; then
    CONTEXT="Data files found in analysis/data/: $(echo $DATA_FILES | tr '\n' ' '). "
  fi
fi

# Check workflow state
if [[ -f "analysis/final_report.md" ]]; then
  CONTEXT="${CONTEXT}Final report exists - workflow may be complete. Review and verify."
elif [[ -f "analysis/experiments/experiment_plan.md" ]]; then
  CONTEXT="${CONTEXT}Experiment plan exists. Continue with model development phase."
elif [[ -f "analysis/eda/eda_report.md" ]]; then
  CONTEXT="${CONTEXT}EDA complete. Proceed to model design phase."
elif [[ -n "$DATA_FILES" ]]; then
  CONTEXT="${CONTEXT}Begin with EDA phase."
fi

# Output JSON with context for Claude
if [[ -n "$CONTEXT" ]]; then
  cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "$CONTEXT"
  }
}
EOF
fi

exit 0
