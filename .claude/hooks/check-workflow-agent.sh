#!/bin/bash
# Stop hook: Intelligent audit of workflow completeness via Claude CLI
# Checks if EDA recommendations and proposed hypotheses were adequately addressed
# Only runs if workflow_mode is enabled in agent-config.json
# Exit 0 + JSON stdout to block, Exit 0 without block to allow

cd "$CLAUDE_PROJECT_DIR"

# Check if workflow mode is enabled in config
CONFIG_FILE=".claude/agent-config.json"
WORKFLOW_MODE=$(jq -r '.workflow_mode // false' "$CONFIG_FILE" 2>/dev/null)
if [[ "$WORKFLOW_MODE" != "true" ]]; then
    exit 0
fi

# Read hook input
INPUT=$(cat)
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')

# Prevent infinite loops
if [[ "$STOP_HOOK_ACTIVE" == "true" ]]; then
    exit 0
fi

# Check if key artifacts exist first (fast fail - let programmatic hook handle missing files)
[[ ! -f "analysis/eda/eda_report.md" ]] && exit 0
[[ ! -f "analysis/experiments/experiment_plan.md" ]] && exit 0

# JSON schema for Claude CLI structured output
SCHEMA='{
  "type": "object",
  "properties": {
    "complete": {
      "type": "boolean",
      "description": "true if all EDA recommendations and hypotheses were adequately addressed"
    },
    "gaps": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of gaps - EDA recommendations or hypotheses NOT addressed by validated models"
    },
    "summary": {
      "type": "string",
      "description": "Brief summary of coverage assessment"
    }
  },
  "required": ["complete", "gaps", "summary"]
}'

# Audit prompt
PROMPT='You are auditing whether a Bayesian modeling workflow adequately addressed its proposed hypotheses.

Your task:
1. Read analysis/eda/eda_report.md - extract modeling recommendations (response scales, likelihood families, variance structures, suggested approaches)
2. Read analysis/experiments/experiment_plan.md - extract proposed models and hypotheses
3. List experiments in analysis/experiments/ and check which have fit results (*/fit/*.csv files)
4. Cross-check: for each substantive EDA recommendation and proposed hypothesis, was at least one model validated?

Focus on STRUCTURALLY DIFFERENT approaches:
- Missing a different parameterization = minor
- Missing an entire response scale or likelihood family = major gap

Return your assessment as JSON. Your job is quality assurance, not obstruction.'

# Call Claude CLI with JSON schema output
# --disallowedTools "Task" prevents spawning subagents
CLI_RESULT=$(claude \
    --print \
    --dangerously-skip-permissions \
    --tools "Read,Glob,Grep" \
    --disallowedTools "Task" \
    --output-format json \
    --json-schema "$SCHEMA" \
    -p "$PROMPT" 2>/dev/null)

# Parse the CLI JSON output - extract the result field
AUDIT_RESULT=$(echo "$CLI_RESULT" | jq -r '.result // .')

IS_COMPLETE=$(echo "$AUDIT_RESULT" | jq -r '.complete // false')
GAPS=$(echo "$AUDIT_RESULT" | jq -r '.gaps // []')
SUMMARY=$(echo "$AUDIT_RESULT" | jq -r '.summary // "Audit failed"')

if [[ "$IS_COMPLETE" == "true" ]]; then
    # Allow stopping
    exit 0
else
    # Block stopping with structured JSON output
    GAPS_TEXT=$(echo "$GAPS" | jq -r 'join("; ")')
    REASON="Continue: Address these gaps before completing - $GAPS_TEXT"

    # Output JSON to stdout to block
    jq -n --arg reason "$REASON" '{"decision": "block", "reason": $reason}'
    exit 0
fi
