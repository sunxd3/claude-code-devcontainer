#!/bin/bash
# Warn if experiment Python file doesn't import bayes_kit

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

[[ ! "$file_path" =~ \.py$ ]] && exit 0
[[ ! "$file_path" =~ experiments/ ]] && exit 0
[[ ! -f "$file_path" ]] && exit 0

if ! grep -q "from bayes_kit import\|import bayes_kit" "$file_path"; then
    echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"<system-reminder>Not importing from bayes_kit. Use shared utilities instead of rewriting common patterns.</system-reminder>\"}}"
fi

exit 0
