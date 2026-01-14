#!/bin/bash
# PreToolUse hook: Enforce using uv for all Python/pip commands
# Blocks bare python/pip and directs Claude to use uv run or uv pip

set -e

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only process Bash tool calls
if [[ "$TOOL_NAME" != "Bash" ]] || [[ -z "$COMMAND" ]]; then
  exit 0
fi

# Block bare python/python3 commands (not via uv run)
if echo "$COMMAND" | grep -qE '(^|&&|;|\|)\s*(python|python3)\s' && ! echo "$COMMAND" | grep -qE 'uv run\s+(python|python3)'; then
  echo "Use 'uv run python' instead of bare 'python' or 'python3'. All Python execution must go through uv." >&2
  exit 2
fi

# Block bare pip/pip3 commands (not via uv pip or uv run pip)
if echo "$COMMAND" | grep -qE '(^|&&|;|\|)\s*(pip|pip3)\s' && ! echo "$COMMAND" | grep -qE 'uv (pip|run (pip|pip3))'; then
  echo "Use 'uv pip' or 'uv run pip' instead of bare 'pip' or 'pip3'. Package management must go through uv." >&2
  exit 2
fi

# Block creating new venvs
if echo "$COMMAND" | grep -qE '(python3? -m venv|virtualenv)'; then
  echo "Do not create new virtual environments. Use 'uv run' which manages environments automatically." >&2
  exit 2
fi

exit 0
