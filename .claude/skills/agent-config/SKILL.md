---
name: agent-config
description: Manage agent configuration settings for workflow behavior, file retention, and other preferences
---

# Agent Configuration

Manage runtime settings that control agent behavior. Settings are stored in `.claude/agent-config.json`.

## Usage

**Read config:**
```bash
cat "$CLAUDE_PROJECT_DIR/.claude/agent-config.json" 2>/dev/null || echo "{}"
```

**Set a value:**
```bash
# Using jq to update/create config
CONFIG_FILE="$CLAUDE_PROJECT_DIR/.claude/agent-config.json"
CURRENT=$(cat "$CONFIG_FILE" 2>/dev/null || echo "{}")
echo "$CURRENT" | jq '.workflow_mode = true' > "$CONFIG_FILE"
```

**Check a value:**
```bash
jq -r '.workflow_mode // false' "$CLAUDE_PROJECT_DIR/.claude/agent-config.json" 2>/dev/null
```

## Available Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `workflow_mode` | bool | false | Enable Bayesian workflow stop hooks |
| `keep_nc_files` | bool | true | Retain NetCDF (.nc) files after analysis |
| `keep_csv_files` | bool | true | Retain CSV output files |
| `cleanup_temp` | bool | false | Auto-cleanup temporary/scratch files |

## Example Config

```json
{
  "workflow_mode": true,
  "keep_nc_files": true,
  "keep_csv_files": true,
  "cleanup_temp": false
}
```

## When to Use

**Enable `workflow_mode`** at the start of a full Bayesian modeling task. This activates stop hooks that verify deliverables.

**Disable `workflow_mode`** (or leave default) for simple questions, debugging, or exploratory work.

**File retention settings** control whether large output files are kept after analysis completes.

## Integration with Hooks

Hooks read from this config file. Example check in a hook:
```bash
CONFIG="$CLAUDE_PROJECT_DIR/.claude/agent-config.json"
WORKFLOW_MODE=$(jq -r '.workflow_mode // false' "$CONFIG" 2>/dev/null)
[[ "$WORKFLOW_MODE" != "true" ]] && exit 0
```
