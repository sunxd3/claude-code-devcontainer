# Set truecolor for known-capable terminals without overriding user config.
if [ -z "${COLORTERM:-}" ]; then
  case "${TERM_PROGRAM:-}" in
    ghostty|iTerm.app|WezTerm|Apple_Terminal|vscode|Hyper|Tabby)
      export COLORTERM=truecolor
      ;;
  esac
fi

if [ -z "${COLORTERM:-}" ]; then
  case "${TERM:-}" in
    ghostty*|xterm-ghostty|xterm-kitty|alacritty|wezterm|foot|foot-256color)
      export COLORTERM=truecolor
      ;;
  esac
fi

# Prefer the available Ghostty terminfo when TERM=ghostty.
if [ "${TERM:-}" = "ghostty" ] && command -v infocmp >/dev/null 2>&1; then
  if infocmp -A /usr/local/share/ghostty-terminfo xterm-ghostty >/dev/null 2>&1 \
    || infocmp xterm-ghostty >/dev/null 2>&1; then
    export TERM=xterm-ghostty
  elif ! infocmp ghostty >/dev/null 2>&1; then
    export TERM=xterm-256color
  fi
fi
