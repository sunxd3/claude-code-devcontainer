# Claude Code CLI Interactive Devcontainer
#
# Base: Ubuntu 24.04 devcontainer (includes zsh, oh-my-zsh, git, common tools)
# Python: Installed via uv

ARG UV_VERSION=0.9
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install uv for Python package management
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

# Install QoL tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    fzf \
    bat \
    fd-find \
    jq \
    htop \
    ripgrep \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Ensure truecolor detection across common terminals.
COPY terminal-colors.sh /etc/profile.d/terminal-colors.sh
RUN chmod 644 /etc/profile.d/terminal-colors.sh \
    && if [ -f /etc/bash.bashrc ] && ! grep -q "terminal-colors.sh" /etc/bash.bashrc; then \
         echo ". /etc/profile.d/terminal-colors.sh" >> /etc/bash.bashrc; \
       fi \
    && if [ -f /etc/zshrc ] && ! grep -q "terminal-colors.sh" /etc/zshrc; then \
         echo ". /etc/profile.d/terminal-colors.sh" >> /etc/zshrc; \
       fi

# Switch to vscode user for remaining setup
USER vscode

# Install Python via uv
RUN uv python install 3.13

# Install Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Skip Claude Code onboarding/setup
RUN mkdir -p ~/.claude \
    && echo '{"hasCompletedOnboarding":true,"autoUpdates":false,"verbose":true}' > ~/.claude.json

# Ensure ~/.local/bin is in PATH for Claude
ENV PATH="/home/vscode/.local/bin:${PATH}"

WORKDIR /workspace
