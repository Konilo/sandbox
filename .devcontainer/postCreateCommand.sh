#!/bin/bash

# https://stackoverflow.com/a/19622569
trap 'exit' ERR

echo 'Running postCreateCommand.sh'

# Named volumes are created by Docker as root-owned. Fix ownership so the
# vscode user can write into them.
sudo chown -R vscode:vscode /home/vscode/.claude
sudo chown -R vscode:vscode /home/vscode/.config/gh

# TinyTeX (for Quarto PDF rendering). Installs into /home/vscode/.TinyTeX
# under the vscode user.
echo 'Installing TinyTeX'
quarto install tinytex

# Clean the renv library so stale symlinks from previous builds don't linger.
# The /renv/cache named volume (renv-cache feature) persists across rebuilds,
# so restore just re-links cached packages without re-downloading them.
echo 'Cleaning renv library'
rm -rf renv/library

echo 'Running renv::restore()'
R -e "renv::restore(prompt = FALSE)"

# Dev/exploratory R packages (not tracked by renv)
echo 'Installing dev R packages'
R -e "
renv::install('languageserver', prompt = FALSE)
install.packages('vscDebugger', repos = 'https://manuelhentschel.r-universe.dev')
"

# Python deps via uv
echo 'Running uv sync'
uv sync

# Claude Code CLI
echo 'Installing Claude Code CLI'
curl -fsSL https://claude.ai/install.sh | bash
