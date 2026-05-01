FROM rocker/r-ver:4.5.1
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    sudo \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv (latest) and Python (3.12)
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN uv python install 3.12

# Quarto 1.7.32
RUN wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.32/quarto-1.7.32-linux-amd64.deb && \
    dpkg -i quarto-1.7.32-linux-amd64.deb && \
    rm quarto-1.7.32-linux-amd64.deb

# renv (latest)
RUN R -e "install.packages('renv')"

# Project deps (renv::restore, uv sync) and TinyTeX are installed by
# .devcontainer/postCreateCommand.sh under the 'vscode' user.
