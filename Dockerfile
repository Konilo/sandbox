# Install OS and R 4.5.1
FROM rocker/r-ver:4.5.1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv and Python 3.12
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv python install 3.12
ENV PATH="/root/.local/bin:$PATH"

# Install Quarto 1.7.32
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.32/quarto-1.7.32-linux-amd64.deb && \
    dpkg -i quarto-1.7.32-linux-amd64.deb && \
    rm quarto-1.7.32-linux-amd64.deb

# Install renv (latest) and R dependencies
RUN R -e "install.packages('renv')"
COPY renv.lock renv.lock
RUN mkdir renv
COPY renv/settings.json renv/settings.json
COPY renv/activate.R renv/activate.R
COPY .Rprofile .Rprofile
RUN R -e "renv::restore()"

# Create virtual environment and sync Python dependencies
COPY pyproject.toml uv.lock .python-version ./
RUN uv venv --python 3.12 && \
    uv sync
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

# Setup git
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
ENV GIT_USER_NAME=${GIT_USER_NAME}
ENV GIT_USER_EMAIL=${GIT_USER_EMAIL}
RUN git config --global --add safe.directory /app && \
    git config --global push.autoSetupRemote true
RUN if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then \
    git config --global user.name "$GIT_USER_NAME" && \
    git config --global user.email "$GIT_USER_EMAIL"; \
    fi

COPY . /app
