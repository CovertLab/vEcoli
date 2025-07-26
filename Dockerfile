ARG PLATFORM

FROM --platform=$PLATFORM python:3.12.9-slim

# Set environment variables for compilation
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CC=gcc
ENV CXX=g++

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    clang \
    gcc \
    g++ \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    zip \
    unzip \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    crossbuild-essential-arm64 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager) using official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY . .

# Pre-install problematic packages individually
RUN uv pip install --system numpy cython setuptools wheel

# Install remaining dependencies
RUN uv sync --frozen --extra dev 

# Install pre-commit hooks (optional, may fail in container)
RUN uv run pre-commit install || echo "Pre-commit install skipped"

# Install SDKMAN and Java 17 (recommended for Nextflow)
RUN curl -s "https://get.sdkman.io" | bash && \
    bash -c "source /root/.sdkman/bin/sdkman-init.sh && sdk install java 17.0.10-tem"

# Set JAVA_HOME and update PATH
ENV JAVA_HOME="/root/.sdkman/candidates/java/current"
ENV PATH="$PATH:$JAVA_HOME/bin:/root/.sdkman/bin"

# Install uv (Python package manager) using official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Nextflow using official self-installer
RUN export CAPSULE_LOG=none && \
    curl -s https://get.nextflow.io | bash && \
    chmod +x nextflow && \
    mkdir -p /root/.local/bin && \
    mv nextflow /root/.local/bin/ && \
    nextflow info

# Create .env file if it doesn't exist
RUN touch .env

# Create uvenv alias function
RUN echo '#!/bin/bash\nuv run --env-file /app/.env --project /app "$@"' > /usr/local/bin/uvenv && \
    chmod +x /usr/local/bin/uvenv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uvenv python -c "import sys; print('Python environment OK'); sys.exit(0)" || exit 1

# Default to running the test installation workflow
CMD ["uvenv", "runscripts/workflow.py", "--config", "configs/test_installation.json"]

# Labels for metadata
LABEL maintainer="vEcoli Team of Covert Lab, Stanford University"
LABEL description="Vivarium E. coli whole-cell model simulation environment"
LABEL version="1.1.0"