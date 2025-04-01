# Container image #1: runtime image.
# This Dockerfile builds the runtime environment for the whole cell model.
#
# Use runscripts/container/build-runtime-image.sh to build.

ARG from=ghcr.io/astral-sh/uv:0.6.10-python3.12-bookworm-slim@sha256:21051ff1d154bb559756e842d8b49f6e1f12cbbf75895b818df6e5c4ab50a2e7
FROM ${from}

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "alias ll='ls -l'" >> ~/.bashrc \
    && cp ~/.bashrc /

# Update and install in the same layer so it won't install from old updates.
# procps necessary for `ps` command used by Nextflow to track processes.
RUN apt-get update && apt-get install -y git gcc procps

# Install the project into `/vEcoli`
WORKDIR /vEcoli

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

COPY uv.lock pyproject.toml .env /vEcoli/
# Install the project's dependencies using the lockfile and settings
RUN uv sync --frozen --no-install-project --no-dev

# Ensure that uv commands use the virtual environment we created
ENV UV_PROJECT_ENVIRONMENT="/vEcoli/.venv"

# Place executables in the environment at the front of the path
ENV PATH="/vEcoli/.venv/bin:$PATH"

# Create an entrypoint script that sources the .env file before running the command
RUN echo '#!/bin/bash\n\
if [ -f /vEcoli/.env ]; then\n\
  echo "Loading environment variables from /vEcoli/.env"\n\
  set -a\n\
  source /vEcoli/.env\n\
  set +a\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
