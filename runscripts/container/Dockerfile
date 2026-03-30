# This Dockerfile builds a container image of the whole cell model.
#
# Use runscripts/container/build-image.sh to build.

FROM ghcr.io/astral-sh/uv@sha256:a265bdb0593070eee9dfededde2e9d586917f64cfd297e290d78584e473d81e4

LABEL application="Whole Cell Model of Escherichia coli" \
    email="wholecellteam@lists.stanford.edu" \
    license="https://github.com/CovertLab/vEcoli/blob/master/LICENSE" \
    organization="Covert Lab at Stanford" \
    website="https://www.covert.stanford.edu/"

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "alias ll='ls -l'" >> ~/.bashrc \
    && cp ~/.bashrc /

# gcc necessary for compiling C extensions in some Python packages.
# procps necessary for `ps` command used by Nextflow to track processes.
# nano is a text editor for convenience.
# curl is necessary for authentication on Google Cloud VMs
# g++ greatly improves performance of GillesPy2
# ca-certificates and unzip necessary for AWS CLI installation
# AWS CLI installation necessary for Nextflow's AWS Batch executor
RUN apt-get update && apt-get install -y gcc procps nano curl g++ ca-certificates unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 for S3 access in AWS Batch (supports both x86_64 and ARM64)
RUN ARCH=$(uname -m) \
    && if [ "$ARCH" = "x86_64" ]; then \
         AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"; \
       elif [ "$ARCH" = "aarch64" ]; then \
         AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"; \
       else \
         echo "Unsupported architecture: $ARCH" && exit 1; \
       fi \
    && curl "$AWS_CLI_URL" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# Install the project into `/vEcoli`
WORKDIR /vEcoli

# Enable uv bytecode compilation to improve startup time
ENV UV_COMPILE_BYTECODE=1

# Use Docker cache mount for uv to speed up subsequent builds
ENV UV_CACHE_DIR=/root/.cache/uv

# Silence warning about cache and sync targets being on different filesystems
ENV UV_LINK_MODE=copy

# Install the dependencies only to leverage Docker layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen --no-install-project

# Activate the virtual environment
ENV PATH="/vEcoli/.venv/bin:$PATH"

# Create an entrypoint script that sources the .env file
RUN echo '#!/bin/bash\n\
if [ -f /vEcoli/.env ]; then\n\
  echo "Loading environment variables from /vEcoli/.env"\n\
  set -a\n\
  source /vEcoli/.env\n\
  set +a\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Copy the project into the image (likely to change and invalidate the cache)
ADD . /vEcoli

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen

# Record Docker Image metadata in ENV variables, viewable by `docker inspect`
# and accessible to programs in the container.
ARG git_hash=""
ARG git_branch=""
ARG timestamp=""
ENV IMAGE_GIT_HASH="$git_hash" \
	IMAGE_GIT_BRANCH="$git_branch" \
	IMAGE_TIMESTAMP="$timestamp"

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
