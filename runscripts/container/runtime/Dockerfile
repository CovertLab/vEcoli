# Container image #1: wcm-runtime.
# This Dockerfile builds the runtime environment for the whole cell model.
#
# To build this image locally from the vEcoli/ project root directory:
#
#     > docker build -f cloud/docker/runtime/Dockerfile -t ${USER}-wcm-runtime .
#
# (To build using the Cloud Build service and store in the Container Registry,
# run `cloud/build.sh`.)
#
# Add option `--build-arg from=ABC` to build from a different base image "ABC"
# but DO NOT USE an alpine base since the simulation math comes out different!
# See https://pythonspeed.com/articles/alpine-docker-python/ for more reasons.
ARG from=ghcr.io/astral-sh/uv:0.5.21-python3.12-bookworm-slim@sha256:d7758d4b7176a067f7ae48239b44a8ebe9c16b00d1ec53867d41317a72e59717
FROM ${from}

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "alias ll='ls -l'" >> ~/.bashrc \
    && cp ~/.bashrc /

# Update and install in the same layer so it won't install from old updates.
# procps necessary for `ps` command used by Nextflow to track processes.
RUN apt-get update && apt-get install -y git gcc procps

# Install the project into `/vEcoli`
WORKDIR /vEcoli

# This gets better performance and more reproducible results
ENV OPENBLAS_NUM_THREADS=1

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

COPY uv.lock pyproject.toml /vEcoli/
# Install the project's dependencies using the lockfile and settings
RUN uv sync --frozen --no-install-project --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/vEcoli/.venv/bin:$PATH"

CMD ["/bin/bash"]
