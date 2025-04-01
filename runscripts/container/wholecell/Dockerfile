# Container image #2: code image.
# This Dockerfile builds a container image with the vEcoli whole cell model
# code, layered on the runtime container image.
#
# Use runscripts/container/build-code-image.sh to build.

ARG from=runtime:latest
FROM ${from}

# Record Docker Image metadata in ENV variables, viewable by `docker inspect`
# and accessible to programs in the container.
ARG git_hash=""
ARG git_branch=""
ARG timestamp=""
ENV IMAGE_GIT_HASH="$git_hash" \
	IMAGE_GIT_BRANCH="$git_branch" \
	IMAGE_TIMESTAMP="$timestamp"

LABEL application="Whole Cell Model of Escherichia coli" \
    email="wholecellteam@lists.stanford.edu" \
    license="https://github.com/CovertLab/vEcoli/blob/master/LICENSE" \
    organization="Covert Lab at Stanford" \
    website="https://www.covert.stanford.edu/"

COPY . /vEcoli
WORKDIR /vEcoli

RUN uv sync --frozen

# Since this build runs as root, set permissions so running the container as
# another user will work: Parca writes into /vEcoli/cache/.
RUN (umask 000 && mkdir -p /vEcoli/cache)

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
