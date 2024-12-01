#!/bin/sh
# Use Google Cloud Build or local Docker to build a personalized image with
# current state of the vEcoli repo. If using Cloud Build, store the built
# image in the "vecoli" repository in Artifact Registry.
#
# ASSUMES: The current working dir is the vEcoli/ project root.

set -eu

RUNTIME_IMAGE="${USER}-wcm-runtime"
WCM_IMAGE="${USER}-wcm-code"
RUN_LOCAL=0

usage_str="Usage: build-wcm.sh [-r RUNTIME_IMAGE] \
[-w WCM_IMAGE] [-a] [-b BIND_PATH] [-l]\n\
    -r: Docker tag of wcm-runtime image to build from; defaults to \
"$USER-wcm-runtime" (must exist in Artifact Registry).\n\
    -w: Docker tag of wcm-code image to build; defaults to "$USER-wcm-code".\n\
    -l: Build image locally.\n"

print_usage() {
  printf "$usage_str"
}

while getopts 'r:w:abl:' flag; do
  case "${flag}" in
    r) RUNTIME_IMAGE="${OPTARG}" ;;
    w) WCM_IMAGE="${OPTARG}" ;;
    l) RUN_LOCAL=1 ;;
    *) print_usage
       exit 1 ;;
  esac
done

GIT_HASH=$(git rev-parse HEAD)
GIT_BRANCH=$(git symbolic-ref --short HEAD)
TIMESTAMP=$(date '+%Y%m%d.%H%M%S')
mkdir -p source-info
git diff HEAD > source-info/git_diff.txt

if (( $RUN_LOCAL )); then
    echo "=== Locally building WCM code Docker Image ${WCM_IMAGE} on ${RUNTIME_IMAGE} ==="
    echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
    docker build -f runscripts/container/wholecell/Dockerfile -t "${WCM_IMAGE}" \
        --build-arg from="${RUNTIME_IMAGE}" \
        --build-arg git_hash="${GIT_HASH}" \
        --build-arg git_branch="${GIT_BRANCH}" \
        --build-arg timestamp="${TIMESTAMP}" .
else
    echo "=== Cloud-building WCM code Docker Image ${WCM_IMAGE} on ${RUNTIME_IMAGE} ==="
    echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
    # For this script to work on a Compute Engine VM, you must
    # - Set default Compute Engine region and zone for your project
    # - Set access scope to "Allow full access to all Cloud APIs" when
    #   creating VM
    # - Run gcloud init in VM
    REGION=$(gcloud config get compute/region)
    # This needs a config file to identify the project files to upload and the
    # Dockerfile to run.
    gcloud builds submit --timeout=15m --region=$REGION \
      --config runscripts/container/cloud_build.json \
      --substitutions="_WCM_RUNTIME=${RUNTIME_IMAGE},\
_WCM_CODE=${WCM_IMAGE},_GIT_HASH=${GIT_HASH},_GIT_BRANCH=${GIT_BRANCH},\
_TIMESTAMP=${TIMESTAMP}"
fi

rm source-info/git_diff.txt
