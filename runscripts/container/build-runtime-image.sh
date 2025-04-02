#!/bin/sh
# Use Google Cloud Build, local Docker, or HPC cluster Apptainer to build
# a personalized image with uv.lock packages installed. If using Cloud Build,
# store the built image in the "vecoli" repository in Artifact Registry.
#
# ASSUMES: The current working dir is the vEcoli/ project root.

set -eu

# Cleanup function to handle all temporary files
cleanup() {
    # Remove copied files if they exist
    [ -f runscripts/container/runtime/pyproject.toml ] && rm -f runscripts/container/runtime/pyproject.toml
    [ -f runscripts/container/runtime/uv.lock ] && rm -f runscripts/container/runtime/uv.lock
    [ -f runscripts/container/runtime/.env ] && rm -f runscripts/container/runtime/.env
    echo "Cleaned up temporary files"
}

# Register cleanup on exit, interrupt, and error
trap cleanup EXIT INT TERM

RUNTIME_IMAGE="${USER}-runtime"
RUN_LOCAL=0
BUILD_APPTAINER=0

usage_str="Usage: build-runtime-image.sh [-r RUNTIME_IMAGE] [-a] [-l]
  -r: Path of built Apptainer image if -a, otherwise Docker tag
    for the runtime image to build; defaults to ${RUNTIME_IMAGE}.
  -a: Build Apptainer image (cannot use with -l).
  -l: Build image locally."

print_usage() {
  echo "$usage_str"
}

while getopts 'r:al' flag; do
  case "${flag}" in
    r)
      RUNTIME_IMAGE="${OPTARG}"
      ;;
    a)
      if [ "$RUN_LOCAL" -ne 0 ]; then
        print_usage
        exit 1
      else
        BUILD_APPTAINER=1
      fi
      ;;
    l)
      if [ "$BUILD_APPTAINER" -ne 0 ]; then
        print_usage
        exit 1
      else
        RUN_LOCAL=1
      fi
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
done

# Copy required payload files rather than using a config at
# the project root which would upload the entire project.
cp pyproject.toml runscripts/container/runtime/
cp uv.lock runscripts/container/runtime/
cp .env runscripts/container/runtime/

if [ "$RUN_LOCAL" -ne 0 ]; then
    echo "=== Locally building runtime Docker Image: ${RUNTIME_IMAGE} ==="
    docker build -f runscripts/container/runtime/Dockerfile -t "${RUNTIME_IMAGE}" .
elif [ "$BUILD_APPTAINER" -ne 0 ]; then
    echo "=== Building runtime Apptainer Image: ${RUNTIME_IMAGE} ==="
    apptainer build --force "${RUNTIME_IMAGE}" runscripts/container/runtime/Singularity
else
    echo "=== Cloud-building runtime Docker Image: ${RUNTIME_IMAGE} ==="
    # For this script to work on a Compute Engine VM, you must
    # - Set default Compute Engine region and zone for your project
    # - Set access scope to "Allow full access to all Cloud APIs" when
    #   creating VM
    # - Run gcloud init in VM
    REGION=$(gcloud config get compute/region)
    # Enable Kaniko cache to speed up build times
    gcloud config set builds/use_kaniko True
    # This needs a config file to identify the project files to upload and the
    # Dockerfile to run.
    gcloud builds submit --timeout=3h --region="$REGION" --tag \
      '${LOCATION}-docker.pkg.dev/${PROJECT_ID}/vecoli/'${RUNTIME_IMAGE} \
      runscripts/container/runtime/
fi
