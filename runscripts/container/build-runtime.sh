#!/bin/sh
# Use Google Cloud Build, local Docker, or HPC cluster Apptainer to build
# a personalized image with requirements.txt installed. If using Cloud Build,
# store the built image in the "vecoli" repository in Artifact Registry.
#
# ASSUMES: The current working dir is the vEcoli/ project root.

set -eu
trap 'rm -f runscripts/container/runtime/pyproject.toml runscripts/container/runtime/uv.lock' EXIT

RUNTIME_IMAGE="${USER}-wcm-runtime"
RUN_LOCAL=0
BUILD_APPTAINER=0

usage_str="Usage: build-runtime.sh [-r RUNTIME_IMAGE] [-a] [-l]\n\
    -r: Path of built Apptainer image if -a, otherwise Docker tag \
for the wcm-runtime image to build; defaults to ${USER}-wcm-runtime\n\
    -a: Build Apptainer image (cannot use with -l).\n\
    -l: Build image locally.\n"

print_usage() {
  printf "%s" "$usage_str"
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

# Copy required payload files so rather than using a config at
# the project root which would upload the entire project.
cp pyproject.toml runscripts/container/runtime/
cp uv.lock runscripts/container/runtime/

if [ "$RUN_LOCAL" -ne 0 ]; then
    echo "=== Locally building WCM runtime Docker Image: ${RUNTIME_IMAGE} ==="
    docker build -f runscripts/container/runtime/Dockerfile -t "${RUNTIME_IMAGE}" .
elif [ "$BUILD_APPTAINER" -ne 0 ]; then
    echo "=== Building WCM runtime Apptainer Image: ${RUNTIME_IMAGE} ==="
    apptainer build --force "${RUNTIME_IMAGE}" runscripts/container/runtime/Singularity
else
    echo "=== Cloud-building WCM runtime Docker Image: ${RUNTIME_IMAGE} ==="
    # For this script to work on a Compute Engine VM, you must
    # - Set default Compute Engine region and zone for your project
    # - Set access scope to "Allow full access to all Cloud APIs" when
    #   creating VM
    # - Run gcloud init in VM
    REGION=$(gcloud config get compute/region)
    # This needs a config file to identify the project files to upload and the
    # Dockerfile to run.
    gcloud builds submit --timeout=3h --region="$REGION" --tag \
      '${LOCATION}-docker.pkg.dev/${PROJECT_ID}/vecoli/'${RUNTIME_IMAGE} \
      runscripts/container/runtime/
fi

rm runscripts/container/runtime/pyproject.toml
rm runscripts/container/runtime/uv.lock
